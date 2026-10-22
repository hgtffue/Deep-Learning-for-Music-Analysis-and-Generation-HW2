"""
Inference for the three tasks.

Unconditional (task a):   python generate.py uncond   --ckpt ../ckpt/basic/last --out ../outputs/task_a/<cfg> --n 20 --temp 1.0 --top_p 0.95
Continuation (task b):    python generate.py continue --ckpt ../ckpt/basic/last --out ../outputs/task_b/<cfg> --temp 1.0 --top_k 10
Accompaniment (bonus):    python generate.py bonus    --ckpt ../ckpt/bonus/last --out ../outputs/bonus/<cfg> --temp 1.0 --top_p 0.9
"""
import argparse
import json
import os
import pathlib
import random

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pretty_midi
import torch
from transformers import GPT2LMHeadModel

import music_tok as mt

T = mt.TOK2IDX
V = mt.VOCAB_SIZE
ROOT = pathlib.Path(__file__).resolve().parent.parent


def group_ids(prefix):
    return [i for t, i in T.items() if t.startswith(prefix)]


G = {k: group_ids(k) for k in ["Pitch_", "Vel_", "Dur_", "TS_", "Tempo_", "Pos_", "Chord_"]}


DEV = "cuda"


def mask_of(ids):
    m = torch.zeros(V, dtype=torch.bool, device=DEV)
    m[ids] = True
    return m


# token-type grammar: which tokens may follow the previous token
M_AFTER_PITCH = mask_of(G["Vel_"])
M_AFTER_VEL = mask_of(G["Dur_"])
M_AFTER_BAR = mask_of(G["TS_"])
M_AFTER_TS = mask_of(G["Tempo_"])
M_GENERAL_BASIC = mask_of([T["Bar"], T["<EOS>"]] + G["Pos_"] + G["Chord_"] + G["Pitch_"])
M_GENERAL_PIANO = mask_of([T["Bar"]] + G["Pos_"] + G["Pitch_"])
M_AFTER_POS_BASIC = mask_of(G["Pitch_"] + G["Chord_"])
M_AFTER_POS_PIANO = mask_of(G["Pitch_"])
M_ONLY_BAR = mask_of([T["Bar"]])


POS0 = T["Pos_0"]


def next_mask(ids, piano_section=False):
    tok = mt.IDX2TOK[ids[-1]]
    if tok.startswith("Pitch_"):
        return M_AFTER_PITCH
    if tok.startswith("Vel_"):
        return M_AFTER_VEL
    if tok == "Bar":
        return M_AFTER_BAR
    if tok.startswith("TS_"):
        return M_AFTER_TS
    if tok == "<BOS>":
        return M_ONLY_BAR
    if tok.startswith("Pos_"):  # a position must be followed by a note (or a chord)
        return M_AFTER_POS_PIANO if piano_section else M_AFTER_POS_BASIC
    m = (M_GENERAL_PIANO if piano_section else M_GENERAL_BASIC).clone()
    # positions inside a bar must strictly increase and stay inside the bar
    beats, pos = section_state(ids)
    m[POS0: POS0 + pos + 1] = False
    m[POS0 + beats * mt.POS_PER_BEAT: POS0 + mt.MAX_POS] = False
    return m


def section_state(ids):
    """beats of the current bar and the last Pos used since the latest Bar/Track token"""
    beats, pos, closed = 4, -1, False
    for i in reversed(ids):
        tok = mt.IDX2TOK[i]
        if tok.startswith("Pos_") and not closed and pos < 0:
            pos = int(tok[4:])
        elif tok in ("Track_Mel", "Track_Pno"):
            closed = True
        elif tok.startswith("TS_"):
            beats = int(tok[3:])
            break
        elif tok == "Bar":
            break
    return beats, pos


def sample_logits(logits, temp, top_k, top_p):
    logits = logits / max(temp, 1e-5)
    if top_k and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.numel()))
        logits[logits < v[-1]] = -float("inf")
    probs = torch.softmax(logits, -1)
    if top_p and top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, -1)
        cut = cum - sp > top_p
        sp[cut] = 0
        probs = torch.zeros_like(probs).scatter_(0, si, sp)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1))


class Generator:
    def __init__(self, ckpt, device="cuda"):
        self.model = GPT2LMHeadModel.from_pretrained(ckpt).to(device).eval()
        self.ctx = self.model.config.n_positions
        self.dev = device
        self.reset([T["<BOS>"]])

    def reset(self, ids):
        self.ids = list(ids)
        self.past = None
        self.fed = 0   # number of tokens of the current window already in the KV cache
        self.win_start = 0

    def _window(self):
        """slide the context window, keeping the last ~half, starting at a Bar token"""
        keep = self.ctx // 2
        cand = len(self.ids) - keep
        bars = [i for i in range(cand, len(self.ids)) if self.ids[i] == T["Bar"]]
        self.win_start = bars[0] if bars else cand
        self.past, self.fed = None, 0

    @torch.no_grad()
    def logits(self):
        """logits for the next token given self.ids"""
        if len(self.ids) - self.win_start >= self.ctx:
            self._window()
        window = self.ids[self.win_start:]
        new = window[self.fed:]
        x = torch.tensor([new], device=self.dev)
        pos = torch.arange(self.fed, self.fed + len(new), device=self.dev)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(input_ids=x, past_key_values=self.past, position_ids=pos, use_cache=True)
        self.past = out.past_key_values
        self.fed = len(window)
        return out.logits[0, -1].float()

    def step(self, temp, top_k, top_p, mask):
        lg = self.logits()
        lg[~mask] = -float("inf")
        t = sample_logits(lg, temp, top_k, top_p)
        self.ids.append(t)
        return t


def count_bars(ids):
    return sum(1 for i in ids if i == T["Bar"])


def generate_basic(gen, prefix_ids, n_bars_total, temp, top_k, top_p, max_tokens=8000):
    """continue prefix until it contains n_bars_total bars"""
    gen.reset(prefix_ids)
    bars = count_bars(prefix_ids)
    for _ in range(max_tokens):
        mask = next_mask(gen.ids)
        if bars < n_bars_total and mask[T["<EOS>"]]:
            mask = mask.clone()
            mask[T["<EOS>"]] = False
        # peek: if the model would open bar n_bars_total+1 we stop
        t = gen.step(temp, top_k, top_p, mask)
        if t == T["Bar"]:
            bars += 1
            if bars > n_bars_total:
                gen.ids.pop()
                break
        if t == T["<EOS>"]:
            gen.ids.pop()
            break
    return [mt.IDX2TOK[i] for i in gen.ids]


def generate_bonus(gen, bar_blocks, temp, top_k, top_p, max_tokens_per_bar=600):
    """
    bar_blocks: list (one per bar) of forced condition tokens
                [Bar, TS, Tempo, Track_Mel, ...melody & chords..., Track_Pno]
    The model only writes the piano part; whenever it emits `Bar`, the next condition block is inserted.
    """
    gen.reset([T["<BOS>"]] + [T[t] for t in bar_blocks[0]])
    for k in range(len(bar_blocks)):
        for _ in range(max_tokens_per_bar):
            mask = next_mask(gen.ids, piano_section=True)
            t = gen.step(temp, top_k, top_p, mask)
            if t == T["Bar"]:
                gen.ids.pop()
                break
        if k + 1 < len(bar_blocks):
            gen.ids += [T[t] for t in bar_blocks[k + 1]]
    return [mt.IDX2TOK[i] for i in gen.ids]


def midi_tracks(path):
    m = pretty_midi.PrettyMIDI(str(path))
    return {i.name.strip().upper(): i.notes for i in m.instruments}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["uncond", "continue", "bonus"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--bars", type=int, default=32)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt_dir", default=None)
    ap.add_argument("--no_chord", action="store_true", help="bonus: drop chord tokens from the condition")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "config.json", "w"), indent=1)
    gen = Generator(args.ckpt)
    results = []

    if args.mode == "uncond":
        for k in range(args.n):
            toks = generate_basic(gen, [T["<BOS>"]], args.bars, args.temp, args.top_k, args.top_p)
            name = f"uncond_{k:02d}"
            mt.decode(toks).write(str(out / f"{name}.mid"))
            results.append({"song_name": name, "tokens": toks})
            print(name, len(toks), "tokens", count_bars([T[t] for t in toks]), "bars", flush=True)

    elif args.mode == "continue":
        pdir = pathlib.Path(args.prompt_dir or ROOT / "POP909/prompts/basic-b")
        for d in sorted(p for p in pdir.iterdir() if p.is_dir()):
            tracks = midi_tracks(d / f"{d.name}.mid")
            song = mt.parse_song(tracks, d / "beat_midi.txt", d / "chord_midi.txt")
            prompt = mt.encode_basic(song)
            n_prompt = count_bars([T[t] for t in prompt])
            prompt_ids = [T[t] for t in prompt]
            toks = generate_basic(gen, prompt_ids, n_prompt + 24, args.temp, args.top_k, args.top_p)
            mt.decode(toks).write(str(out / f"{d.name}.mid"))
            results.append({"song_name": d.name, "tokens": toks, "prompt_bars": n_prompt})
            print(d.name, "prompt bars", n_prompt, "total tokens", len(toks), flush=True)

    else:
        pdir = pathlib.Path(args.prompt_dir or ROOT / "POP909/prompts/bonus")
        for d in sorted(p for p in pdir.iterdir() if p.is_dir()):
            mel = midi_tracks(d / "melody.mid")
            mel_notes = mel.get("MELODY", next(iter(mel.values())))
            song = mt.parse_song({"MELODY": mel_notes, "PIANO": []}, d / "beat_midi.txt", d / "chord_midi.txt")
            blocks = [mt.encode_bonus_bar(b, with_piano=False) for b in song["bars"]]
            if args.no_chord:  # ablation: melody-only condition (also drop Pos left without notes)
                blocks = [[t for i, t in enumerate(bl) if not t.startswith("Chord_") and not (
                    t.startswith("Pos_") and not any(x.startswith("Pitch_") for x in bl[i + 1:i + 3]))]
                    for bl in blocks]
            toks = generate_bonus(gen, blocks, args.temp, args.top_k, args.top_p)
            piano = mt.decode(toks, grid=song["grid"]).instruments[1]
            pm = pretty_midi.PrettyMIDI(initial_tempo=120)
            melody = pretty_midi.Instrument(0, name="MELODY"); melody.notes = list(mel_notes)
            pm.instruments = [melody, piano]
            pm.write(str(out / f"{d.name}.mid"))
            results.append({"song_name": d.name, "tokens": toks})
            print(d.name, len(song["bars"]), "bars,", len(piano.notes), "piano notes", flush=True)

    json.dump(results, open(out / "tokens.json", "w"))


if __name__ == "__main__":
    main()
