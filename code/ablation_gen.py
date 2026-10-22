"""
Unconditional 32-bar generation for the ablation models.

python ablation_gen.py --ckpt ../ckpt/abl_nochord/last --out ../outputs/ablation/nochord --no_chord
python ablation_gen.py --ckpt ../ckpt/basic/best_val  --out ../outputs/ablation/no_constraint --no_constraint
python ablation_gen.py --ckpt ../ckpt/abl_midilike/last --out ../outputs/ablation/midilike --midilike

tokens.json is always written in REMI form (MIDI-like output is converted, 4/4 bars) so that
eval_basic.py can be used unchanged; the raw model output goes to raw_tokens.json.
"""
import argparse
import json
import pathlib
import random

import numpy as np
import torch

import generate as g
import midilike as ml
import music_tok as mt

T = mt.TOK2IDX


def grammar_ok(ids):
    """fraction of tokens that the constrained decoder would have allowed"""
    ok = sum(bool(g.next_mask(ids[:i])[ids[i]]) for i in range(1, len(ids)))
    return ok / max(1, len(ids) - 1)


def gen_remi(gen, n_bars, temp, top_k, top_p, constrained=True, no_chord=False, max_tokens=8000):
    gen.reset([T["<BOS>"]])
    bars = 0
    free = torch.ones(mt.VOCAB_SIZE, dtype=torch.bool, device=g.DEV)
    free[[T["<PAD>"], T["<BOS>"], T["Track_Mel"], T["Track_Pno"]]] = False
    for _ in range(max_tokens):
        mask = (g.next_mask(gen.ids) if constrained else free).clone()
        if no_chord:
            mask[g.G["Chord_"]] = False
        if bars < n_bars:
            mask[T["<EOS>"]] = False
        t = gen.step(temp, top_k, top_p, mask)
        if t == T["Bar"]:
            bars += 1
            if bars > n_bars:
                gen.ids.pop()
                break
        if t == T["<EOS>"]:
            gen.ids.pop()
            break
    return list(gen.ids)


def gen_midilike(gen, n_bars, temp, top_k, top_p, max_tokens=10000):
    L = ml.TOK2IDX
    gen.reset([L["<BOS>"]])
    mask = torch.ones(ml.VOCAB_SIZE, dtype=torch.bool, device=g.DEV)
    mask[[L["<PAD>"], L["<BOS>"], L["<EOS>"]]] = False
    end, t_now = n_bars * 4 * mt.POS_PER_BEAT, 0
    for _ in range(max_tokens):
        t = gen.step(temp, top_k, top_p, mask)
        tok = ml.IDX2TOK[t]
        if tok.startswith("TimeShift_"):
            t_now += int(tok[10:])
            if t_now >= end:
                break
    return [ml.IDX2TOK[i] for i in gen.ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--bars", type=int, default=32)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_chord", action="store_true")
    ap.add_argument("--no_constraint", action="store_true")
    ap.add_argument("--midilike", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "config.json", "w"), indent=1)
    gen = g.Generator(args.ckpt)
    results, raw, stats = [], [], []
    for k in range(args.n):
        name = f"uncond_{k:02d}"
        if args.midilike:
            mtoks = gen_midilike(gen, args.bars, args.temp, args.top_k, args.top_p)
            notes = ml.decode(mtoks)
            toks = ml.notes_to_remi(notes, args.bars, tempo=80)
            n_on = sum(t.startswith("NoteOn_") for t in mtoks)
            n_off = sum(t.startswith("NoteOff_") for t in mtoks)
            raw.append({"song_name": name, "tokens": mtoks})
            stats.append({"song_name": name, "tokens": len(mtoks), "note_on": n_on, "note_off": n_off})
        else:
            ids = gen_remi(gen, args.bars, args.temp, args.top_k, args.top_p,
                           constrained=not args.no_constraint, no_chord=args.no_chord)
            toks = [mt.IDX2TOK[i] for i in ids]
            stats.append({"song_name": name, "tokens": len(ids), "bars": g.count_bars(ids),
                          "grammar_ok": grammar_ok(ids)})
        mt.decode(toks).write(str(out / f"{name}.mid"))
        results.append({"song_name": name, "tokens": toks})
        print(name, stats[-1], flush=True)
    json.dump(results, open(out / "tokens.json", "w"))
    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    if raw:
        json.dump(raw, open(out / "raw_tokens.json", "w"))


if __name__ == "__main__":
    main()
