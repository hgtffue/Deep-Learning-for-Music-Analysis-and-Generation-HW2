"""
Train a decoder-only Transformer (GPT-2) on POP909 token sequences.

python train.py --data ../data/basic_sequences.json --out ../ckpt/basic
python train.py --data ../data/bonus_sequences.json --out ../ckpt/bonus
"""
import argparse
import json
import math
import os
import pathlib
import random
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import music_tok as mt
import midilike


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=1536)
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--n_embd", type=int, default=512)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--n_val", type=int, default=16, help="songs held out to monitor loss")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_losses", type=str, default="1.2,0.9,0.7",
                    help="also keep a snapshot the first time train loss drops below these")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop_at", type=int, default=0, help="stop early (lr schedule still uses --steps)")
    ap.add_argument("--no_aug", action="store_true", help="disable transposition augmentation")
    ap.add_argument("--vocab", choices=["remi", "midilike"], default="remi")
    return ap.parse_args()


class Transposer:
    """Pitch / chord-root transposition on token ids (data augmentation)."""

    def __init__(self):
        V = mt.VOCAB_SIZE
        self.maps = {}
        for s in range(-6, 7):
            m = np.arange(V)
            for tok, i in mt.TOK2IDX.items():
                head, _, num = tok.rpartition("_")
                if head in ("Pitch", "NoteOn", "NoteOff"):
                    m[i] = mt.TOK2IDX.get(f"{head}_{int(num) + s}", -1)
                elif tok.startswith("Chord_") and tok != "Chord_N":
                    _, r, q = tok.split("_", 2)
                    r2 = mt.ROOT_NAMES[(mt.ROOT_NAMES.index(r) + s) % 12]
                    m[i] = mt.TOK2IDX[f"Chord_{r2}_{q}"]
            self.maps[s] = m

    def __call__(self, ids):
        for _ in range(5):
            s = random.randint(-6, 6)
            out = self.maps[s][ids]
            if (out >= 0).all():
                return out
        return ids


class Sampler:
    def __init__(self, seqs, ctx, augment):
        self.seqs = [np.array([mt.TOK2IDX[t] for t in s], dtype=np.int64) for s in seqs]
        self.bars = [np.flatnonzero(s == mt.TOK2IDX.get("Bar", -1)) for s in self.seqs]
        self.lens = np.array([len(s) for s in self.seqs], dtype=float)
        self.ctx = ctx
        self.tr = Transposer() if augment else None

    def sample(self):
        k = np.random.choice(len(self.seqs), p=self.lens / self.lens.sum())
        s, bars = self.seqs[k], self.bars[k]
        # start at beginning of song (with BOS) or at a random bar line
        if random.random() < 0.1:
            st = 0
        elif len(bars) == 0:  # representation without Bar tokens (MIDI-like): any position
            st = random.randrange(max(1, len(s) - self.ctx // 4))
        else:
            st = int(random.choice(bars[:max(1, np.searchsorted(bars, len(s) - self.ctx // 4))]))
        x = s[st: st + self.ctx + 1]
        if self.tr is not None:
            x = self.tr(x)
        if st + self.ctx + 1 >= len(s) or len(x) < self.ctx + 1:  # append EOS at song end
            x = np.concatenate([x, [mt.TOK2IDX["<EOS>"]]])[: self.ctx + 1]
        pad = self.ctx + 1 - len(x)
        y = x.copy()
        if pad > 0:
            x = np.concatenate([x, np.zeros(pad, dtype=np.int64)])
            y = np.concatenate([y, np.full(pad, -100, dtype=np.int64)])
        # GPT2LMHeadModel shifts labels internally, so labels are aligned with inputs
        return x[:-1], y[:-1]

    def batch(self, n):
        xs, ys = zip(*[self.sample() for _ in range(n)])
        return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))


def main():
    args = parse_args()
    if args.vocab == "midilike":  # swap the vocabulary used everywhere in this module
        mt.TOK2IDX, mt.VOCAB_SIZE = midilike.TOK2IDX, midilike.VOCAB_SIZE
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=1)

    data = json.load(open(args.data))
    random.Random(1).shuffle(data)
    val, train = data[: args.n_val], data[args.n_val:]
    tr = Sampler([d["tokens"] for d in train], args.ctx, augment=not args.no_aug)
    va = Sampler([d["tokens"] for d in val], args.ctx, augment=False)

    cfg = GPT2Config(vocab_size=mt.VOCAB_SIZE, n_positions=args.ctx, n_embd=args.n_embd,
                     n_layer=args.n_layer, n_head=args.n_head, resid_pdrop=args.dropout,
                     embd_pdrop=args.dropout, attn_pdrop=args.dropout,
                     bos_token_id=mt.TOK2IDX["<BOS>"], eos_token_id=mt.TOK2IDX["<EOS>"],
                     pad_token_id=mt.TOK2IDX["<PAD>"])
    dev = "cuda"
    model = GPT2LMHeadModel(cfg).to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))) * 0.95 + 0.05)

    thresholds = sorted([float(x) for x in args.save_losses.split(",") if x], reverse=True)
    log = open(out / "log.txt", "a")
    ema, t0, best_val = None, time.time(), 1e9
    for step in range(1, args.steps + 1):
        model.train()
        x, y = tr.batch(args.batch)
        x, y = x.to(dev), y.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=y).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        ema = loss.item() if ema is None else 0.99 * ema + 0.01 * loss.item()

        while thresholds and ema < thresholds[0] and step > 2000:
            th = thresholds.pop(0)
            model.save_pretrained(out / f"loss_{th}")
            print(f"saved snapshot at train loss < {th} (step {step})")
            log.write(f"snapshot loss_{th} step {step}\n")

        if step % 100 == 0:
            msg = f"step {step} loss {ema:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vl = np.mean([model(input_ids=a.to(dev), labels=b.to(dev)).loss.item()
                              for a, b in (va.batch(args.batch) for _ in range(8))])
            msg = f"== step {step} val_loss {vl:.4f} train_loss {ema:.4f}"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            model.save_pretrained(out / "last")
            if vl < best_val:
                best_val = vl
                model.save_pretrained(out / "best_val")
        if args.stop_at and step >= args.stop_at:
            model.save_pretrained(out / "last")
            break


if __name__ == "__main__":
    main()
