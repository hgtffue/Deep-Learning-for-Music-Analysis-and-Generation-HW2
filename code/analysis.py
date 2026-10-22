"""
Extra analyses for the report (CPU only).

  1. memorization: how many k-bar windows of the generated pieces appear verbatim (up to
     transposition) in the training set, compared with unseen real songs
  2. task b: does the continuation stay consistent with the prompt (pitch classes, key, density, tempo)
  3. figures: loss curves, task a metrics

python analysis.py [memo] [taskb] [figs]
"""
import json
import pathlib
import random
import sys

import numpy as np
import pandas as pd

import midilike as ml
import music_tok as mt

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figs"


def split():
    data = json.load(open(ROOT / "data/basic_sequences.json"))
    random.Random(1).shuffle(data)  # same split as train.py
    return data[16:], data[:16]


def bars_of(tokens):
    """list of bars, each a list of (pos, pitch)"""
    bars, cur, pos = [], None, 0
    for t in tokens:
        if t == "Bar":
            cur = []
            bars.append(cur)
        elif t.startswith("Pos_"):
            pos = int(t[4:])
        elif t.startswith("Pitch_") and cur is not None:
            cur.append((pos, int(t[6:])))
    return bars


def note_stream(tokens, n_bars=None, skyline=False):
    """notes sorted by (onset, pitch) -> transposition-invariant steps (d_onset, d_pitch)
    skyline: keep only the highest note of every onset (a rough melody line)"""
    notes = sorted((on, p) for on, p, _, _ in ml.remi_to_notes(tokens))
    if skyline:
        top = {}
        for on, p in notes:
            top[on] = max(p, top.get(on, 0))
        notes = sorted(top.items())
    if n_bars is not None:
        cut = _bar_cut(tokens, n_bars)
        notes = [n for n in notes if n[0] < cut]
    return [(b[0] - a[0], b[1] - a[1]) for a, b in zip(notes, notes[1:])]


def _bar_cut(tokens, n_bars):
    t, k, beats = 0, 0, 4
    for tok in tokens:
        if tok == "Bar":
            if k:
                t += beats * mt.POS_PER_BEAT
            k += 1
            beats = 4
            if k > n_bars:
                return t
        elif tok.startswith("TS_"):
            beats = int(tok[3:])
    return 10 ** 9


NS = (10, 20, 40)  # ~ half a bar / one bar / two bars of the merged texture (about 20 notes per bar)
NS_MEL = (8, 16)    # melody (skyline) n-grams: roughly 1 and 2 bars


def ngrams(steps, n, min_distinct=4):
    """hashes of all n-grams; trivial ones (e.g. one note repeated) become None and are ignored"""
    return [hash(w) if len(set(w)) >= min_distinct else None
            for w in (tuple(steps[i:i + n]) for i in range(len(steps) - n + 1))]


def longest_run(hits, n):
    best = cur = 0
    for x in hits:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best + n - 1 if best else 0


def memorization():
    train, val = split()
    sets = {("all", n): set() for n in NS} | {("mel", n): set() for n in NS_MEL}
    for s in train:
        st = {"all": note_stream(s["tokens"]), "mel": note_stream(s["tokens"], skyline=True)}
        for kind, n in sets:
            sets[kind, n].update(h for h in ngrams(st[kind], n) if h is not None)

    def score(pieces):
        r = {}
        streams = {"all": [note_stream(t, 32) for t in pieces], "mel": [note_stream(t, 32, True) for t in pieces]}
        for kind, n in sets:
            g = [h for st in streams[kind] for h in ngrams(st, n) if h is not None]
            r[f"{kind} {n}-gram copied"] = float(np.mean([h in sets[kind, n] for h in g])) if g else 0.0
        for kind, n in (("all", NS[0]), ("mel", NS_MEL[0])):
            runs = [longest_run([h in sets[kind, n] for h in ngrams(st, n)], n) for st in streams[kind]]
            r[f"{kind} longest copy mean"] = float(np.mean(runs))
            r[f"{kind} longest copy max"] = int(np.max(runs))
        return r

    rows = []
    groups = [(d.name, d / "tokens.json") for d in sorted((ROOT / "outputs/task_a").iterdir())]
    groups += [(f"ablation/{d.name}", d / "tokens.json") for d in sorted((ROOT / "outputs/ablation").glob("*"))
               if (d / "tokens.json").exists() and "bonus" not in d.name]
    for name, f in groups:
        rows.append({"group": name, **score([s["tokens"] for s in json.load(open(f))])})
    rows.append({"group": "real (16 held-out songs, first 32 bars)", **score([s["tokens"] for s in val])})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "memorization.csv", index=False)
    print(df.to_markdown(index=False, floatfmt=".3f"))


# ------------------------------------------------------------------ task b
KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def pc_hist(bars):
    h = np.zeros(12)
    for b in bars:
        for _, p in b:
            h[p % 12] += 1
    return h


def key_of(h):
    best = max(((np.corrcoef(np.roll(prof, r), h)[0, 1], r, q) for q, prof in (("maj", KS_MAJ), ("min", KS_MIN))
                for r in range(12)))
    return f"{mt.ROOT_NAMES[best[1]]}:{best[2]}"


def tempos(tokens):
    return [int(t[6:]) for t in tokens if t.startswith("Tempo_")]


def consistency(tokens, n_prompt=8):
    bars = bars_of(tokens)
    a, b = bars[:n_prompt], bars[n_prompt:32]
    ha, hb = pc_hist(a), pc_hist(b)
    tp = tempos(tokens)
    dens_a = np.mean([len(x) for x in a]); dens_b = np.mean([len(x) for x in b])
    pcs = np.array([[pc_hist([x])] for x in b]).reshape(len(b), 12)
    return {"pc_cosine": float(ha @ hb / (np.linalg.norm(ha) * np.linalg.norm(hb) + 1e-9)),
            "key_prompt": key_of(ha), "key_cont": key_of(hb), "same_key": key_of(ha) == key_of(hb),
            "notes/bar prompt": dens_a, "notes/bar cont": dens_b, "density_ratio": dens_b / max(dens_a, 1e-9),
            "tempo prompt": float(np.mean(tp[:n_prompt])), "tempo cont": float(np.mean(tp[n_prompt:32])),
            "empty_bars_cont": int(sum(len(x) == 0 for x in b)),
            "pc_entropy_cont": float(np.mean([_ent(r) for r in pcs if r.sum()]))}


def _ent(h):
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def taskb():
    rows = []
    for d in sorted((ROOT / "outputs/task_b").iterdir()):
        for s in json.load(open(d / "tokens.json")):
            rows.append({"config": d.name, "prompt": s["song_name"], **consistency(s["tokens"], s["prompt_bars"])})
    _, val = split()
    ref = pd.DataFrame([consistency(s["tokens"]) for s in val])
    df = pd.DataFrame(rows)
    df.to_csv(RES / "task_b_consistency.csv", index=False)
    cols = ["config", "prompt", "pc_cosine", "key_prompt", "key_cont", "notes/bar prompt", "notes/bar cont",
            "tempo prompt", "tempo cont", "empty_bars_cont", "pc_entropy_cont"]
    print(df[cols].to_markdown(index=False, floatfmt=".3f"))
    print("\nreal songs (16 held-out, bars 1-8 vs 9-32): pc_cosine %.3f (min %.3f), same key %.0f%%, density ratio %.2f, pc_entropy %.3f"
          % (ref.pc_cosine.mean(), ref.pc_cosine.min(), 100 * ref.same_key.mean(), ref.density_ratio.mean(),
             ref.pc_entropy_cont.mean()))
    ref.to_csv(RES / "task_b_consistency_real.csv", index=False)


# ------------------------------------------------------------------ figures
def read_log(path):
    tr, va = {}, {}
    for line in open(path):
        p = line.split()
        if line.startswith("step "):
            tr[int(p[1])] = float(p[3])
        elif line.startswith("== step"):
            va[int(p[2])] = float(p[4])
    return tr, va


def figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, (name, d) in zip(ax, [("basic (task a/b)", "basic"), ("bonus", "bonus")]):
        tr, va = read_log(ROOT / "ckpt" / d / "log.txt")
        a.plot(list(tr), list(tr.values()), label="train (EMA)", lw=1)
        a.plot(list(va), list(va.values()), "o-", label="validation", ms=3)
        bs = min(va, key=va.get)
        a.axvline(bs, ls="--", c="gray", lw=0.8)
        a.annotate(f"best val {va[bs]:.3f} @ {bs}", (bs, va[bs]), xytext=(5, 25), textcoords="offset points")
        a.set_title(name); a.set_xlabel("step"); a.set_ylabel("loss"); a.set_ylim(0.6, 2.5); a.legend()
    for lbl, d in [("REMI no chord", "abl_nochord"), ("MIDI-like", "abl_midilike"), ("no aug", "abl_noaug")]:
        if (ROOT / "ckpt" / d / "log.txt").exists():
            _, va = read_log(ROOT / "ckpt" / d / "log.txt")
            ax[0].plot(list(va), list(va.values()), "x:", ms=3, lw=0.8, label=f"val, {lbl}")
    ax[0].legend(fontsize=7)
    if (ROOT / "ckpt/abl_bonus_nochord/log.txt").exists():
        _, va = read_log(ROOT / "ckpt/abl_bonus_nochord/log.txt")
        ax[1].plot(list(va), list(va.values()), "x:", ms=3, lw=0.8, label="val, melody only (no chord)")
        ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "loss_curves.png", dpi=150); plt.close(fig)

    s = pd.read_csv(RES / "task_a_summary.csv")
    si = pd.read_csv(RES / "task_a_si.csv").set_index("group")
    s["key"] = s.Config.str.split("_").str[0]
    gen = s[s.key.str.match(r"a\d")]
    real = s[s.Config.str.contains("first 32")].iloc[0]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.2))
    for a, m in zip(ax[:3], ["H1", "H4", "GS"]):
        a.bar(gen.key, gen[m], color="#7aa6c2")
        a.axhline(real[m], c="crimson", ls="--", label="real (first 32 bars)")
        lo = min(gen[m].min(), real[m]); hi = max(gen[m].max(), real[m])
        a.set_ylim(lo - (hi - lo) * 0.6, hi + (hi - lo) * 0.4); a.set_title(m)
    ax[0].legend(fontsize=7)
    x = np.arange(len(si))
    for j, c in enumerate(["SI_short", "SI_mid", "SI_long"]):
        ax[3].bar(x + (j - 1) * 0.27, si[c], 0.27, label=c)
    ax[3].set_xticks(x); ax[3].set_xticklabels(si.index); ax[3].set_title("SI"); ax[3].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "task_a_metrics.png", dpi=150); plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["memo", "taskb", "figs"]
    if "memo" in todo:
        memorization()
    if "taskb" in todo:
        taskb()
    if "figs" in todo:
        figs()
