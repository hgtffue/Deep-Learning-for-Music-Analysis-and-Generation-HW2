"""
Ablation pipeline (sequential, resumable). Every ablation model uses the same architecture and
lr schedule (25k-step cosine) as the main model and is stopped at step 9000, where the main
model had its best validation loss, so it is compared with ckpt/basic/best_val (step 9000).
Sampling for task a ablations: T=1.0, top-p 0.95 (= config a1).

  base           REMI + chord, main model @ 9k
  no_constraint  same model, constrained decoding switched off
  nochord        REMI without Chord tokens
  midilike       MIDI-like events (NoteOn/NoteOff/TimeShift/Vel), no chord
  noaug          REMI + chord, no transposition augmentation
  bonus_nochord  bonus model whose condition has no chord tokens (melody only)
"""
import json
import os
import pathlib
import subprocess
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", PYTHONUNBUFFERED="1")
PY = sys.executable
OUT = ROOT / "outputs/ablation"
CK = ROOT / "ckpt"


def run(*args, cwd=ROOT / "code"):
    print(">>", " ".join(map(str, args)), flush=True)
    subprocess.run([PY, *map(str, args)], cwd=cwd, env=env, check=True)


def train(name, data, *extra, steps=25000):
    if not (CK / name / "last").exists():
        run("train.py", "--data", ROOT / "data" / data, "--out", CK / name, "--batch", 8,
            "--steps", steps, "--stop_at", 9000, "--save_losses", "", *extra)


def gen(name, ckpt, *extra):
    if not (OUT / name / "tokens.json").exists():
        run("ablation_gen.py", "--ckpt", ckpt, "--out", OUT / name, *extra)


gen("base", CK / "basic/best_val")
gen("no_constraint", CK / "basic/best_val", "--no_constraint")
train("abl_nochord", "basic_nochord.json")
gen("nochord", CK / "abl_nochord/last", "--no_chord")
train("abl_midilike", "basic_midilike.json", "--vocab", "midilike")
gen("midilike", CK / "abl_midilike/last", "--midilike")
train("abl_noaug", "basic_sequences.json", "--no_aug")
gen("noaug", CK / "abl_noaug/last")
train("abl_bonus_nochord", "bonus_nochord.json", steps=12000)
bd = OUT / "bonus_nochord"
if not (bd / "metrics.csv").exists():
    run("generate.py", "bonus", "--ckpt", CK / "abl_bonus_nochord/last", "--out", bd,
        "--temp", 0.8, "--top_k", 10, "--no_chord")
    run("eval_bonus.py", "--prompt_dir", ROOT / "POP909/prompts/bonus", "--gen_midi_dir", bd,
        "--output_csv", bd / "metrics.csv", cwd=ROOT / "HW2_scripts")

# metrics
def run_eval(seq_path, csv_path):
    if not csv_path.exists():
        run("eval_basic.py", "--tok2idx_path", ROOT / "data/tok2idx.json", "--sequences_path", seq_path,
            "--output_csv", csv_path, cwd=ROOT / "HW2_scripts")
    return pd.read_csv(csv_path).iloc[:, 1:].mean()


rows = []
for name in ["base", "no_constraint", "nochord", "midilike", "noaug"]:
    m = run_eval(OUT / name / "tokens.json", OUT / name / "metrics.csv")
    rows.append({"ablation": name, **m.to_dict()})
pd.DataFrame(rows).to_csv(ROOT / "results/ablation_summary.csv", index=False)
print(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"))
print("ABLATION DONE", flush=True)
