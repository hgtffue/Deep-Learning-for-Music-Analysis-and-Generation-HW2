"""
Structureness Indicator (SI) for task a.
  1) render every task-a config + 20 random real-data pieces (first 32 bars) to wav
  2) compute fitness scape plots with MusDr/run_python_scapeplot.py (parallel)
  3) SI_short (3-8s), SI_mid (8-15s), SI_long (>15s) averaged per config
"""
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from render import render_midi  # must come first: sets up the FluidSynth DLL before pretty_midi is imported
import music_tok as mt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
MUSDR = ROOT / "HW2_scripts/MusDr"
WORK = ROOT / "results/si"
env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def render(mid, wav):
    if not wav.exists():
        render_midi(mid, wav)


audio = WORK / "audio"
for d in ["audio", "ssm", "scplot", "real_mid"]:
    (WORK / d).mkdir(parents=True, exist_ok=True)

# generated pieces
for cfg in sorted((ROOT / "outputs/task_a").iterdir()):
    for mid in sorted(cfg.glob("*.mid")):
        dst = audio / f"{cfg.name.split('_')[0]}__{mid.stem}.wav"
        render(mid, dst)  # all SI audio rendered with the same renderer

# real data reference: 20 random songs, first 32 bars, same decoding pipeline
real = json.load(open(ROOT / "data/basic_sequences_32bars.json"))
for s in random.Random(0).sample(real, 20):
    mid = WORK / "real_mid" / f"{s['song_name']}.mid"
    mt.decode(s["tokens"]).write(str(mid))
    render(mid, audio / f"real__{s['song_name']}.wav")

# skip files whose scape plot already exists (resumable)
todo = WORK / "todo"
shutil.rmtree(todo, ignore_errors=True); todo.mkdir()
for w in audio.glob("*.wav"):
    if not (WORK / "scplot" / f"{w.stem}_fitness.npy").exists():
        shutil.copy(w, todo / w.name)
print(f"{len(list(todo.glob('*.wav')))} scape plots to compute with {JOBS} jobs", flush=True)
subprocess.run([sys.executable, "run_python_scapeplot.py", "-a", str(todo), "-s", str(WORK / "ssm"),
                "-p", str(WORK / "scplot"), "-j", str(JOBS)], cwd=MUSDR, env=env, check=True)

# SI
sys.path[:0] = [str(MUSDR), str(MUSDR / "musdr")]
from musdr.eval_metrics import compute_structure_indicator  # noqa: E402

rows = []
for f in sorted((WORK / "scplot").glob("*_fitness.npy")):
    group, name = f.stem.replace("_fitness", "").split("__")
    rows.append({"group": group, "piece": name,
                 "SI_short": compute_structure_indicator(str(f), 3, 8),
                 "SI_mid": compute_structure_indicator(str(f), 8, 15),
                 "SI_long": compute_structure_indicator(str(f), 15)})
df = pd.DataFrame(rows)
df.to_csv(WORK / "si_per_piece.csv", index=False)
summary = df.groupby("group")[["SI_short", "SI_mid", "SI_long"]].mean()
summary.to_csv(ROOT / "results/task_a_si.csv")
print(summary.to_markdown(floatfmt=".4f"))
print("SI DONE", flush=True)
