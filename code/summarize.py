"""Run eval_basic.py on every config folder of task a and print a markdown table."""
import json
import os
import pathlib
import subprocess
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "HW2_scripts"
env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")


def run_eval(seq_path, csv_path):
    if not csv_path.exists():
        subprocess.run([sys.executable, "eval_basic.py", "--tok2idx_path", str(ROOT / "data/tok2idx.json"),
                        "--sequences_path", str(seq_path), "--output_csv", str(csv_path)],
                       cwd=SCRIPTS, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    df = pd.read_csv(csv_path)
    return df.iloc[:, 1:].mean()


rows = []
for d in sorted((ROOT / "outputs/task_a").iterdir()):
    if not (d / "tokens.json").exists():
        continue
    cfg = json.load(open(d / "config.json"))
    m = run_eval(d / "tokens.json", d / "metrics.csv")
    rows.append([d.name, pathlib.Path(cfg["ckpt"]).name, cfg["temp"], cfg["top_k"] or "-",
                 cfg["top_p"] if cfg["top_p"] < 1 else "-", *m.values])
for name, f in [("Real data (first 32 bars)", "basic_sequences_32bars.json"), ("Real data (full songs)", "basic_sequences.json")]:
    m = run_eval(ROOT / "data" / f, ROOT / "results" / f"real_{f.replace('.json', '')}.csv")
    rows.append([name, "", "", "", "", *m.values])

df = pd.DataFrame(rows, columns=["Config", "Checkpoint", "Temp", "Top-k", "Top-p", "H1", "H4", "GS"])
df.to_csv(ROOT / "results/task_a_summary.csv", index=False)
print(df.to_markdown(index=False, floatfmt=".4f"))
