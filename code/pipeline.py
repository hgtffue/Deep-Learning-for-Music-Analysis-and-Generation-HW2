"""Run all remaining steps sequentially (generation -> bonus training -> bonus generation -> render -> eval)."""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", PYTHONUNBUFFERED="1")
PY = sys.executable


def run(*args, cwd=ROOT / "code"):
    print(">>", " ".join(map(str, args)), flush=True)
    subprocess.run([PY, *map(str, args)], cwd=cwd, env=env, check=True)


def done(d):
    return (ROOT / d / "tokens.json").exists()


CK = ROOT / "ckpt/basic"
TASK_A = {
    "a1_loss1.2_t1.0_p0.95": (CK / "loss_1.2", ["--temp", 1.0, "--top_p", 0.95]),
    "a2_loss1.2_t0.8_k10":   (CK / "loss_1.2", ["--temp", 0.8, "--top_k", 10]),
    "a3_loss1.2_t1.2_pure":  (CK / "loss_1.2", ["--temp", 1.2]),
    "a4_loss0.9_t1.0_p0.95": (CK / "loss_0.9", ["--temp", 1.0, "--top_p", 0.95]),
    "a5_loss0.84_t1.0_p0.95": (CK / "last", ["--temp", 1.0, "--top_p", 0.95]),
}
TASK_B = {
    "b1_t1.0_p0.95": ["--temp", 1.0, "--top_p", 0.95],
    "b2_t0.8_k10":   ["--temp", 0.8, "--top_k", 10],
    "b3_t1.2_p0.98": ["--temp", 1.2, "--top_p", 0.98],
}
BONUS = {
    "c1_t1.0_p0.9": ["--temp", 1.0, "--top_p", 0.9],
    "c2_t0.8_k10":  ["--temp", 0.8, "--top_k", 10],
    "c3_t1.2_p0.95": ["--temp", 1.2, "--top_p", 0.95],
}

for name, (ck, a) in TASK_A.items():
    if not done(f"outputs/task_a/{name}"):
        run("generate.py", "uncond", "--ckpt", ck, "--out", ROOT / "outputs/task_a" / name, "--n", 20, *a)
for name, a in TASK_B.items():
    if not done(f"outputs/task_b/{name}"):
        run("generate.py", "continue", "--ckpt", CK / "loss_1.2", "--out", ROOT / "outputs/task_b" / name, *a)

if not (ROOT / "ckpt/bonus/last").exists():
    run("train.py", "--data", ROOT / "data/bonus_sequences.json", "--out", ROOT / "ckpt/bonus",
        "--batch", 8, "--steps", 12000, "--save_losses", "")
for name, a in BONUS.items():
    if not done(f"outputs/bonus/{name}"):
        run("generate.py", "bonus", "--ckpt", ROOT / "ckpt/bonus/best_val", "--out", ROOT / "outputs/bonus" / name, *a)

run("render.py", ROOT / "outputs/task_b", ROOT / "outputs/bonus", ROOT / "outputs/task_a/a1_loss1.2_t1.0_p0.95")
run("summarize.py")
for name in BONUS:
    d = ROOT / "outputs/bonus" / name
    run("eval_bonus.py", "--prompt_dir", ROOT / "POP909/prompts/bonus", "--gen_midi_dir", d,
        "--output_csv", d / "metrics.csv", cwd=ROOT / "HW2_scripts")
print("PIPELINE DONE", flush=True)
