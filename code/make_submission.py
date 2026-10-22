"""Collect everything that goes to the cloud drive into ../submission (code, checkpoints, results)."""
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
SUB = ROOT / "submission"

TASK_A = "a4_loss0.9_t1.0_p0.95"                      # best inference configuration of task a
TASK_B = ["b1_t1.0_p0.95", "b2_t0.8_k10", "b3_t1.2_p0.98"]
BONUS = "c2_t0.8_k10"                                  # best bonus configuration
CKPTS = {"basic_loss0.9": "basic/loss_0.9",            # task a (a4)
         "basic_loss1.2": "basic/loss_1.2",            # task b
         "bonus": "bonus/best_val"}                    # bonus


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


shutil.rmtree(SUB, ignore_errors=True)
for f in sorted((ROOT / "code").glob("*.py")):
    copy(f, SUB / "code" / f.name)
for name, src in CKPTS.items():
    for f in ["config.json", "generation_config.json", "model.safetensors"]:
        copy(ROOT / "ckpt" / src / f, SUB / "checkpoints" / name / f)

for f in sorted((ROOT / "outputs/task_a" / TASK_A).glob("*.*")):
    if f.suffix in (".mid", ".wav"):
        copy(f, SUB / "results/task_a" / f.name)
for cfg in TASK_B:
    for f in sorted((ROOT / "outputs/task_b" / cfg).glob("*.*")):
        if f.suffix in (".mid", ".wav"):
            copy(f, SUB / "results/task_b" / f.stem / f"{cfg.split('_')[0]}{f.suffix}")
for f in sorted((ROOT / "outputs/bonus" / BONUS).glob("*.*")):
    if f.suffix in (".mid", ".wav"):
        copy(f, SUB / "results/bonus" / f.name)
copy(ROOT / "submission_README.md", SUB / "README.md")
n = sum(1 for _ in SUB.rglob("*") if _.is_file())
size = sum(f.stat().st_size for f in SUB.rglob("*") if f.is_file()) / 2 ** 20
print(f"{n} files, {size:.0f} MB in {SUB}")
