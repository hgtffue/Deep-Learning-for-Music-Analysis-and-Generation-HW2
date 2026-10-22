"""
Quick inference demo: download the checkpoints (if missing), then generate a few listening samples
for task a, task b and the bonus task with the best configurations from the README.

    python demo.py [--out ../demo_out]

Writes <out>/task_a (2 pieces), <out>/task_b (3 prompts, b1) and <out>/bonus (10 songs, c2) as .mid + .wav.
"""
import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = pathlib.Path(__file__).resolve().parent
CK = ROOT / "checkpoints"


def run(*args):
    print(">", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=CODE, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "demo_out"))
    args = ap.parse_args()
    out = pathlib.Path(args.out).resolve()

    if not all((CK / c / "model.safetensors").exists() for c in ["basic_loss0.9", "basic_loss1.2", "bonus"]):
        run("download.py")
    run("generate.py", "uncond", "--ckpt", str(CK / "basic_loss0.9"), "--out", str(out / "task_a"),
        "--n", "2", "--temp", "1.0", "--top_p", "0.95", "--wav")
    run("generate.py", "continue", "--ckpt", str(CK / "basic_loss1.2"), "--out", str(out / "task_b"),
        "--temp", "1.0", "--top_p", "0.95", "--wav")
    run("generate.py", "bonus", "--ckpt", str(CK / "bonus"), "--out", str(out / "bonus"),
        "--temp", "0.8", "--top_k", "10", "--wav")
    print("listening samples in", out)


if __name__ == "__main__":
    main()
