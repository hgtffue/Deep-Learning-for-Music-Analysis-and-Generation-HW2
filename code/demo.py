"""
Inference demo: download the checkpoints (if missing), then regenerate all submitted results
with the configurations from the README.

    python demo.py [--out ../demo_out]

Writes, as .mid + .wav:
    <out>/task_a/              20 pieces, 32 bars (a4: loss 0.90, T 1.0, top-p 0.95)
    <out>/task_b/{b1,b2,b3}/   3 prompts each, 8 + 24 bars (loss 1.20)
    <out>/bonus/               10 songs, 16 bars (c2: T 0.8, top-k 10)
"""
import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = pathlib.Path(__file__).resolve().parent
CK = ROOT / "checkpoints"
TASK_B = {"b1": ["--temp", "1.0", "--top_p", "0.95"],
          "b2": ["--temp", "0.8", "--top_k", "10"],
          "b3": ["--temp", "1.2", "--top_p", "0.98"]}


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
        "--n", "20", "--temp", "1.0", "--top_p", "0.95", "--wav")
    for cfg, sampling in TASK_B.items():
        run("generate.py", "continue", "--ckpt", str(CK / "basic_loss1.2"), "--out", str(out / "task_b" / cfg),
            *sampling, "--wav")
    run("generate.py", "bonus", "--ckpt", str(CK / "bonus"), "--out", str(out / "bonus"),
        "--temp", "0.8", "--top_k", "10", "--wav")
    print("listening samples in", out)


if __name__ == "__main__":
    main()
