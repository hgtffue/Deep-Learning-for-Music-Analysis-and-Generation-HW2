"""
Download the model checkpoints and rendering tools from the cloud drive into the repo root (uses gdown).

    python download.py              # checkpoints/ + tools/ (everything needed for inference)
    python download.py --results    # also results/ (the submitted generation results)

Files that already exist are skipped.
"""
import argparse
import pathlib
import time

import gdown

ROOT = pathlib.Path(__file__).resolve().parent.parent
FOLDER = "https://drive.google.com/drive/folders/1DtMPrHbmGz7sVhg6q9yqqLGSFMj-gofX"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="store_true", help="also download the submitted generation results")
    ap.add_argument("--dest", default=str(ROOT), help="target folder (default: repo root)")
    args = ap.parse_args()
    want = {"checkpoints", "tools"} | ({"results"} if args.results else set())

    files = gdown.download_folder(FOLDER, skip_download=True, quiet=True)
    for f in files:
        parts = pathlib.PureWindowsPath(f.path).parts[1:]       # drop the drive folder's own name
        if not parts or parts[0] not in want or {"include", "lib"} & set(parts):  # FluidSynth headers: not needed
            continue
        dst = pathlib.Path(args.dest, *parts)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):                                  # Drive occasionally refuses a request
            try:
                gdown.download(id=f.id, output=str(dst), quiet=True)
                break
            except gdown.exceptions.FileURLRetrievalError:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        print(dst.relative_to(args.dest), flush=True)
    print("done")


if __name__ == "__main__":
    main()
