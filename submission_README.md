# HW2 – Symbolic Music Generation (POP909)

Model: GPT-2 (8 layers, 8 heads, d_model 512, context 1536, 26.2M parameters) trained from scratch on
the POP909 training set, REMI tokens with chord / tempo / meter tokens. Details are in the report.

## Folder structure

Source code is on GitHub: https://github.com/hgtffue/Deep-Learning-for-Music-Analysis-and-Generation-HW2
This cloud folder contains the model checkpoints, rendering tools and the submitted results:

```
README.md, requirements.txt
checkpoints/
  basic_loss0.9/         task a model (train loss 0.90) – used for the submitted task a results
  basic_loss1.2/         task b model (train loss 1.20, best validation loss)
  bonus/                 bonus accompaniment model
tools/                   third-party files for .wav rendering only (FluidR3_GM.sf2, FluidSynth Windows build)
results/                 the submitted generation results
  task_a/                20 × (.mid + .wav), 32 bars, best configuration: T=1.0, top-p 0.95
  task_b/<prompt>/       b1 / b2 / b3 .mid + .wav (8-bar prompt + 24 generated bars)
  bonus/<prompt>/        c1 / c2 / c3 .mid + .wav for 001–010: MELODY (given) + generated PIANO, 16 bars
```

## 1. Installation

```
git clone https://github.com/hgtffue/Deep-Learning-for-Music-Analysis-and-Generation-HW2.git hw2
```
Copy `checkpoints/` and `tools/` from this cloud folder into the cloned `hw2/` folder, so that it contains
`code/`, `checkpoints/`, `tools/`. Then (Python 3.13, CUDA GPU required):
```
cd hw2
pip install -r requirements.txt
```
If pip installs a CPU-only PyTorch, install the CUDA build first:
`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`.

**For `.wav` output (`--wav`)**: pyfluidsynth needs the FluidSynth shared library, which is not pip-installable.
- Windows: nothing to do – the FluidSynth build in `tools/fs` is picked up automatically.
- Linux: `sudo apt install fluidsynth`  ·  macOS: `brew install fluid-synth`

The soundfont `tools/sf/FluidR3_GM.sf2` is used (another path can be given with the `SF2_PATH` environment variable).

On Windows, if you see `OMP: Error #15 ... libiomp5md.dll`, set `KMP_DUPLICATE_LIB_OK=TRUE`
(generate.py already sets it).

## 2. Inference

Run from the `code/` folder. `--out` sets the output folder (created if needed); every run writes
`<name>.mid` (+ `<name>.wav` with `--wav`), `tokens.json` and `config.json` there. `--seed` (default 0) fixes the sampling.

### Task a – unconditional generation (32 bars)

```
cd code
python generate.py uncond --ckpt ../checkpoints/basic_loss0.9 --out ../out/task_a --n 20 --temp 1.0 --top_p 0.95 --wav
```
`--n` = number of pieces, `--bars` = number of bars (default 32).

### Task b – continuation (8-bar prompt + 24 bars)

`--prompt_dir` is a folder with one sub-folder per prompt, each containing `<id>.mid`, `beat_midi.txt`
and `chord_midi.txt` (i.e. `POP909/prompts/basic-b` from the homework). The three configurations:

```
python generate.py continue --ckpt ../checkpoints/basic_loss1.2 --prompt_dir <path>/prompts/basic-b --out ../out/task_b/b1 --temp 1.0 --top_p 0.95 --wav
python generate.py continue --ckpt ../checkpoints/basic_loss1.2 --prompt_dir <path>/prompts/basic-b --out ../out/task_b/b2 --temp 0.8 --top_k 10 --wav
python generate.py continue --ckpt ../checkpoints/basic_loss1.2 --prompt_dir <path>/prompts/basic-b --out ../out/task_b/b3 --temp 1.2 --top_p 0.98 --wav
```

| Config | Temperature | Top-k | Top-p |
|---|---|---|---|
| b1 | 1.0 | – | 0.95 |
| b2 | 0.8 | 10 | – |
| b3 | 1.2 | – | 0.98 |

**Best configuration per prompt song:**

| Prompt | Best config |
|---|---|
| 001 | **b1** |
| 002 | **b1** |
| 003 | **b1** |

### Bonus – piano accompaniment for a melody (16 bars)

`--prompt_dir` contains one sub-folder per song with `melody.mid`, `beat_midi.txt`, `chord_midi.txt`
(i.e. `POP909/prompts/bonus`). The output `.mid` has the given MELODY track and the generated PIANO track.

The three configurations:

```
python generate.py bonus --ckpt ../checkpoints/bonus --prompt_dir <path>/prompts/bonus --out ../out/bonus/c1 --temp 1.0 --top_p 0.9 --wav
python generate.py bonus --ckpt ../checkpoints/bonus --prompt_dir <path>/prompts/bonus --out ../out/bonus/c2 --temp 0.8 --top_k 10 --wav
python generate.py bonus --ckpt ../checkpoints/bonus --prompt_dir <path>/prompts/bonus --out ../out/bonus/c3 --temp 1.2 --top_p 0.95 --wav
```

| Config | Temperature | Top-k | Top-p |
|---|---|---|---|
| c1 | 1.0 | – | 0.9 |
| **c2 (best, for all prompt songs)** | 0.8 | 10 | – |
| c3 | 1.2 | – | 0.95 |

If `--prompt_dir` is omitted, `POP909/prompts/basic-b` resp. `POP909/prompts/bonus` next to `code/` is used.

### Rendering existing MIDI files

```
python render.py <folder> [<folder> ...]      # every .mid below the folders -> .wav next to it
```

## 3. Training (not needed for inference)

Requires the POP909 training set under `POP909/training` next to `code/`.
```
python music_tok.py        # tokenize -> data/*.json
python train.py --data ../data/basic_sequences.json --out ../ckpt/basic --batch 8 --steps 25000
python train.py --data ../data/bonus_sequences.json --out ../ckpt/bonus --batch 8 --steps 12000 --save_losses ""
```
Evaluation / analysis scripts (`summarize.py`, `si_eval.py`, `ablation.py`, `analysis.py`) additionally need
the homework's `HW2_scripts` (eval_basic.py, eval_bonus.py, MusDr).
