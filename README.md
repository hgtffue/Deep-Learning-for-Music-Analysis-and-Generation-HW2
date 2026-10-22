# HW2 – Symbolic Music Generation (POP909)

Report: [report.md](report.md). Code: `code/`. Generated MIDI / tokens / metrics: `outputs/`, `results/`.

Not included in the repo (size / licensing): `POP909/` (dataset), `ckpt/` (model checkpoints),
`data/*.json` except `tok2idx.json` (rebuilt by `music_tok.py`), `.wav` files, `tools/`
(FluidSynth + FluidR3_GM.sf2), `HW2_scripts/MusDr`.

## Setup

```
git clone https://github.com/slSeanWU/MusDr HW2_scripts/MusDr
# librosa >= 0.10: in HW2_scripts/MusDr/mueller_audio_tools/ssm_features.py (line 347) use
#   x, Fs = librosa.load(fn_wav, sr=Fs)
```
Put FluidSynth under `tools/fs/` and `FluidR3_GM.sf2` under `tools/sf/` (used by `code/render.py`).
On Windows set `KMP_DUPLICATE_LIB_OK=TRUE`.

## Reproduce

```
cd code
python music_tok.py                 # POP909/training -> data/*.json
python crop_real.py                 # first 32 bars of real songs (reference for metrics)
python train.py --data ../data/basic_sequences.json --out ../ckpt/basic --batch 8 --steps 25000
python pipeline.py                  # task a, task b, bonus training/generation, render, eval
python si_eval.py 10                # SI (structureness indicator) for task a
python ablation.py                  # ablation studies
python analysis.py                  # memorization, task b consistency, figures
```
