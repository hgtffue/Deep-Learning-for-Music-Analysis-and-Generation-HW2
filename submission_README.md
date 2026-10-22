# HW2 – Symbolic Music Generation (POP909)

Model: GPT-2 (8 layers, 8 heads, d_model 512, context 1536, 26.2M parameters) trained from scratch on
the POP909 training set, REMI tokens with chord / tempo / meter tokens. See the report for details.

## Folder structure

```
code/                    source code (tokenization, training, inference, evaluation)
checkpoints/
  basic_loss0.9/         task a model (train loss 0.90) – used for the submitted task a results
  basic_loss1.2/         task b model (train loss 1.20, best validation loss)
  bonus/                 bonus accompaniment model
results/
  task_a/                20 × (.mid + .wav), 32 bars, best configuration a4: T=1.0, top-p 0.95
  task_b/<prompt>/       b1.mid/.wav, b2.mid/.wav, b3.mid/.wav (8-bar prompt + 24 generated bars)
  bonus/                 001–010 .mid/.wav: MELODY (given) + generated PIANO, 16 bars, T=0.8, top-k 10
```

## Task b configurations and best configuration per prompt

| Config | Checkpoint | Temperature | Top-k | Top-p |
|---|---|---|---|---|
| b1 | basic_loss1.2 | 1.0 | – | 0.95 |
| b2 | basic_loss1.2 | 0.8 | 10 | – |
| b3 | basic_loss1.2 | 1.2 | – | 0.98 |

| Prompt | Best config |
|---|---|
| 001 | **b1** |
| 002 | **b1** |
| 003 | **b1** |

## Inference

Requirements: Python 3.10+, `torch`, `transformers`, `pretty_midi`, `numpy` (and `pyfluidsynth`,
`soundfile` + FluidSynth / FluidR3_GM.sf2 for wav rendering). On Windows set `KMP_DUPLICATE_LIB_OK=TRUE`.

```
cd code
# task a: 20 unconditional pieces, 32 bars
python generate.py uncond --ckpt ../checkpoints/basic_loss0.9 --out ../out/task_a --n 20 --temp 1.0 --top_p 0.95

# task b: continue each prompt (folder with <id>.mid, beat_midi.txt, chord_midi.txt) by 24 bars
python generate.py continue --ckpt ../checkpoints/basic_loss1.2 --prompt_dir <POP909>/prompts/basic-b --out ../out/task_b/b1 --temp 1.0 --top_p 0.95
python generate.py continue --ckpt ../checkpoints/basic_loss1.2 --prompt_dir <POP909>/prompts/basic-b --out ../out/task_b/b2 --temp 0.8 --top_k 10
python generate.py continue --ckpt ../checkpoints/basic_loss1.2 --prompt_dir <POP909>/prompts/basic-b --out ../out/task_b/b3 --temp 1.2 --top_p 0.98

# bonus: piano accompaniment for each melody prompt (melody.mid, beat_midi.txt, chord_midi.txt)
python generate.py bonus --ckpt ../checkpoints/bonus --prompt_dir <POP909>/prompts/bonus --out ../out/bonus --temp 0.8 --top_k 10

# render .mid -> .wav (expects FluidSynth under tools/fs and FluidR3_GM.sf2 under tools/sf)
python render.py ../out
```

Training (needs the POP909 training set under `POP909/training`):
```
python music_tok.py
python train.py --data ../data/basic_sequences.json --out ../ckpt/basic --batch 8 --steps 25000
python train.py --data ../data/bonus_sequences.json --out ../ckpt/bonus --batch 8 --steps 12000 --save_losses ""
```
