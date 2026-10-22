# HW2 – Symbolic Music Generation（報告草稿）

## 0. 程式與輸出結構

```
code/music_tok.py   tokenization（REMI + chord + tempo/meter），以及 token → MIDI 解碼
code/train.py       GPT-2 訓練（basic / bonus 共用）
code/generate.py    推論：uncond（task a）、continue（task b）、bonus（伴奏）
code/render.py      FluidSynth + FluidR3_GM.sf2 轉 .wav
code/summarize.py   對 task a 各組設定跑 eval_basic.py 並彙整表格
code/si_eval.py     task a 的 SI（MusDr scape plot）計算
code/pipeline.py    依序執行所有生成 / 訓練 / 評估
outputs/task_a/<config>/   每組 20 首 32 小節 .mid（+ tokens.json, metrics.csv）
outputs/task_b/<config>/   3 個 prompt 各自的 8+24 小節 .mid + .wav
outputs/bonus/<config>/    10 首 16 小節 MELODY+PIANO .mid + .wav + metrics.csv
```

## 1. Token Representation

採用 **REMI 風格**的單一序列表示，並加入和弦與速度資訊（**w/ Chord**）：

```
<BOS> Bar TS_4 Tempo_90 Pos_0 Chord_B_maj Pitch_47 Vel_65 Dur_6 Pitch_75 Vel_121 Dur_2 Pos_1 Pitch_54 Vel_49 Dur_5 ... Bar TS_4 Tempo_90 ...
```

| Token 類別 | 數量 | 說明 |
|---|---|---|
| `<PAD> <BOS> <EOS>` | 3 | 特殊 token |
| `Bar` | 1 | 小節開頭 |
| `TS_1..8` | 8 | 該小節的拍數（POP909 有 4 拍與 6 拍的曲子） |
| `Tempo_30..210` | 37 | 每小節速度，以 5 BPM 量化 |
| `Pos_0..31` | 32 | 小節內位置，16 分音符格（每拍 4 格） |
| `Chord_<root>_<quality>` + `Chord_N` | 181 | 12 根音 × 15 種和弦性質，轉位省略 |
| `Pitch_21..108` | 88 | 音高 |
| `Vel_1..125` | 32 | 力度，以 4 量化 |
| `Dur_1..64` | 64 | 時值（16 分音符為單位，最長 4 小節） |
| `Track_Mel / Track_Pno` | 2 | 只在 bonus 使用 |
| **合計** | **448** | |

重點：
- **時間格線使用 `beat_midi.txt` 的人工標註拍點**，而非 MIDI 內的 tempo（POP909 MIDI tempo 固定為 120）。音符起點先換算成「連續拍數」再四捨五入到 16 分音符格，因此可以去除演奏的速度起伏。每小節的速度由該小節的拍點間距算出，以 `Tempo_x` token 表示。
- 和弦取自 `chord_midi.txt`，只在和弦改變時於對應位置插入 `Chord_*` token。
- Basic task 把 MELODY / BRIDGE / PIANO 三軌合併。每首歌從第一個有音符的小節開始。
- `Pos_*` 與 `Pitch_*` 的 id 皆為連續，符合 MusDr（`eval_basic.py`）的假設。
- 平均序列長度約 6,049 tokens/首（約 60 tokens/小節）；32 小節約 2,000–3,500 tokens。

## 2. Model Architecture

Hugging Face `GPT2LMHeadModel`（decoder-only Transformer），從頭訓練：

| 參數 | 值 |
|---|---|
| layers / heads / d_model | 8 / 8 / 512 |
| context length | 1536 |
| 參數量 | 26.2M |
| dropout | 0.1 |

## 3. Implementation Details

- **資料增強**：每個訓練樣本隨機移調 −6 ~ +6 半音（Pitch 與 Chord 根音同步移動，若超出音域則重新抽）。
- **取樣方式**：依歌曲長度加權隨機選曲，從隨機的 `Bar` token 開始切 1536 個 token（10% 機率從 `<BOS>` 開始）。若碰到歌曲結尾則補 `<EOS>` 與 padding（loss 忽略）。
- **最佳化**：AdamW（lr 4e-4、betas (0.9, 0.95)、weight decay 0.01），1000 步 warmup + cosine decay，batch size 8，grad clip 1.0，bf16 autocast。
- **訓練步數**：basic 25,000 步（約 1.8 小時，RTX 5060 8GB）；bonus 12,000 步。
- **驗證**：保留 16 首歌只用來監控 validation loss，其餘 880 首用於訓練。
- 訓練中在 train loss 首次低於 1.2 與 0.9 時各存一份 checkpoint，以比較「不同 loss 的模型」。

Basic 模型 loss 曲線（摘要）：

| step | 1k | 3k | 5k | 7k | 9k | 13k | 17k | 21k | 25k |
|---|---|---|---|---|---|---|---|---|---|
| train | 2.13 | 1.51 | 1.35 | 1.25 | 1.20 | 1.07 | 0.95 | 0.88 | 0.84 |
| val | 2.00 | 1.53 | 1.43 | 1.33 | **1.29** | 1.39 | 1.39 | 1.47 | 1.54 |

validation loss 在約 9k 步最低，之後開始 overfit。

### 推論（Inference）

- 自回歸取樣，搭配 KV cache；當長度超過 context 時，保留最後約 768 個 token（從 `Bar` 對齊處開始）作為新 context，重新計算 cache。
- **文法限制（constrained decoding）**：`Pitch` 後只能接 `Vel`、`Vel` 後只能接 `Dur`、`Bar → TS → Tempo` 依序出現、`Pos` 後必須接音符（或和弦）。同一小節內 `Pos` 必須遞增，且不能超出該小節拍數。這只排除不合法的序列，不影響音樂內容。
- 32 小節的終止條件：當模型要開第 33 個 `Bar` 時停止；未滿 32 小節前禁止 `<EOS>`。

## 4. Basic Task a – Unconditional Generation（每組 20 首、32 小節）

| Config | Model | Representation | w/ Chord | Loss (train) | Top-K | Top-p | Temperature | H1 | **H4** | **GS** | SI_short | SI_mid | SI_long |
|---|---|---|:-:|---|---|---|---|---|---|---|---|---|---|
| a1 | GPT-2 | REMI | O | 1.20 (best val) | – | 0.95 | 1.0 | 2.1093 | 2.4458 | 0.9268 | 0.4335 | 0.4116 | 0.3613 |
| a2 | GPT-2 | REMI | O | 1.20 (best val) | 10 | – | 0.8 | 2.0358 | 2.4061 | 0.9468 | 0.5529 | 0.5487 | 0.4808 |
| a3 | GPT-2 | REMI | O | 1.20 (best val) | – | – | 1.2 | 2.0237 | 2.6536 | 0.8145 | 0.3041 | 0.2633 | 0.2041 |
| **a4** | GPT-2 | REMI | O | 0.90 | – | 0.95 | 1.0 | 2.1631 | 2.5637 | 0.9107 | 0.4593 | 0.4407 | 0.3869 |
| a5 | GPT-2 | REMI | O | 0.84 (final) | – | 0.95 | 1.0 | 2.0683 | 2.5615 | 0.9183 | 0.3943 | 0.3899 | 0.3242 |
| Real data (前 32 小節) | | | | | | | | 2.1137 | 2.6133 | 0.8894 | 0.3672 | 0.3733 | 0.3448 |
| Real data (整首) | | | | | | | | 2.1291 | 2.6322 | 0.8807 | – | – | – |

（Real data 用的是同一套 tokenization。由於生成的都是 32 小節，主要比較對象是「前 32 小節」。）

SI（Structureness Indicator，MusDr）：所有 .mid 都用同一個 renderer（pretty_midi + FluidSynth + FluidR3_GM.sf2）轉成音訊，再算 SSM 與 fitness scape plot。SI_short / mid / long 分別是 3–8 秒、8–15 秒、>15 秒時間尺度內 fitness 的最大值。Real data 是從訓練集隨機抽 20 首（seed 0）的前 32 小節，用同一套流程計算（`code/si_eval.py`，逐首結果在 `results/si/si_per_piece.csv`）。

觀察：
- **溫度 / 取樣範圍**：較保守的取樣（a2：T=0.8, top-k 10）讓 H4 偏低、GS 偏高，代表音高集中、節奏過度重複。提高溫度並使用完整分布（a3：T=1.2）讓 H4 最接近真實資料，但 GS 明顯下降（0.81），代表節奏較不穩定。T=1.0 + top-p 0.95 介於兩者之間。
- **模型 loss**：相同取樣設定下（a1/a4/a5），train loss 較低的模型（0.90、0.84）在 H4 上更接近真實資料，GS 也略為下降、更接近真實值。雖然 validation loss 在 9k 步之後上升，生成結果的統計特性反而更像訓練資料（可能是記憶訓練資料所致）。
- **SI（結構性）**：保守取樣的 a2 SI 最高（0.55 / 0.55 / 0.48），遠高於真實資料（0.37 / 0.37 / 0.34）。這不代表結構較好，而是反映它重複同一段素材（與 GS 偏高、H4 偏低一致，也有少數樣本退化成密集重複音）。a3（T=1.2）SI 最低（0.30 / 0.26 / 0.20），長時間尺度幾乎沒有重複結構。T=1.0 + top-p 0.95 的 a1/a4/a5 最接近真實資料，其中 a5 的 short/mid/long 與真實值差距最小，a4 稍高但仍接近。
- 綜合 H1/H4/GS 與真實資料的距離，**a4 最接近**，作為 listening samples（`outputs/task_a/a4_loss0.9_t1.0_p0.95/*.wav`，a1 也已轉成 wav）。

## 5. Basic Task b – Prompt Continuation（8 + 24 小節）

使用 task a 的 checkpoint（train loss 1.20 / best val），不重新訓練。將 prompt 用相同方式 tokenize（同樣使用 prompt 提供的 `beat_midi.txt` / `chord_midi.txt`），以此為前綴續寫，直到總共 32 小節。

| Config | Model | Representation | Checkpoint | Temperature | Top-K | Top-p |
|---|---|---|---|---|---|---|
| b1 | GPT-2 | REMI + Chord | loss 1.20 | 1.0 | – | 0.95 |
| b2 | GPT-2 | REMI + Chord | loss 1.20 | 0.8 | 10 | – |
| b3 | GPT-2 | REMI + Chord | loss 1.20 | 1.2 | – | 0.98 |

輸出：`outputs/task_b/<config>/{001,002,003}.mid/.wav`

**每個 prompt 的最佳設定：（TODO：請聆聽後填寫）**

| Prompt | Best config | 理由 |
|---|---|---|
| 001 | | |
| 002 | | |
| 003 | | |

## 6. Bonus – Accompaniment Generation

**架構：Option A（MELODY 與 PIANO 逐小節交錯，decoder-only）**

每小節的序列：
```
Bar TS_k Tempo_x Track_Mel  [Pos Chord? Pitch Vel Dur ...]  Track_Pno  [Pos Pitch Vel Dur ...]
```
- 只使用 MELODY 和 PIANO 兩軌（不含 BRIDGE）。和弦 token 放在旋律段落中作為條件。
- 模型與訓練設定同 basic（GPT-2、26.2M、移調增強），12,000 步。validation loss 最低在 9k 步（1.020），推論使用這個 checkpoint。
- **推論流程**：先送入第 1 小節的條件段（`Bar … Track_Mel <旋律與和弦> Track_Pno`），模型只產生鋼琴段落（限制只能輸出 `Pos/Pitch/Vel/Dur/Bar`，`Pos` 必須遞增）。當模型輸出 `Bar` 時，攔截並改為強制插入下一小節的條件段，如此重複 16 小節。
- **對齊**：輸出 MIDI 的鋼琴音符依 prompt 的 `beat_midi.txt` 拍點放回真實時間。MELODY 軌直接沿用原始 `melody.mid`。驗證方式：把 ground-truth 鋼琴經 tokenize → 解碼後再跑 `eval_bonus.py`，得到 CA 0.750（ceiling 0.7625），DP/DV/DIOI ≈ 1.0、DD 0.96，確認格線與評估程式的小節一致（剩餘差距來自 16 分音符量化）。

| Config | Temperature | Top-K | Top-p | CA | CA-ceiling | OA_P (DP) | OA_V (DV) | OA_D (DD) | OA_IOI (DIOI) |
|---|---|---|---|---|---|---|---|---|---|
| c1 | 1.0 | – | 0.9 | 0.7438 | 0.7625 | 0.9088 | 0.5725 | 0.6949 | 0.5524 |
| **c2** | 0.8 | 10 | – | **0.7625** | 0.7625 | 0.9008 | 0.5311 | **0.7261** | **0.5945** |
| c3 | 1.2 | – | 0.95 | 0.7188 | 0.7625 | 0.9041 | 0.5937 | 0.6911 | 0.5518 |
| GT（經量化，上限參考） | | | | 0.7500 | 0.7625 | 1.0000 | 1.0000 | 0.9618 | 0.9989 |

- **最佳設定為 c2**（T=0.8, top-k 10）：CA 達到 ceiling（0.7625），Duration / IOI 的 OA 也最高。溫度越高，CA 越低、velocity OA 越高（力度分布較分散）。
- 提交的 listening samples：`outputs/bonus/c2_t0.8_k10/*.wav`

## 7. 結論 / 心得（TODO）
