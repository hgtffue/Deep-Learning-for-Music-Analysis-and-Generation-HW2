"""
REMI-style tokenization for POP909 (with chord & tempo tokens), plus decoding back to MIDI.

Per-bar layout (basic, all tracks merged):
    Bar TS_<beats> Tempo_<bpm> { Pos_p [Chord_<root>_<qual>] { Pitch_x Vel_v Dur_d }* }*

Per-bar layout (bonus, MELODY + PIANO interleaved bar by bar):
    Bar TS_<beats> Tempo_<bpm> Track_Mel { Pos_p [Chord_..] {Pitch Vel Dur}* }*
                               Track_Pno { Pos_p {Pitch Vel Dur}* }*

Time grid: POS_PER_BEAT positions per beat, positions are counted from the bar's downbeat
using the human-annotated beats in beat_midi.txt (so expressive timing is removed).
"""
import bisect
import json
import pathlib

import numpy as np
import pretty_midi

POS_PER_BEAT = 4                 # 16th-note grid
MAX_BEATS = 8
MAX_POS = MAX_BEATS * POS_PER_BEAT
MAX_DUR = 64                     # in grid units (4 bars of 4/4)
PITCH_LO, PITCH_HI = 21, 108
TEMPI = list(range(30, 211, 5))

ROOT_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ROOTS = {'C': 0, 'B#': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'Fb': 4,
         'F': 5, 'E#': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
         'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11}
QUALITIES = ["maj", "min", "dim", "aug", "sus2", "sus4", "maj7", "min7", "7",
             "hdim7", "dim7", "maj6", "min6", "minmaj7", "sus4(b7)"]


def build_vocab():
    toks = ["<PAD>", "<BOS>", "<EOS>", "Bar", "Track_Mel", "Track_Pno"]
    toks += [f"TS_{b}" for b in range(1, MAX_BEATS + 1)]
    toks += [f"Tempo_{t}" for t in TEMPI]
    toks += [f"Pos_{i}" for i in range(MAX_POS)]                 # contiguous (needed by MusDr GS)
    toks += ["Chord_N"] + [f"Chord_{r}_{q}" for r in ROOT_NAMES for q in QUALITIES]
    toks += [f"Pitch_{p}" for p in range(PITCH_LO, PITCH_HI + 1)]  # contiguous (needed by MusDr H)
    toks += [f"Vel_{v}" for v in range(1, 128, 4)]
    toks += [f"Dur_{d}" for d in range(1, MAX_DUR + 1)]
    return {t: i for i, t in enumerate(toks)}


TOK2IDX = build_vocab()
IDX2TOK = {i: t for t, i in TOK2IDX.items()}
VOCAB_SIZE = len(TOK2IDX)


# ---------------------------------------------------------------- parsing helpers
def parse_chord_label(label):
    """'A:min7/b7' -> 'Chord_A_min7' (inversions dropped); unknown -> Chord_N"""
    if label == "N" or ":" not in label:
        return "Chord_N"
    root, qual = label.split(":", 1)
    qual = qual.split("/")[0]
    if root not in ROOTS:
        return "Chord_N"
    if qual not in QUALITIES:
        qual = "min" if qual.startswith("min") else "maj"
    return f"Chord_{ROOT_NAMES[ROOTS[root]]}_{qual}"


def load_beats(beat_path):
    times, orders = [], []
    for line in open(beat_path):
        p = line.split()
        if not p:
            continue
        times.append(float(p[0]))
        orders.append(float(p[2]) if len(p) > 2 else 0.0)
    return np.array(times), orders


def beat_to_bar(orders):
    """For every annotated beat return (bar_idx, beat_in_bar); same logic as TA tokenization.py."""
    downs = [i for i, o in enumerate(orders) if o == 1.0]
    bpb = downs[1] - downs[0] if len(downs) >= 2 else 4
    info, bar = [], -1
    first = downs[0] if downs else 0
    if first > 0:
        bar = 0
        for i in range(first):
            info.append((0, (bpb - (first - i)) % bpb))
    for i in range(first, len(orders)):
        if orders[i] == 1.0 or bar == -1:
            bar += 1
            k = 0
        info.append((bar, k))
        k += 1
    return info


class Grid:
    """Maps seconds <-> (bar, pos) using annotated beats."""

    def __init__(self, beat_times, orders):
        self.bt = np.asarray(beat_times, dtype=float)
        self.info = beat_to_bar(orders)
        self.n_bars = self.info[-1][0] + 1
        # beats per bar & first beat index of each bar
        self.bar_first_beat, self.bar_beats = {}, {}
        for i, (b, k) in enumerate(self.info):
            self.bar_first_beat.setdefault(b, i - k)  # pickup bar: virtual beat index may be < 0
            self.bar_beats[b] = max(self.bar_beats.get(b, 0), k + 1)
        last = self.n_bars - 1
        if last > 0:  # last bar is usually incomplete in the annotation
            self.bar_beats[last] = max(self.bar_beats[last], self.bar_beats[last - 1])
        self.ibi = np.median(np.diff(self.bt)) if len(self.bt) > 1 else 0.5

    def sec_to_beat(self, t):
        bt = self.bt
        if t <= bt[0]:
            return (t - bt[0]) / (bt[1] - bt[0])
        if t >= bt[-1]:
            return len(bt) - 1 + (t - bt[-1]) / (bt[-1] - bt[-2])
        i = bisect.bisect_right(bt, t) - 1
        return i + (t - bt[i]) / (bt[i + 1] - bt[i])

    def beat_to_sec(self, b):
        bt = self.bt
        if b <= 0:
            return bt[0] + b * (bt[1] - bt[0])
        if b >= len(bt) - 1:
            return bt[-1] + (b - len(bt) + 1) * (bt[-1] - bt[-2])
        i = int(np.floor(b))
        return bt[i] + (b - i) * (bt[i + 1] - bt[i])

    def beat_to_barpos(self, beat_f):
        """fractional beat index -> (bar, pos) after rounding to grid"""
        q = int(round(beat_f * POS_PER_BEAT))
        bi, sub = divmod(q, POS_PER_BEAT)
        if bi < 0:
            return None
        if bi >= len(self.info):
            last_bar, last_k = self.info[-1]
            extra = bi - (len(self.info) - 1)
            k = last_k + extra
            bar = last_bar
            while k >= self.bar_beats.get(bar, 4):
                k -= self.bar_beats.get(bar, 4)
                bar += 1
            return bar, k * POS_PER_BEAT + sub
        bar, k = self.info[bi]
        return bar, k * POS_PER_BEAT + sub

    def bar_tempo(self, bar):
        s = self.bar_first_beat[bar]
        e = s + self.bar_beats[bar]
        s, e = max(0, s), min(len(self.bt) - 1, e)
        ibi = (self.bt[e] - self.bt[s]) / (e - s) if e > s else self.ibi
        return 60.0 / max(ibi, 1e-3)


def quant_tempo(bpm):
    return int(min(210, max(30, round(bpm / 5) * 5)))


# ---------------------------------------------------------------- song -> bars
def parse_song(tracks, beat_path, chord_path=None):
    """
    tracks: dict name -> list of pretty_midi.Note
    returns dict with list of bars: {beats, tempo, chords:[(pos,tok)], notes:{name:[(pos,pitch,vel,dur)]}}
    """
    times, orders = load_beats(beat_path)
    g = Grid(times, orders)
    bars = {}

    def get_bar(b):
        if b not in bars:
            bars[b] = {"chords": [], "notes": {k: [] for k in tracks}}
        return bars[b]

    for name, notes in tracks.items():
        for n in notes:
            sb = g.sec_to_beat(n.start)
            eb = g.sec_to_beat(n.end)
            bp = g.beat_to_barpos(sb)
            if bp is None:
                continue
            bar, pos = bp
            if pos >= MAX_POS:
                continue
            dur = int(min(MAX_DUR, max(1, round((eb - sb) * POS_PER_BEAT))))
            pitch = min(PITCH_HI, max(PITCH_LO, n.pitch))
            vel = max(1, min(125, n.velocity // 4 * 4 + 1))
            get_bar(bar)["notes"][name].append((pos, pitch, vel, dur))

    if chord_path is not None and pathlib.Path(chord_path).exists():
        prev = None
        for line in open(chord_path):
            p = line.split()
            if len(p) < 3:
                continue
            tok = parse_chord_label(p[2])
            if tok == prev:
                continue
            prev = tok
            bp = g.beat_to_barpos(g.sec_to_beat(float(p[0])))
            if bp is None or bp[1] >= MAX_POS:
                continue
            get_bar(bp[0])["chords"].append((bp[1], tok))

    if not bars:
        return None
    n_bars = max(max(bars) + 1, 1)
    out = []
    for b in range(n_bars):
        d = bars.get(b, {"chords": [], "notes": {k: [] for k in tracks}})
        d["beats"] = min(MAX_BEATS, g.bar_beats.get(b, g.bar_beats[g.n_bars - 1]))
        d["tempo"] = quant_tempo(g.bar_tempo(b)) if b in g.bar_first_beat else quant_tempo(60 / g.ibi)
        out.append(d)
    return {"bars": out, "grid": g}


def _bar_body(events_notes, chords=None):
    """events_notes: list of (pos,pitch,vel,dur); chords: list of (pos,tok)"""
    by_pos = {}
    for pos, p, v, d in events_notes:
        by_pos.setdefault(pos, {"c": None, "n": []})["n"].append((p, v, d))
    for pos, tok in chords or []:
        by_pos.setdefault(pos, {"c": None, "n": []})["c"] = tok
    toks = []
    for pos in sorted(by_pos):
        e = by_pos[pos]
        toks.append(f"Pos_{pos}")
        if e["c"]:
            toks.append(e["c"])
        seen = set()
        for p, v, d in sorted(e["n"], key=lambda x: (x[0], -x[2])):
            if p in seen:
                continue
            seen.add(p)
            toks += [f"Pitch_{p}", f"Vel_{v}", f"Dur_{d}"]
    return toks


def first_note_bar(song, names):
    for i, b in enumerate(song["bars"]):
        if any(b["notes"][k] for k in names):
            return i
    return None


def encode_basic(song, use_chord=True, start_bar=None):
    names = list(song["bars"][0]["notes"].keys())
    s = first_note_bar(song, names) if start_bar is None else start_bar
    toks = ["<BOS>"]
    for b in song["bars"][s:]:
        toks += ["Bar", f"TS_{b['beats']}", f"Tempo_{b['tempo']}"]
        notes = [n for k in names for n in b["notes"][k]]
        toks += _bar_body(notes, b["chords"] if use_chord else None)
    return toks


def encode_bonus_bar(b, mel="MELODY", pno="PIANO", with_piano=True):
    toks = ["Bar", f"TS_{b['beats']}", f"Tempo_{b['tempo']}", "Track_Mel"]
    toks += _bar_body(b["notes"][mel], b["chords"])
    toks += ["Track_Pno"]
    if with_piano:
        toks += _bar_body(b["notes"][pno])
    return toks


def encode_bonus(song, start_bar=None):
    s = first_note_bar(song, ["MELODY", "PIANO"]) if start_bar is None else start_bar
    toks = ["<BOS>"]
    for b in song["bars"][s:]:
        toks += encode_bonus_bar(b)
    return toks


# ---------------------------------------------------------------- tokens -> MIDI
def decode(tokens, grid=None, bar_offset=0, program=0):
    """
    tokens: list of token strings.
    grid: if given (a Grid), place bars on the annotated beats (bar index = bar_offset + k);
          otherwise use TS/Tempo tokens.
    Returns PrettyMIDI. If Track_* tokens are present -> MELODY + PIANO instruments.
    """
    two_track = any(t in ("Track_Mel", "Track_Pno") for t in tokens)
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    insts = {}
    if two_track:
        insts["MELODY"] = pretty_midi.Instrument(program, name="MELODY")
        insts["PIANO"] = pretty_midi.Instrument(program, name="PIANO")
    else:
        insts["PIANO"] = pretty_midi.Instrument(program, name="PIANO")
    track = "PIANO"
    bar_idx = -1
    bar_start = 0.0
    beats, tempo = 4, 120
    pos = 0
    cur_pitch = cur_vel = None
    next_bar_start = 0.0

    def pos_to_sec(bar_k, p):
        if grid is not None:
            gb = bar_offset + bar_k
            fb = grid.bar_first_beat.get(gb)
            if fb is None:  # beyond annotation: extrapolate
                lastb = grid.n_bars - 1
                fb = grid.bar_first_beat[lastb] + (gb - lastb) * grid.bar_beats[lastb]
            return grid.beat_to_sec(fb + p / POS_PER_BEAT)
        return bar_start + p / POS_PER_BEAT * 60.0 / tempo

    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "Bar":
            bar_idx += 1
            bar_start = next_bar_start
            beats, pos = 4, 0
        elif t.startswith("TS_"):
            beats = int(t[3:])
        elif t.startswith("Tempo_"):
            tempo = int(t[6:])
        elif t == "Track_Mel":
            track, pos = "MELODY", 0
        elif t == "Track_Pno":
            track, pos = "PIANO", 0
        elif t.startswith("Pos_"):
            pos = int(t[4:])
        elif t.startswith("Pitch_") and bar_idx >= 0:
            if i + 2 < len(tokens) and tokens[i + 1].startswith("Vel_") and tokens[i + 2].startswith("Dur_"):
                p = int(t[6:])
                v = int(tokens[i + 1][4:])
                d = int(tokens[i + 2][4:])
                st = pos_to_sec(bar_idx, pos)
                if grid is not None:
                    en = pos_to_sec(bar_idx, pos + d)
                else:
                    en = st + d / POS_PER_BEAT * 60.0 / tempo
                insts[track if track in insts else "PIANO"].notes.append(
                    pretty_midi.Note(velocity=v, pitch=p, start=st, end=max(en, st + 0.03)))
                i += 3
                continue
        if t.startswith("Tempo_") or t.startswith("TS_") or t == "Bar":
            next_bar_start = bar_start + beats * 60.0 / tempo
        i += 1
    pm.instruments = list(insts.values())
    return pm


# ---------------------------------------------------------------- dataset building
def load_pop909_song(folder, mode="basic"):
    folder = pathlib.Path(folder)
    mid = pretty_midi.PrettyMIDI(str(folder / f"{folder.name}.mid"))
    tracks = {i.name.strip().upper(): i.notes for i in mid.instruments}
    if mode == "bonus":
        tracks = {"MELODY": tracks.get("MELODY", []), "PIANO": tracks.get("PIANO", [])}
    return parse_song(tracks, folder / "beat_midi.txt", folder / "chord_midi.txt")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", default="../POP909/training")
    ap.add_argument("--out_dir", default="../data")
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(TOK2IDX, open(out / "tok2idx.json", "w"), indent=1)
    basic, bonus = [], []
    for d in sorted(pathlib.Path(args.dataset_path).iterdir()):
        if not d.is_dir():
            continue
        try:
            s = load_pop909_song(d, "basic")
            basic.append({"song_name": d.name, "tokens": encode_basic(s)})
            s = load_pop909_song(d, "bonus")
            bonus.append({"song_name": d.name, "tokens": encode_bonus(s)})
        except Exception as e:
            print("error", d.name, e)
    json.dump(basic, open(out / "basic_sequences.json", "w"))
    json.dump(bonus, open(out / "bonus_sequences.json", "w"))
    lens = [len(x["tokens"]) for x in basic]
    print(f"vocab {VOCAB_SIZE}, songs {len(basic)}, mean len {np.mean(lens):.0f}, max {max(lens)}")
    lens = [len(x["tokens"]) for x in bonus]
    print(f"bonus mean len {np.mean(lens):.0f}, max {max(lens)}")
