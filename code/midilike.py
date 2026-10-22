"""
MIDI-like event representation (Oore et al., 2018), used only for the tokenization ablation.

    <BOS> [Vel_v] NoteOn_p ... TimeShift_k ... NoteOff_p ...

Same 16th-note grid (from the annotated beats) as the REMI representation, but no Bar / Pos /
TS / Tempo / Chord tokens and no explicit duration: time only advances through TimeShift_1..16
(at most one 4/4 bar per token) and notes are closed by NoteOff. Vel_ is emitted only when the
velocity changes.
"""
import music_tok as mt

MAX_SHIFT = 16


def build_vocab():
    toks = ["<PAD>", "<BOS>", "<EOS>"]  # same ids as the REMI vocabulary
    toks += [f"TimeShift_{k}" for k in range(1, MAX_SHIFT + 1)]
    toks += [f"Vel_{v}" for v in range(1, 128, 4)]
    toks += [f"NoteOn_{p}" for p in range(mt.PITCH_LO, mt.PITCH_HI + 1)]
    toks += [f"NoteOff_{p}" for p in range(mt.PITCH_LO, mt.PITCH_HI + 1)]
    return {t: i for i, t in enumerate(toks)}


TOK2IDX = build_vocab()
IDX2TOK = {i: t for t, i in TOK2IDX.items()}
VOCAB_SIZE = len(TOK2IDX)


def remi_to_notes(tokens):
    """REMI tokens -> [(onset, pitch, vel, dur)] in absolute 16th-note units"""
    notes, bar_start, beats, pos, started = [], 0, 4, 0, False
    for i, t in enumerate(tokens):
        if t == "Bar":
            if started:
                bar_start += beats * mt.POS_PER_BEAT
            started, beats, pos = True, 4, 0
        elif t.startswith("TS_"):
            beats = int(t[3:])
        elif t.startswith("Pos_"):
            pos = int(t[4:])
        elif t.startswith("Pitch_") and i + 2 < len(tokens) and tokens[i + 2].startswith("Dur_"):
            notes.append((bar_start + pos, int(t[6:]), int(tokens[i + 1][4:]), int(tokens[i + 2][4:])))
    return notes


def encode(notes):
    """[(onset, pitch, vel, dur)] -> MIDI-like tokens"""
    ev = []  # (time, order, token, vel)  note-offs before note-ons at the same time
    by_pitch = {}
    for on, p, v, d in sorted(notes):
        by_pitch.setdefault(p, []).append((on, v, d))
    for p, ns in by_pitch.items():
        for k, (on, v, d) in enumerate(ns):
            if k > 0 and ns[k - 1][0] == on:  # duplicate onset of the same pitch
                continue
            nxt = next((n[0] for n in ns[k + 1:] if n[0] > on), None)
            end = on + d if nxt is None else min(on + d, nxt)  # a re-attack closes the previous note
            ev.append((on, 1, p, v))
            ev.append((end, 0, p, None))
    ev.sort(key=lambda e: (e[0], e[1], e[2]))
    toks, t, vel = ["<BOS>"], 0, None
    for time, kind, p, v in ev:
        while time > t:
            k = min(MAX_SHIFT, time - t)
            toks.append(f"TimeShift_{k}")
            t += k
        if kind == 1:
            if v != vel:
                toks.append(f"Vel_{v}")
                vel = v
            toks.append(f"NoteOn_{p}")
        else:
            toks.append(f"NoteOff_{p}")
    return toks


def decode(tokens, max_dur=mt.MAX_DUR):
    """MIDI-like tokens -> [(onset, pitch, vel, dur)]; unmatched note-ons are closed after max_dur"""
    t, vel, active, notes = 0, 65, {}, []
    for tok in tokens:
        if tok.startswith("TimeShift_"):
            t += int(tok[10:])
        elif tok.startswith("Vel_"):
            vel = int(tok[4:])
        elif tok.startswith("NoteOn_"):
            p = int(tok[7:])
            if p in active:
                on, v = active.pop(p)
                notes.append((on, p, v, max(1, t - on)))
            active[p] = (t, vel)
        elif tok.startswith("NoteOff_"):
            p = int(tok[8:])
            if p in active:
                on, v = active.pop(p)
                notes.append((on, p, v, max(1, t - on)))
    for p, (on, v) in active.items():
        notes.append((on, p, v, max(1, min(max_dur, t - on))))
    return sorted(notes)


def current_time(tokens):
    return sum(int(t[10:]) for t in tokens if t.startswith("TimeShift_"))


def notes_to_remi(notes, n_bars=32, tempo=90):
    """put notes on 4/4 bars so that the MusDr metrics and the REMI decoder can be reused"""
    bar_len = 4 * mt.POS_PER_BEAT
    bars = [[] for _ in range(n_bars)]
    for on, p, v, d in notes:
        b, pos = divmod(on, bar_len)
        if b < n_bars:
            bars[b].append((pos, p, v, min(mt.MAX_DUR, d)))
    toks = ["<BOS>"]
    for b in bars:
        toks += ["Bar", "TS_4", f"Tempo_{tempo}"] + mt._bar_body(b)
    return toks
