"""
Render every .mid in the given folders to .wav (FluidR3_GM soundfont).  python render.py <dir> [<dir> ...]

Uses pretty_midi + pyfluidsynth (the FluidSynth DLL shipped in tools/fs) instead of the fluidsynth CLI:
the CLI player occasionally never terminates on dense generated MIDIs and writes an endless wav.
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FS_BIN = next((ROOT / "tools/fs").rglob("fluidsynth.exe")).parent
os.add_dll_directory(str(FS_BIN))
os.environ["PATH"] = str(FS_BIN) + os.pathsep + os.environ["PATH"]
SF = ROOT / "tools/sf/FluidR3_GM.sf2"
SR = 44100

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pretty_midi  # noqa: E402
import soundfile  # noqa: E402


def render_midi(mid, wav):
    pm = pretty_midi.PrettyMIDI(str(mid))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio = pm.fluidsynth(fs=SR, sf2_path=str(SF))
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9
    soundfile.write(str(wav), audio.astype(np.float32), SR, subtype="PCM_16")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        for mid in sorted(pathlib.Path(d).rglob("*.mid")):
            render_midi(mid, mid.with_suffix(".wav"))
            print(mid.with_suffix(".wav"), flush=True)
