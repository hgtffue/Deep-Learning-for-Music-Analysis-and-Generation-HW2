"""
Render every .mid in the given folders to .wav.   python render.py <dir> [<dir> ...] [--sf2 path]

Uses pretty_midi + pyfluidsynth (pip) and the FluidSynth shared library (not pip-installable):
  * Windows: put the FluidSynth release under ../tools/fs (any sub-folder containing fluidsynth.exe / the DLLs)
  * Linux:   apt install fluidsynth        macOS: brew install fluid-synth
Soundfont: FluidR3_GM.sf2, looked up at $SF2_PATH, then ../tools/sf/FluidR3_GM.sf2.
(The fluidsynth CLI is avoided on purpose: it occasionally never terminates on dense generated MIDIs.)
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if sys.platform == "win32" and (ROOT / "tools/fs").exists():
    FS_BIN = next((ROOT / "tools/fs").rglob("fluidsynth.exe")).parent
    os.add_dll_directory(str(FS_BIN))
    os.environ["PATH"] = str(FS_BIN) + os.pathsep + os.environ["PATH"]
SF = pathlib.Path(os.environ.get("SF2_PATH", ROOT / "tools/sf/FluidR3_GM.sf2"))
SR = 44100

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pretty_midi  # noqa: E402
import soundfile  # noqa: E402

try:  # pretty_midi may have been imported before the DLL path was set: retry its pyfluidsynth import
    import pretty_midi.fluidsynth as _pmfs
    if not _pmfs._HAS_FLUIDSYNTH:
        import fluidsynth
        _pmfs.fluidsynth, _pmfs._HAS_FLUIDSYNTH = fluidsynth, True
except ImportError:
    pass


def render_midi(mid, wav, sf2=None):
    sf2 = pathlib.Path(sf2 or SF)
    if not sf2.exists():
        raise FileNotFoundError(f"soundfont not found: {sf2} (set SF2_PATH or put it under tools/sf/)")
    pm = pretty_midi.PrettyMIDI(str(mid))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio = pm.fluidsynth(fs=SR, sf2_path=str(sf2))
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9
    soundfile.write(str(wav), audio.astype(np.float32), SR, subtype="PCM_16")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--sf2", default=None)
    args = ap.parse_args()
    for d in args.dirs:
        for mid in sorted(pathlib.Path(d).rglob("*.mid")):
            render_midi(mid, mid.with_suffix(".wav"), args.sf2)
            print(mid.with_suffix(".wav"), flush=True)
