"""Duration, level, clipping and the longest silence of every wav in a folder (a sanity check before listening).
Run on .30: venv/bin/python work/wav_stats.py <dir>"""
import glob
import os
import sys

import numpy as np
import soundfile as sf

for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.wav"))):
    y, sr = sf.read(f, dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    hop = int(0.02 * sr)
    n = len(y) // hop
    e = np.sqrt(np.mean(y[:n * hop].reshape(n, hop) ** 2, 1) + 1e-12) if n else np.zeros(0)
    sil = e < 10 ** (-45 / 20)
    run = best = 0
    for s in sil[np.argmax(~sil):len(sil) - np.argmax(~sil[::-1])] if (~sil).any() else []:
        run = run + 1 if s else 0
        best = max(best, run)
    print(f"{os.path.basename(f):28s} {len(y) / sr:5.2f}s sr={sr} rms={20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9):6.1f}dB "
          f"peak={np.abs(y).max():.3f} clip={int((np.abs(y) > 0.999).sum())} longest inner silence={best * 0.02:.2f}s")
