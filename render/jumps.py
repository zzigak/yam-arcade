"""Largest frame-to-frame changes in a frames dir (pops, time jumps), relative to their neighbourhood.
    ~/simquarry/genesis-world/.venv/bin/python render/jumps.py <frames_dir> [--top 10]"""
import os, sys
import numpy as np
from PIL import Image
d = sys.argv[1]; top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 10
fs = sorted(int(f[:5]) for f in os.listdir(d) if f.endswith(".png") and f[:5].isdigit())
prev, rows = None, []
for f in fs:
    im = np.asarray(Image.open(os.path.join(d, f"{f:05d}.png")).convert("L").resize((160, 90)), float)
    if prev is not None:
        rows.append((f, float(np.abs(im - prev).mean())))
    prev = im
dif = np.array([r[1] for r in rows])
med = np.array([np.median(dif[max(0, i - 15):i + 16]) for i in range(len(dif))])
score = dif / np.maximum(med, 0.05)
print(f"{len(fs)} frames; median diff {np.median(dif):.2f}, max {dif.max():.2f}")
for i in np.argsort(-score)[:top]:
    print(f"  frame {rows[i][0]:5d} t {rows[i][0] / 30:6.2f}s diff {dif[i]:5.2f} (x{score[i]:4.1f} local median)")
