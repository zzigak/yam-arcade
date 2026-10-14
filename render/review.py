"""Review a rendered fly-through: frame-to-frame jumps and stride contact sheets.

    ~/simquarry/genesis-world/.venv/bin/python render/review.py render/frames/fly_draft [--stride 30]

Writes <dir>/review/diff.json (mean |frame - previous| per frame, on 160x90 greyscale), prints the biggest jumps
with the camera speed and the station clocks at that moment (a jump the camera does not explain is a pop, a time
jump or a missing object), and contact sheets of every <stride>-th frame plus every knot time (sheet_*.jpg).
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flypath  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--stride", type=int, default=30)
    a = ap.parse_args()
    out = os.path.join(a.dir, "review")
    os.makedirs(out, exist_ok=True)
    fs = sorted(int(f[:5]) for f in os.listdir(a.dir) if f.endswith(".png") and f[:5].isdigit())
    missing = sorted(set(range(flypath.N_FRAMES)) - set(fs))
    print(f"{len(fs)} frames, {len(missing)} missing" + (f" (first {missing[:10]})" if missing else ""))
    prev, rows = None, []
    p = flypath.Path()
    tt = np.arange(flypath.N_FRAMES) / flypath.FPS
    eye, tgt, _ = p(tt)
    d = tgt - eye
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    speed = np.r_[0, np.linalg.norm(np.diff(eye, axis=0), axis=1) * flypath.FPS]
    pan = np.r_[0, np.degrees(np.arccos(np.clip((d[1:] * d[:-1]).sum(1), -1, 1))) * flypath.FPS]
    for f in fs:
        im = np.asarray(Image.open(os.path.join(a.dir, f"{f:05d}.png")).convert("L").resize((160, 90)), float)
        if prev is not None and prev[0] == f - 1:
            rows.append(dict(f=f, t=round(f / flypath.FPS, 2), diff=round(float(np.abs(im - prev[1]).mean()), 3),
                             speed=round(float(speed[f]), 3), pan=round(float(pan[f]), 2)))
        prev = (f, im)
    json.dump(rows, open(os.path.join(out, "diff.json"), "w"))
    dif = np.array([r["diff"] for r in rows])
    # a jump is large relative to its neighbours (camera motion changes slowly, a pop does not)
    med = np.array([np.median(dif[max(0, i - 15):i + 16]) for i in range(len(dif))])
    score = dif / np.maximum(med, 0.05)
    order = np.argsort(-score)[:15]
    print("largest frame-to-frame jumps relative to their neighbourhood:")
    for i in order:
        r = rows[i]
        clocks = " ".join(f"{s}={flypath.sim_time(s, r['t']):.2f}" for s in flypath.CLOCKS)
        print(f"  frame {r['f']:5d} t {r['t']:6.2f}s diff {r['diff']:6.2f} (x{score[i]:4.1f} local median)  "
              f"cam {r['speed']:.2f} m/s pan {r['pan']:.1f} deg/s  {clocks}")
    knots = [int(round(k[0] * flypath.FPS)) for k in flypath.KNOTS]
    sel = sorted(set([f for f in fs if f % a.stride == 0] + [min(k, flypath.N_FRAMES - 1) for k in knots]))
    sel = [f for f in sel if f in set(fs)]
    import subprocess
    py = sys.executable
    for k in range(0, len(sel), 24):
        chunk = sel[k:k + 24]
        subprocess.run([py, os.path.join(HERE, "contact_sheet.py"), a.dir,
                        os.path.join(out, f"sheet_{k // 24:02d}.jpg"), "--frames", ",".join(map(str, chunk)),
                        "--cols", "6", "--width", "320"], check=True)


if __name__ == "__main__":
    main()
