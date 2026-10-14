"""Locate in-wall and spilled particles in a run's raw.npz (container-local coordinates), and when they appeared."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import station as st  # noqa: E402
import containers  # noqa: E402

raw_dir = sys.argv[1]
raw = st.load_raw(raw_dir)
meta = json.load(open(os.path.join(raw_dir, "events.json")))
containers.CUPS = meta.get("cups", "procedural")
C = {n: containers.build(n) for n in ("pitcher", "tumbler", "mug", "teacup")}
P = raw["particles"]
t = raw["t"]
for n in C:
    print(n, "in-wall over time (every 2 s):", raw[f"inwall/{n}"][::60].tolist())
i = len(t) - 1
for n, c in C.items():
    pos, q = raw[f"{n}/pos"][i], raw[f"{n}/quat"][i]
    loc = (P[i] - pos) @ st.quat_to_R(q)
    _, wall = c.classify_local(loc)
    for j in np.where(wall)[0]:
        r = np.hypot(*loc[j, :2])
        first = next(k for k in range(len(t)) if True)
        print(f"  {n} in-wall particle {j}: local {np.round(loc[j], 4)} r={r:.4f} phi={np.degrees(np.arctan2(loc[j, 1], loc[j, 0])):.0f}")
# spilled particles at the end: where (station frame)
labels = []
for n, c in C.items():
    pos, q = raw[f"{n}/pos"][i], raw[f"{n}/quat"][i]
    cav, _ = c.classify_local((P[i] - pos) @ st.quat_to_R(q))
    labels.append(cav)
out = ~np.any(labels, axis=0)
print("outside every container at the end:", int(out.sum()))
if out.any():
    q = P[i][out]
    print("  x range", q[:, 0].min().round(3), q[:, 0].max().round(3), " y", q[:, 1].min().round(3), q[:, 1].max().round(3),
          " z", q[:, 2].min().round(4), q[:, 2].max().round(4))
    # when did they leave? first frame each was outside
    idx = np.where(out)[0]
    first = []
    for j in idx[:200]:
        for k in range(len(t)):
            inside = False
            for n, c in C.items():
                pos, qq = raw[f"{n}/pos"][k], raw[f"{n}/quat"][k]
                cav, _ = c.classify_local(((P[k][j] - pos) @ st.quat_to_R(qq))[None])
                inside |= bool(cav[0])
            if not inside:
                first.append(float(t[k]))
                break
    print("  left a container at t =", np.round(np.unique(np.round(first, 1)), 1).tolist())
