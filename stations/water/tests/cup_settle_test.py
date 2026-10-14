"""Leak test for every container: start water inside each (pitcher and the three cups), settle, and count particles
that leave their container or enter a wall.  usage: python cup_settle_test.py [--secs 1.0]"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import station as st  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--secs", type=float, default=1.0)
ap.add_argument("--out", default=os.path.join(HERE, "out_cup_settle"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
fill = {"pitcher": 600.0, "tumbler": 330.0, "mug": 250.0, "teacup": 165.0}
cams = dict(cups=dict(pos=(-0.45, -0.45, 1.05), lookat=(-0.02, 0.04, 0.80), fov=40, res=(1280, 720)))
env = st.build(backend="gpu", pitcher_mount="free", cameras=cams, water_in=fill)
print("coupler", env.coupler_class, "particles", env.particles().shape[0], flush=True)
label0, _, names = env.classify()
start = {n: int(np.sum(label0 == i)) for i, n in enumerate(names)}
print("start counts", start, flush=True)
fps, t0 = 30, time.time()
spf = int(round(1.0 / (fps * env.scene.sim_options.dt)))
import imageio.v3 as iio  # noqa: E402
for f in range(int(args.secs * fps) + 1):
    if f % 6 == 0:
        p = env.particles()
        label, in_wall, _ = env.classify(p)
        cnt = {n: int(np.sum(label == i)) for i, n in enumerate(names)}
        sp = env.particle_speeds()
        print(f"t={f / fps:.2f} wall={time.time() - t0:.0f}s counts={cnt} outside={int(np.sum(label < 0))} "
              f"in_wall={ {n: int(w.sum()) for n, w in in_wall.items()} } v={sp.mean():.4f}/{np.percentile(sp, 99):.3f}/"
              f"{sp.max():.2f}", flush=True)
    if f % 15 == 0:
        iio.imwrite(os.path.join(args.out, f"cups_{f:04d}.png"), env.cams["cups"].render()[0])
    for _ in range(spf):
        env.scene.step()
np.save(os.path.join(args.out, "final_particles.npy"), env.particles())
print("volumes", json.dumps(env.volumes()), "capacities", {n: round(env.infos[n]["capacity"] * 1e6) for n in names})
