"""Settle test: water in the standing pitcher, cups on the table. Counts leaks / wall penetrations, measures volume.

usage: python settle_test.py [--secs 1.5] [--mu 2e-4] [--size 0.004]
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import station as st  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--secs", type=float, default=1.5)
ap.add_argument("--mu", type=float, default=st.LIQUID["mu"])
ap.add_argument("--gamma", type=float, default=st.LIQUID["gamma"])
ap.add_argument("--substeps", type=int, default=16)
ap.add_argument("--out", default=os.path.join(HERE, "out_settle"))
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--solver", default="WCSPH")
ap.add_argument("--stiffness", type=float, default=st.LIQUID["stiffness"])
ap.add_argument("--nofast", action="store_true")
ap.add_argument("--cf", type=float, default=st.RIGID_COUP["coup_friction"])
ap.add_argument("--water", type=float, default=st.WATER_ML)
ap.add_argument("--bd", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

cams = dict(
    front=dict(pos=(0.0, -1.0, 1.15), lookat=(0.0, 0.02, 0.82), fov=36, res=(1280, 720)),
    close=dict(pos=(0.45, -0.35, 0.98), lookat=(0.27, -0.03, 0.83), fov=40, res=(960, 720)),
)
t0 = time.time()
env = st.build(backend="cpu" if args.cpu else "gpu", pitcher_mount="free", cameras=cams, substeps=args.substeps,
               liquid=dict(mu=args.mu, gamma=args.gamma, stiffness=args.stiffness), pressure_solver=args.solver,
               fast_coupling=not args.nofast, coup=dict(coup_friction=args.cf), water_ml=args.water,
               boundary_density=bool(args.bd))
print("coupler:", env.coupler_class)
print(f"build {time.time() - t0:.1f}s; particles {env.particles().shape[0]}; z_fill {env.z_fill:.4f}", flush=True)
p0 = env.particles()
print("particle bbox", p0.min(0), p0.max(0))
init_pose = {n: env.pose(n) for n in env.links}

fps = 30
steps_per_frame = int(round(1.0 / (fps * env.scene.sim_options.dt)))
n_frames = int(args.secs * fps)
t0 = time.time()
for f in range(n_frames + 1):
    if f % 3 == 0 or f == n_frames:
        p = env.particles()
        label, in_wall, names = env.classify(p)
        n_out = int(np.sum(label < 0))
        n_wall = {n: int(w.sum()) for n, w in in_wall.items()}
        v = env.liquid.get_particles_vel()
        v = v.cpu().numpy().reshape(-1, 3) if hasattr(v, "cpu") else np.asarray(v).reshape(-1, 3)
        disp = {n: float(np.linalg.norm(env.pose(n)[0] - init_pose[n][0]) * 1e3) for n in env.links}
        ztop = np.percentile(p[:, 2], 99.5)
        print(f"t={f / fps:5.2f}s wall={time.time() - t0:6.1f}s outside={n_out} in_wall={n_wall} "
              f"vmax={np.abs(v).max():.3f} vmean={np.linalg.norm(v, axis=1).mean():.4f} z99.5={ztop:.4f} "
              f"disp_mm={ {k: round(x, 2) for k, x in disp.items()} }", flush=True)
    if f % 15 == 0 or f == n_frames:
        for cn, cam in env.cams.items():
            rgb = cam.render()[0]
            import imageio.v3 as iio
            iio.imwrite(os.path.join(args.out, f"{cn}_{f:04d}.png"), rgb)
    if f < n_frames:
        for _ in range(steps_per_frame):
            env.scene.step()

# volume calibration: free surface height in the pitcher vs the visual cavity volume below it
p = env.particles()
loc = env.to_local("pitcher", p)
label, _, _ = env.classify(p)
inp = loc[label == 0]
# surface height: mean z of the topmost layer (particles within d of the 99.9 percentile) + d/2
zt = np.percentile(inp[:, 2], 99.9)
top = inp[inp[:, 2] > zt - st.PARTICLE_SIZE]
z_surf = top[:, 2].mean() + 0.5 * st.rest_spacing()
vol = env.containers["pitcher"].volume_below(z_surf)
print(f"N in pitcher {len(inp)}  surface z {z_surf:.4f}  visual volume {vol * 1e6:.1f} ml  "
      f"v_p/d^3 = {vol / len(inp) / st.PARTICLE_SIZE ** 3:.4f} (Genesis particle volume 0.8)")
clr = env.containers["pitcher"].wall_clearance_local(inp)
print("min clearance to visual inner wall/floor (mm):", clr.min() * 1e3, " 1st pct:", np.percentile(clr, 1) * 1e3)
np.save(os.path.join(args.out, "final_particles.npy"), p)
v = env.liquid.get_particles_vel()
v = v.cpu().numpy().reshape(-1, 3) if hasattr(v, "cpu") else np.asarray(v).reshape(-1, 3)
sp = np.linalg.norm(v, axis=1)
locall = env.to_local("pitcher", p)
clr_all = env.containers["pitcher"].wall_clearance_local(locall)
o = np.argsort(-sp)[:15]
for i in o:
    print(f"  fast: speed {sp[i]:.3f} v {np.round(v[i], 3)} local {np.round(locall[i], 4)} clearance_mm {clr_all[i] * 1e3:.2f}")
for lo, hi in [(0, 0.004), (0.004, 0.008), (0.008, 0.02), (0.02, 1)]:
    m = (clr_all >= lo) & (clr_all < hi)
    print(f"  clearance [{lo * 1e3:.0f},{hi * 1e3:.0f}) mm: n={m.sum()} mean speed {sp[m].mean():.4f} p99 {np.percentile(sp[m], 99):.4f}")
zt = locall[:, 2].max()
m = locall[:, 2] > zt - 0.008
print(f"  top layers: n={m.sum()} mean speed {sp[m].mean():.4f}")
