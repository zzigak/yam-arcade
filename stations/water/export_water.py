"""raw.npz (from run.py) -> compositor export in out/water/ (export.py Recorder) + checks.json + preview.mp4.

    python export_water.py --raw out/water/raw_oracle [--out out/water] [--every 1]

Water surface: splashsurf (pysplashsurf 0.14, the copy already in the Genesis venv) on the particle centres with the
equal-volume particle radius r = 0.62 s (s = rest spacing, so (4/3) pi r^3 = s^3 = one particle's volume), SPH smoothing
length 2 r (kernel support 4 r = 9.2 mm), marching-cubes cell 0.75 r (1.7 mm; 0.5 r doubled the export to
~1 MB per frame for no visible gain), iso-threshold 0.6, 25 weighted-Laplacian
smoothing iterations.  Surface vertices that the kernel smoothing pushes past a container's inner wall are moved back
onto the wall (water meets the wall there); how many and how far is reported in checks.json.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import station as st  # noqa: E402
from containers import build as build_container  # noqa: E402
from station import load_raw  # noqa: E402

sys.path.insert(0, st.ROOT)
from export import Recorder  # noqa: E402
import layout  # noqa: E402

SURF = dict(smoothing_length=2.0, cube_size=0.75, iso_surface_threshold=0.6, mesh_smoothing_iters=25,
            mesh_smoothing_weights=True, mesh_cleanup=True, normals_smoothing_iters=10)
MATERIALS = dict(
    pitcher=dict(kind="glass", roughness=0.05),                                   # clear SAN body + handle
    pitcher_lid=dict(kind="plastic", color=(0.95, 0.95, 0.94), roughness=0.35),  # opaque white lid
    tumbler=dict(kind="glass", roughness=0.02),
    mug=dict(kind="ceramic", color=(0.20, 0.36, 0.55), roughness=0.25),          # blue glazed stoneware
    teacup=dict(kind="ceramic", color=(0.96, 0.95, 0.92), roughness=0.15),       # white porcelain
    water=dict(kind="water", color=(0.80, 0.90, 1.00)),
)
# SAM 3D cups: their own per-vertex colours (no `color`), glass without colour (SAM 3D paints glass opaque grey)
MATERIALS_SAM3D = dict(
    tumbler=dict(kind="glass", roughness=0.02),
    mug=dict(kind="ceramic", roughness=0.3),
    teacup=dict(kind="ceramic", roughness=0.15),
)


def particle_radius():
    return 0.62 * st.rest_spacing()


def reconstruct(pts, r):
    import pysplashsurf
    mesh_with_data, _ = pysplashsurf.reconstruction_pipeline(
        pts.astype(np.float64), particle_radius=r, compute_normals=False, multi_threading=True, **SURF)
    return np.asarray(mesh_with_data.mesh.vertices), np.asarray(mesh_with_data.mesh.triangles)


def clamp_to_cavities(V, containers, poses, skin=0.0003):
    """Move surface vertices that lie inside a container's visual wall solid back into its cavity (radially, or up
    onto the floor).  Returns V, n_moved, max_move (m)."""
    V = V.copy()
    n_moved, max_move = 0, 0.0
    for name, c in containers.items():
        pos, quat = poses[name]
        R = st.quat_to_R(quat)
        loc = (V - pos) @ R
        _, in_wall = c.classify_local(loc)
        if not in_wall.any():
            continue
        L = loc[in_wall]
        spec = c.spec
        r = np.hypot(L[:, 0], L[:, 1])
        phi = np.arctan2(L[:, 1], L[:, 0])
        zc = np.clip(L[:, 2], spec.z_floor, spec.z_rim)
        R_in = c._lookup(c._R_in, phi, zc)
        R_out = c._lookup(c._R_out, phi, zc)
        new = L.copy()
        # nearest exit: to the cavity side (radially in), the outside (radially out), the floor top, or the rim top
        d_in = r - (R_in - skin)
        d_out = (R_out + skin) - r
        d_floor = (spec.z_floor + skin) - L[:, 2]
        d_rim = (spec.z_rim + skin) - L[:, 2]
        d_floor = np.where(r < R_in, d_floor, np.inf)
        choice = np.argmin(np.stack([d_in, d_out, np.abs(d_floor), d_rim]), axis=0)
        s_in = (R_in - skin) / np.maximum(r, 1e-9)
        s_out = (R_out + skin) / np.maximum(r, 1e-9)
        new[choice == 0, :2] *= s_in[choice == 0, None]
        new[choice == 1, :2] *= s_out[choice == 1, None]
        new[choice == 2, 2] = spec.z_floor + skin
        new[choice == 3, 2] = spec.z_rim + skin
        move = np.linalg.norm(new - L, axis=1)
        n_moved += int(in_wall.sum())
        max_move = max(max_move, float(move.max()))
        loc[in_wall] = new
        V = loc @ R.T + pos
    return V, n_moved, max_move


def export_meshes(out_dir):
    """Container meshes for the compositor, in the container (link) frame: OBJ for the procedural ones, a Y-up GLB
    with the per-vertex colours for SAM 3D cups. Returns (paths, materials)."""
    import containers
    from sam3d_fit import export_visual_glb
    os.makedirs(out_dir, exist_ok=True)
    paths, mats = {}, dict(MATERIALS)
    for name in ("pitcher", "tumbler", "mug", "teacup"):
        c = build_container(name)
        if c.spec.visual_mesh is not None:
            paths[name] = export_visual_glb(c.spec, os.path.join(out_dir, f"water_{name}_sam3d.glb"))
            mats[name] = MATERIALS_SAM3D[name]
            continue
        parts = c.visual_parts()
        if name == "pitcher":
            body = trimesh.util.concatenate([parts["body"], parts["handle"]])
            paths["pitcher"] = os.path.join(out_dir, "water_pitcher.obj")
            body.export(paths["pitcher"])
            paths["pitcher_lid"] = os.path.join(out_dir, "water_pitcher_lid.obj")
            parts["lid"].export(paths["pitcher_lid"])
        else:
            m = trimesh.util.concatenate(list(parts.values()))
            paths[name] = os.path.join(out_dir, f"water_{name}.obj")
            m.export(paths[name])
    return paths, mats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default=os.path.join(st.ROOT, "out", "water"))
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--no-surface", action="store_true")
    args = ap.parse_args()
    raw = load_raw(args.raw)
    meta = json.load(open(os.path.join(args.raw, "events.json")))
    robot_poses = None
    rp = os.path.join(args.raw, "robot_poses.pkl")
    if os.path.exists(rp):
        robot_poses = pickle.load(open(rp, "rb"))

    import containers as cmod
    cmod.CUPS = meta.get("cups", "procedural")      # the cups this run was simulated with
    containers = {n: build_container(n) for n in ("pitcher", "tumbler", "mug", "teacup")}
    mesh_paths, mats = export_meshes(os.path.join(HERE, "assets", "export"))
    rec = Recorder(args.out, station="water", fps=layout.RENDER_FPS)
    for name, path in mesh_paths.items():
        rec.add_rigid(name, path, material=mats[name])
    r = particle_radius()
    rec.add_particles("water", radius=r, material=MATERIALS["water"])
    if not args.no_surface:
        rec.add_meshseq("water_surface", material=MATERIALS["water"])
    if robot_poses is not None:
        sys.path.insert(0, os.path.join(st.ROOT, "robots"))
        import yam_cart
        parts = yam_cart.export_parts("big_yam", "robot")
        for obj, link, mesh, mat in parts:
            rec.add_rigid(obj, mesh, material=mat)

    T = raw["t"]
    frames = range(0, len(T), args.every)
    surf_stats = dict(n_moved=[], max_move=[], n_verts=[], secs=[])
    for i in frames:
        poses = {n: (raw[f"{n}/pos"][i], raw[f"{n}/quat"][i]) for n in containers}
        rigid = {"pitcher": poses["pitcher"], "pitcher_lid": poses["pitcher"],
                 **{n: poses[n] for n in ("tumbler", "mug", "teacup")}}
        if robot_poses is not None:
            rigid.update(robot_poses[i])
        pts = raw["particles"][i]
        meshes = None
        if not args.no_surface:
            t0 = time.time()
            V, F = reconstruct(pts, r)
            V, n_moved, max_move = clamp_to_cavities(V, containers, poses)
            surf_stats["n_moved"].append(n_moved)
            surf_stats["max_move"].append(max_move)
            surf_stats["n_verts"].append(len(V))
            surf_stats["secs"].append(time.time() - t0)
            meshes = {"water_surface": (V, F)}
        rec.frame(float(T[i]), rigid=rigid, particles={"water": pts}, meshes=meshes)
        if i % 60 == 0:
            print(f"frame {i}/{len(T)} verts={surf_stats['n_verts'][-1] if surf_stats['n_verts'] else 0} "
                  f"moved={surf_stats['n_moved'][-1] if surf_stats['n_moved'] else 0}", flush=True)
    rec.close(extra_meta=dict(
        mode=meta["mode"],
        oracle=meta["mode"] == "oracle",
        note=("pitcher driven by a disclosed 6-DoF PD mount (oracle), not the robot" if meta["mode"] == "oracle"
              else "Big YAM right arm grasps the pitcher handle and pours (IK + joint PD)"),
        surface=dict(particle_radius=r, **SURF),
        events=meta["events"],
    ))
    with open(os.path.join(args.out, "surface_stats.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in surf_stats.items()}, f)
    print("export done:", args.out, flush=True)


if __name__ == "__main__":
    main()
