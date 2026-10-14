"""Numerical checks on the installed assets; results are written into each info.json under "checks".
usage: python -I checks.py NAME...   (genesis venv)"""
import json, os, sys
import numpy as np
import trimesh

ROOT = os.path.expanduser("~/projects/yam-arcade/assets/sam3d")


def wall_thickness(m, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    pts, fi = trimesh.sample.sample_surface(m, n, seed=seed)
    nrm = m.face_normals[fi]
    o = pts - nrm * 1e-5
    loc, ray_idx, _ = m.ray.intersects_location(o, -nrm, multiple_hits=False)
    d = np.full(n, np.nan)
    d[ray_idx] = np.linalg.norm(loc - o[ray_idx], axis=1)
    return d[np.isfinite(d)]


def check(name):
    d = f"{ROOT}/{name}"
    info = json.load(open(f"{d}/info.json"))
    vis = trimesh.load(f"{d}/visual_lo.obj", force="mesh", process=False)
    ck = dict(visual_lo_bbox_m=np.round(vis.extents, 4).tolist(), visual_min_z_m=round(float(vis.bounds[0, 2]), 5),
              visual_center_xy_m=np.round(vis.bounds.mean(0)[:2], 5).tolist())
    if os.path.exists(f"{d}/collision_hollow.obj"):
        c = trimesh.load(f"{d}/collision_hollow.obj", force="mesh")
        cx, cy = info["collision"]["cavity_center_xy_m"]
        top = c.bounds[1, 2] + 0.05
        loc, _, _ = c.ray.intersects_location([[cx, cy, top]], [[0, 0, -1.0]], multiple_hits=True)
        first = float(loc[:, 2].max()) if len(loc) else None
        t = wall_thickness(c)
        ck.update(collision_watertight=bool(c.is_watertight), collision_winding_consistent=bool(c.is_winding_consistent),
                  collision_volume_positive=bool(c.volume > 0), collision_euler=int(c.euler_number),
                  collision_body_count=int(len(c.split(only_watertight=False))),
                  opening_ray_first_hit_z_m=None if first is None else round(first, 4),
                  opening_open=bool(first is not None and first < c.bounds[1, 2] - 0.01),
                  wall_thickness_mm_p5_p50=[round(float(np.percentile(t, 5)) * 1000, 2), round(float(np.median(t)) * 1000, 2)],
                  collision_vs_visual_bbox_diff_mm=np.round((c.extents - vis.extents) * 1000, 1).tolist())
    if name in ("pitcher", "pitcher_glass"):
        v = vis.vertices; H = vis.extents[2]
        topv = v[v[:, 2] > vis.bounds[0, 2] + 0.85 * H]
        tip = topv[np.argmin(topv[:, 0])]
        ck["spout_tip_m"] = np.round(tip, 4).tolist()
        hv = v[v[:, 0] > vis.bounds[1, 0] - 0.02]
        ck["handle_outer_x_m"] = round(float(vis.bounds[1, 0]), 4)
        ck["handle_z_range_m"] = [round(float(hv[:, 2].min()), 4), round(float(hv[:, 2].max()), 4)]
    info["checks"] = ck
    json.dump(info, open(f"{d}/info.json", "w"), indent=1)
    print(name, json.dumps(ck))


if __name__ == "__main__":
    for n in sys.argv[1:]:
        check(n)
