"""Mahjong station runs, checks, export and previews.

    python run.py validate   # no robot: scattered start settles; solved state (all 14 in the rack) holds; success()
    python run.py demo       # robot: stands two tiles in their slots; Recorder export, preview video, stills
    python run.py all        # both (each in its own process), then merges out/mahjong/checks.json

Run with ~/simquarry/genesis-world/.venv/bin/python. Rendering uses Mesa's software EGL (set below) so the preview
camera never touches the shared GPU; physics is gs.cpu, float64.
"""
import json
import math
import os
import subprocess
import sys
import time

os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spec  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "out", "mahjong")
STILLS = os.path.join(OUT, "stills")

# demo plan. "rack": (arm, tile, slot) for a face-up tile. "flip": (arm A, arm B, tile, slot, spot, face_dir) for a
# face-down tile: A stands it up on the table at `spot` facing B, B regrasps it and racks it. The left arm racks a
# face-up tile from the middle of the table; the right arm takes the face-up tile off the top of the right-hand stack;
# then the face-down Red dragon from the far left is flipped by the left arm and racked by the right arm.
DEMO_PLAN = [
    ("rack", ("left", "tile_07_2m", 1)),
    ("rack", ("right", "tile_11_6p", 5)),
    ("flip", ("left", "right", "tile_01_7z", 12, (0.0, -0.20), (1.0, 0.0, 0.0))),
]


def plan_targets():
    """[(tile, slot)] of the demo plan."""
    return [(a[1], a[2]) if kind == "rack" else (a[2], a[3]) for kind, a in DEMO_PLAN]

VIDEO_RES = (640, 360)
STILL_RES = (1600, 900)


def check(value, limit, ok):
    v = value.item() if isinstance(value, np.generic) else value
    return dict(value=v, limit=limit, ok=bool(ok))


# ------------------------------------------------------------------------------------------- geometry (independent)
def obb_overlap(c1, R1, h1, c2, R2, h2):
    """Separating-axis test for two oriented boxes. Returns the smallest overlap depth (m) over the 15 axes: > 0 means
    the boxes interpenetrate by that much, <= 0 means separated (value = -gap along the best separating axis)."""
    axes = [R1[:, i] for i in range(3)] + [R2[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            a = np.cross(R1[:, i], R2[:, j])
            n = np.linalg.norm(a)
            if n > 1e-6:
                axes.append(a / n)
    d = c2 - c1
    best = np.inf
    for L in axes:
        r1 = np.sum(h1 * np.abs(R1.T @ L))
        r2 = np.sum(h2 * np.abs(R2.T @ L))
        best = min(best, r1 + r2 - abs(d @ L))
    return best


def world_boxes(env):
    """Every collision box in the scene except the robot: (owner, centre, R, half-extents)."""
    from station import qmat, TABLE_Z
    import layout
    boxes = []
    t = layout.TABLE
    boxes.append(("table", np.array([0, 0, TABLE_Z - 0.5 * t["top_thickness"]]), np.eye(3),
                  0.5 * np.array([t["size_x"], t["size_y"], t["top_thickness"]])))
    rp, rq = env.rack_pose()
    Rr = qmat(rq)
    for name, c, s, rx in spec.rack_boxes():
        ca, sa = math.cos(rx), math.sin(rx)
        Rb = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        boxes.append(("rack", rp + Rr @ np.array(c), Rr @ Rb, 0.5 * np.array(s)))
    P, Q = env.tile_poses()
    for i, tl in enumerate(env.tiles):
        boxes.append((tl.name, P[i], qmat(Q[i]), 0.5 * np.array([spec.TILE_W, spec.TILE_H, spec.TILE_T])))
    return boxes


def max_overlaps(env):
    """Largest box-box overlap per pair category (tile-tile, tile-table, tile-rack, rack-table), in mm."""
    B = world_boxes(env)
    out = {"tile-tile": -1e9, "tile-table": -1e9, "tile-rack": -1e9, "rack-table": -1e9}
    worst = {}
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            a, b = B[i][0], B[j][0]
            if a == b:
                continue
            ka = "tile" if a.startswith("tile") else a
            kb = "tile" if b.startswith("tile") else b
            key = "-".join(sorted([ka, kb], key=lambda k: ["tile", "rack", "table"].index(k)))
            if key not in out:
                continue
            if np.linalg.norm(B[i][1] - B[j][1]) > np.linalg.norm(B[i][3]) + np.linalg.norm(B[j][3]):
                continue
            ov = 1000 * obb_overlap(B[i][1], B[i][2], B[i][3], B[j][1], B[j][2], B[j][3])
            if ov > out[key]:
                out[key] = ov
                worst[key] = (a, b)
    return {k: round(float(v), 4) for k, v in out.items()}, worst


def export_frame_error_mm(env):
    """Max |bounds(Genesis visual vertices in the recorded link frame) - bounds(exported OBJ)| over rack + tiles: proves
    the exported mesh frame is the frame whose pose is recorded (Genesis re-anchors free bodies at their COM)."""
    import trimesh
    import station
    from station import qmat
    err = 0.0
    items = [(env.rack, os.path.join(station.ASSETS, "rack", "rack.obj"))] + [(t.entity, t.mesh) for t in env.tiles]
    for ent, obj in items:
        p = np.asarray(ent.get_pos().cpu().numpy(), float)
        R = qmat(ent.get_quat().cpu().numpy())
        Vw = np.asarray(ent.links[0].vgeoms[0].get_vverts().cpu().numpy(), float).reshape(-1, 3)
        Vl = (Vw - p) @ R
        m = trimesh.load(obj, force="mesh", process=False)
        err = max(err, 1000 * float(np.abs(np.stack([Vl.min(0), Vl.max(0)]) - m.bounds).max()))
    return err


# --------------------------------------------------------------------------------------------------------- validate
def settle_run(layout_name, T=6.0, T_settle=1.0):
    import station
    env = station.build(layout_name=layout_name, robot=False,
                        cameras={"rack_still": station.CAMERAS["rack_full"][:3] + (STILL_RES,),
                                 "tiles_still": station.CAMERAS["tiles"][:3] + (STILL_RES,)})
    dt = env.dt
    P0, Q0 = env.tile_poses()
    n1, n2 = int(round(T_settle / dt)), int(round(T / dt))
    pen_max, pen_first, vmax_late, disp_max = 0.0, 0.0, 0.0, 0.0
    for i in range(n2):
        env.scene.step()
        if i + 1 == n1:
            P1, Q1 = env.tile_poses()
            R1 = env.rack_pose()
        c = env.contacts()
        if len(c["penetration"]):
            if i + 1 >= n1:
                pen_max = max(pen_max, float(c["penetration"].max()))
            else:
                pen_first = max(pen_first, float(c["penetration"].max()))
        if i % 25 == 0:
            P, _ = env.tile_poses()
            disp_max = max(disp_max, float(np.linalg.norm(P - P0, axis=1).max()))
        if i + 1 > n2 - int(round(1.0 / dt)):
            lin, ang = env.tile_speeds()
            vmax_late = max(vmax_late, float(lin.max()))
    P2, Q2 = env.tile_poses()
    R2 = env.rack_pose()
    drift = np.linalg.norm(P2 - P1, axis=1)
    rot = np.array([math.degrees(2 * math.acos(min(1.0, abs(float(a @ b))))) for a, b in zip(Q1, Q2)])
    ov, worst = max_overlaps(env)
    ok, det = env.success()
    imgs = {n: np.asarray(c.render()[0]) for n, c in env.cameras.items()}
    res = dict(layout=layout_name, drift_mm=1000 * float(drift.max()), rot_deg=float(rot.max()),
               rack_drift_mm=1000 * float(np.linalg.norm(R2[0] - R1[0])), vmax_last1s_mm_s=1000 * vmax_late,
               pen_max_mm=1000 * pen_max, pen_first_1s_mm=1000 * pen_first, disp_total_max_mm=1000 * disp_max, overlaps_mm=ov,
               overlap_worst_pairs={k: list(v) for k, v in worst.items()}, success=ok, n_correct=det["n_correct"],
               slots=det["slots"], tile_z=P2[:, 2].tolist(),
               tile_names=[t.name for t in env.tiles], tile_notes=[t.start.note for t in env.tiles])
    return res, imgs


def validate():
    import genesis as gs
    from PIL import Image
    os.makedirs(STILLS, exist_ok=True)
    checks, info = {}, {}
    t0 = time.time()
    sc, imgs = settle_run("scattered")
    Image.fromarray(imgs["tiles_still"]).save(os.path.join(STILLS, "settled_tiles_closeup.png"))
    info["scattered"] = sc
    # tiles lying on the table / on another tile end at the right height (face-down / stacked included)
    T = spec.TILE_T
    zerr = []
    for name, note, z in zip(sc["tile_names"], sc["tile_notes"], sc["tile_z"]):
        want = 0.75 + (1.5 if note.startswith("stacked") else 0.5) * T
        zerr.append(abs(z - want))
    checks["settle_drift_1s_to_6s_mm"] = check(sc["drift_mm"], "<= 0.05", sc["drift_mm"] <= 0.05)
    checks["settle_rotation_1s_to_6s_deg"] = check(sc["rot_deg"], "<= 0.05", sc["rot_deg"] <= 0.05)
    checks["settle_max_speed_last_1s_mm_s"] = check(sc["vmax_last1s_mm_s"], "<= 0.1", sc["vmax_last1s_mm_s"] <= 0.1)
    checks["settle_total_motion_mm"] = check(sc["disp_total_max_mm"], "<= 0.3 (0.1 mm spawn gap, 0.2 for stacks)",
                                             sc["disp_total_max_mm"] <= 0.3)
    checks["settle_solver_penetration_first_1s_mm"] = check(sc["pen_first_1s_mm"], "<= 0.1 (landing from the spawn gap)",
                                                            sc["pen_first_1s_mm"] <= 0.1)
    checks["settle_rest_height_error_mm"] = check(1000 * max(zerr), "<= 0.1 (on table / on the tile below)",
                                                  1000 * max(zerr) <= 0.1)
    checks["settle_solver_penetration_mm"] = check(sc["pen_max_mm"], "<= 0.05 (t >= 1 s)", sc["pen_max_mm"] <= 0.05)
    for k, v in sc["overlaps_mm"].items():
        checks[f"settle_box_overlap_{k}_mm"] = check(v, "<= 0.05 (independent SAT test on the final poses; < 0 = gap)", v <= 0.05)
    checks["scattered_success_is_false"] = check(sc["success"], "False (nothing in the rack yet)", not sc["success"])
    print("scattered", json.dumps({k: v for k, v in sc.items() if k not in ("slots", "tile_z", "tile_names",
                                                                           "tile_notes")}), flush=True)
    info["wall_scattered_s"] = time.time() - t0
    return checks, info


def validate_solved():
    from PIL import Image
    checks, info = {}, {}
    so, imgs = settle_run("solved")
    Image.fromarray(imgs["rack_still"]).save(os.path.join(STILLS, "solved_rack_closeup.png"))
    info["solved"] = so
    checks["solved_success"] = check(so["success"], "True (all 14 tiles seated in order)", so["success"])
    checks["solved_n_correct"] = check(so["n_correct"], "== 14", so["n_correct"] == 14)
    checks["solved_drift_1s_to_6s_mm"] = check(so["drift_mm"], "<= 0.05 (tiles leaning in the rack do not creep)",
                                               so["drift_mm"] <= 0.05)
    checks["solved_rack_drift_mm"] = check(so["rack_drift_mm"], "<= 0.05", so["rack_drift_mm"] <= 0.05)
    checks["solved_max_speed_last_1s_mm_s"] = check(so["vmax_last1s_mm_s"], "<= 0.1", so["vmax_last1s_mm_s"] <= 0.1)
    checks["solved_solver_penetration_mm"] = check(so["pen_max_mm"], "<= 0.05 (t >= 1 s)", so["pen_max_mm"] <= 0.05)
    checks["solved_solver_penetration_first_1s_mm"] = check(so["pen_first_1s_mm"], "<= 0.1 (landing from the spawn gap)",
                                                            so["pen_first_1s_mm"] <= 0.1)
    for k, v in so["overlaps_mm"].items():
        checks[f"solved_box_overlap_{k}_mm"] = check(v, "<= 0.05", v <= 0.05)
    print("solved", json.dumps({k: v for k, v in so.items() if k not in ("slots", "tile_z", "tile_names",
                                                                       "tile_notes")}), flush=True)
    return checks, info


# ------------------------------------------------------------------------------------------------------------- demo
def demo():
    import station
    import demo as D
    from PIL import Image
    import imageio.v2 as imageio
    sys.path.insert(0, ROOT)
    from export import Recorder
    import layout

    os.makedirs(STILLS, exist_ok=True)
    cams = {n: c[:3] + (VIDEO_RES,) for n, c in station.CAMERAS.items() if n != "rack_full"}
    cams.update({f"{n}_still": c[:3] + (STILL_RES,) for n, c in station.CAMERAS.items()})
    env = station.build(layout_name="scattered", robot=True, cameras=cams)
    cart = env.robot
    tiles = {t.name: t for t in env.tiles}
    names = env.link_names()
    robot_links = {l.idx for l in cart.entity.links}
    tile_link = {t.entity.links[0].idx: t.name for t in env.tiles}
    rack_link = env.rack.links[0].idx
    table_link = env.table.links[0].idx

    # ---------------------------------------------------------------- export
    rec = Recorder(OUT, station="mahjong", fps=layout.RENDER_FPS)
    for t in env.tiles:
        rec.add_rigid(t.name, t.mesh, material=dict(kind="plastic", color=station.TILE_COLOR, roughness=0.25,
                                                       texture=f"meshes/tile_{t.kind}.png"))
    rec.add_rigid("rack", os.path.join(station.ASSETS, "rack", "rack.obj"),
                  material=dict(kind="wood", color=station.RACK_COLOR, roughness=0.55, texture="meshes/rack_wood.png"))
    cart.register(rec)

    def object_poses():
        P, Q = env.tile_poses()
        d = {t.name: (P[i], Q[i]) for i, t in enumerate(env.tiles)}
        d["rack"] = env.rack_pose()
        return d

    # ---------------------------------------------------------------- per-step monitors
    video = imageio.get_writer(os.path.join(OUT, "preview.mp4"), fps=layout.RENDER_FPS, codec="libx264",
                               quality=None, ffmpeg_params=["-crf", "20", "-pix_fmt", "yuv420p", "-preset", "medium"])
    mon = dict(frame=0, pen_by_pair={}, robot_contacts={}, robot_zmin=1e9, frames_recorded=0)
    still_at = {}

    def grid():
        im = [np.asarray(env.cameras[n].render()[0]) for n in ("overview", "top", "tiles", "rack")]
        return np.concatenate([np.concatenate(im[:2], 1), np.concatenate(im[2:], 1)], 0)

    def on_step(r):
        # contacts: penetration per pair category, robot contacts by partner
        c = env.contacts()
        for a, b, pn in zip(c["link_a"], c["link_b"], c["penetration"]):
            a, b = int(a), int(b)
            ka = "robot" if a in robot_links else ("tile" if a in tile_link else names.get(a, "?"))
            kb = "robot" if b in robot_links else ("tile" if b in tile_link else names.get(b, "?"))
            key = "-".join(sorted([ka, kb]))
            mon["pen_by_pair"][key] = max(mon["pen_by_pair"].get(key, 0.0), float(pn))
            if a in robot_links or b in robot_links:
                other = b if a in robot_links else a
                partner = tile_link.get(other, names.get(other, str(other)))
                mon["robot_contacts"][partner] = mon["robot_contacts"].get(partner, 0) + 1
        # record at 30 fps
        if r.t + 1e-9 >= mon["frame"] / layout.RENDER_FPS:
            cart.record(rec, r.t, rigid=object_poses())
            video.append_data(grid())
            mon["frame"] += 1
        for key, (tw, cam) in list(still_at.items()):
            if r.t >= tw:
                Image.fromarray(np.asarray(env.cameras[cam].render()[0])).save(os.path.join(STILLS, key))
                del still_at[key]

    r = D.Runner(env, on_step)
    t0 = time.time()
    # opening frame (t = 0) and stills of the start state
    cart.record(rec, 0.0, rigid=object_poses())
    video.append_data(grid())
    mon["frame"] = 1
    r.wait(0.6)
    for n in ("overview", "top", "tiles"):
        Image.fromarray(np.asarray(env.cameras[f"{n}_still"].render()[0])).save(os.path.join(STILLS, f"start_{n}.png"))
    P_start, Q_start = env.tile_poses()
    R_start = env.rack_pose()
    results = []
    for kind, args in DEMO_PLAN:
        if kind == "rack":
            side, name, slot = args
            arms = side
            still_at[f"{name}_in_hand_overview.png"] = (r.t + 6.0, "overview_still")
        else:
            side_a, side_b, name, slot, spot, face_dir = args
            arms = f"{side_a} (stand up) + {side_b} (rack)"
            still_at[f"{name}_turned_upright_tiles.png"] = (r.t + 8.5, "tiles_still")
            still_at[f"{name}_standing_tiles.png"] = (r.t + 12.5, "tiles_still")
            still_at[f"{name}_regrasp_tiles.png"] = (r.t + 16.0, "tiles_still")
        tile = tiles[name]
        try:
            if kind == "rack":
                r.stand_in_rack(side, tile, slot)
            else:
                r.flip_and_rack(side_a, side_b, tile, slot, spot, face_dir)
            err = None
        except D.IKError as e:
            err = str(e)
            r.log(f"IK failure: {e}")
        results.append(dict(mode=kind, arms=arms, tile=name, kind=tile.kind, start=tile.start.note, slot=slot,
                            error=err))
    r.wait(1.5)
    for n in ("overview", "top", "rack", "rack_full"):
        Image.fromarray(np.asarray(env.cameras[f"{n}_still"].render()[0])).save(os.path.join(STILLS, f"end_{n}.png"))
    video.close()
    ok, det = env.success()
    rows = {row["tile"]: row for row in det["tiles"]}
    P_end, Q_end = env.tile_poses()
    moved = {t.name: 1000 * float(np.linalg.norm(P_end[i] - P_start[i])) for i, t in enumerate(env.tiles)}
    demo_names = {n for n, _ in plan_targets()}
    others_moved = max(v for k, v in moved.items() if k not in demo_names)
    rack_moved = 1000 * float(np.linalg.norm(env.rack_pose()[0] - R_start[0]))
    meta_tiles = [dict(name=t.name, kind=t.kind, label=spec.kind_name(t.kind), start=t.start.note,
                       target_slots=[k for k, kk in enumerate(spec.HAND_SORTED) if kk == t.kind]) for t in env.tiles]
    rec.close(extra_meta=dict(
        task="Stand all 14 tiles in the rack facing the robot, sorted: characters, dots, bamboo, winds, dragons; "
             "numbers ascending left to right (robot's view).",
        hand_sorted=spec.HAND_SORTED, tiles=meta_tiles, demo_plan=[dict(mode=r_["mode"], arms=r_["arms"], tile=r_["tile"],
                                                                       slot=r_["slot"]) for r_ in results],
        demo_end=dict(n_correct=det["n_correct"], success=ok), events=r.events,
        notes="All objects are free rigid bodies (Genesis rigid solver, gs.cpu, float64, dt 2 ms). Robot driven by IK + "
              "joint PD; tiles moved only by friction grasps. Rack is a free body resting on the table."))

    checks, info = {}, {}
    for res in results:
        row = rows[res["tile"]]
        res.update({k: row.get(k) for k in ("slot", "seated", "upright", "facing", "at_rest", "correct", "dx_mm",
                                             "dbu_mm", "up_err_deg", "face_err_deg")})
        res["target_slot"] = dict(plan_targets())[res["tile"]]
        checks[f"demo_{res['tile']}_in_slot_{res['target_slot']}"] = check(
            bool(row["correct"] and row["slot"] == res["target_slot"]),
            "seated (<= 5 mm along, <= 4 mm in profile), upright & facing robot (<= 10 deg), at rest, right slot",
            row["correct"] and row["slot"] == res["target_slot"] and res["error"] is None)
    checks["demo_n_correct_slots"] = check(det["n_correct"], f"== {len(DEMO_PLAN)}", det["n_correct"] == len(DEMO_PLAN))
    slip = max(r.slip_mm.values()) if r.slip_mm else 0.0
    checks["demo_in_hand_slip_mm"] = check(slip, "<= 5 (tile pose in the gripper, lift -> 8 mm above the seat)", slip <= 5.0)
    checks["demo_untouched_tiles_moved_mm"] = check(others_moved, "<= 0.3 (spawn drop only)", others_moved <= 0.3)
    checks["demo_rack_moved_mm"] = check(rack_moved, "<= 1.0", rack_moved <= 1.0)
    pen = {k: 1000 * v for k, v in mon["pen_by_pair"].items()}
    obj_pen = max([v for k, v in pen.items() if "robot" not in k] or [0.0])
    rob_pen = max([v for k, v in pen.items() if "robot" in k] or [0.0])
    checks["demo_object_penetration_mm"] = check(obj_pen, "<= 0.1 (tile/rack/table contacts, every step)",
                                                 obj_pen <= 0.1)
    checks["demo_finger_penetration_mm"] = check(rob_pen, "<= 1.0 (pads squeezing at 6 N, every step)", rob_pen <= 1.0)
    bad = {k: v for k, v in mon["robot_contacts"].items() if k not in demo_names}
    checks["demo_robot_touches_only_its_tiles"] = check(bad, "{} (no robot contact with table, rack, other tiles)",
                                                        not bad)
    ov, worst = max_overlaps(env)
    for k, v in ov.items():
        checks[f"demo_end_box_overlap_{k}_mm"] = check(v, "<= 0.05 (< 0 = gap)", v <= 0.05)
    fe = export_frame_error_mm(env)
    checks["export_mesh_frame_error_mm"] = check(fe, "<= 0.01 (Genesis visual vertices mapped by the recorded pose vs "
                                                     "the exported OBJ bounds, rack + 14 tiles)", fe <= 0.01)
    info.update(results=results, in_hand_slip_mm=r.slip_mm, moved_mm=moved, penetration_by_pair_mm=pen, robot_contact_steps=mon["robot_contacts"],
                frames=mon["frame"], sim_time_s=r.t, wall_s=time.time() - t0, events=r.events)
    print(json.dumps(checks, indent=1), flush=True)
    return checks, info


# ------------------------------------------------------------------------------------------------------------ main
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    if mode == "all":
        for m in ("validate", "solved", "demo"):
            subprocess.run([sys.executable, __file__, m], check=True)
        merged = dict(station="mahjong", checks={}, info={})
        for m in ("validate", "solved", "demo"):
            d = json.load(open(os.path.join(OUT, f"checks_{m}.json")))
            merged["checks"].update(d["checks"])
            merged["info"][m] = d["info"]
        merged["all_ok"] = all(c["ok"] for c in merged["checks"].values())
        merged["n_checks"] = len(merged["checks"])
        merged["n_failed"] = sum(not c["ok"] for c in merged["checks"].values())
        json.dump(merged, open(os.path.join(OUT, "checks.json"), "w"), indent=1, default=float)
        print(f"checks: {merged['n_checks'] - merged['n_failed']}/{merged['n_checks']} ok")
        for k, c in merged["checks"].items():
            if not c["ok"]:
                print("FAILED", k, c)
        return
    fn = dict(validate=validate, solved=validate_solved, demo=demo)[mode]
    checks, info = fn()
    json.dump(dict(checks=checks, info=info), open(os.path.join(OUT, f"checks_{mode}.json"), "w"), indent=1,
              default=float)


if __name__ == "__main__":
    main()
