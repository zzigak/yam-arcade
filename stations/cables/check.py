"""Checks for a cables-station run: reads out/cables/{log.json, traj.npz, meta.json} and writes checks.json
(each check: value, limit, pass). Usage: python check.py [out_dir]"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import station as S  # noqa: E402
import layout  # noqa: E402

LIMITS = dict(
    seg_ratio=(0.9, 1.1),          # every segment, every recorded step
    below_table_mm=-0.5,           # centreline height above the table top minus radius, all frames
    settle_vmax=0.02,              # m/s, fastest cable vertex at the end of the settle
    settle_ke=2e-5,                # J, cable kinetic energy at the end of the settle
    retention=0.5,                 # in the pile: 1 - RMSD(settled, rest) / RMSD(straight, rest), per cable
    curvature_min=0.75,            # in the pile: summed |turning angle| settled / rest >= this (no straightening)
    iso_retention=0.9,             # each cable dropped alone (tests/rest_shape_test.py)
    iso_curvature=(0.9, 1.1),
    iso_drift_mm=0.5,              # RMSD change over the last second, alone on the floor (no ongoing creep)
    frozen_frames=10,              # a cable >2 cm above any support may not stay bit-identical this long
    bin_tilt_deg=3.0,
    bin_shift_mm=10.0,
    pick_in_bin=0.5,               # fraction of the picked cable's vertices inside its bin at the end (majority:
                                   # the YAM Ultra lifts only ~0.25 m, so a cable longer than that trails over the rim)
    robot_link_hits=0,             # cable vertices inside a robot link's hull (links other than the IPC pads)
)


def quat_to_mat(q):
    return S.quat_to_mat(q)


def check(out_dir):
    L = json.load(open(os.path.join(out_dir, "log.json")))
    meta = json.load(open(os.path.join(out_dir, "meta.json")))
    tr = np.load(os.path.join(out_dir, "traj.npz"))
    log, settle, demo = L["log"], L["settle"], L.get("demo")
    res = {}

    def put(name, value, limit, ok, note=None):
        res[name] = dict(value=value, limit=limit, pass_=bool(ok))
        if note:
            res[name]["note"] = note

    names = [n for n, o in meta["objects"].items() if o["kind"] == "rod"]
    # stretch
    lo = min(r["seg_ratio"][0] for r in log if "seg_ratio" in r)
    hi = max(r["seg_ratio"][1] for r in log if "seg_ratio" in r)
    put("segment_stretch", [lo, hi], LIMITS["seg_ratio"], LIMITS["seg_ratio"][0] <= lo and hi <= LIMITS["seg_ratio"][1],
        "min/max segment length / rest, all cables, every sim step")
    # interpenetration (al-ipc lets contacts overlap transiently): smallest surface gap between different cables and
    # between non-adjacent segments of the same cable, every recorded frame (exported part: from the plug boot on)
    radii = {n: S.CABLE_BY_NAME[n]["radius"] for n in names}
    worst_pair, worst_self = np.inf, np.inf
    for k in range(len(tr["t"])):
        P = {n: tr[f"{n}/pts"][k].astype(np.float64) for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                worst_pair = min(worst_pair, float(S.seg_seg_dist(P[a], P[b]).min() - radii[a] - radii[b]))
            D = S.seg_seg_dist(P[a], P[a])
            m = np.abs(np.arange(len(D))[:, None] - np.arange(len(D))[None]) >= 2
            worst_self = min(worst_self, float(D[m].min() - 2 * radii[a]))
    put("no_cable_interpenetration_mm", dict(between_cables=1e3 * worst_pair, within_a_cable=1e3 * worst_self), -0.5,
        1e3 * min(worst_pair, worst_self) >= -0.5,
        "min surface gap over all recorded frames (negative = overlap); IPC keeps it >= 0, al-ipc may dip")
    # NaN
    nan = any(c["nan"] for r in log for c in r["cables"].values())
    put("no_nan", not nan, True, not nan)
    # below table: the centreline may not come closer to the table top than the radius (tolerance 0.5 mm)
    zg = min(c["zgap_min"] for r in log for c in r["cables"].values())
    put("below_table_mm", 1e3 * zg, LIMITS["below_table_mm"], 1e3 * zg >= LIMITS["below_table_mm"],
        "min over frames and cables of (lowest centreline z - table top - radius)")
    # table bounds (every frame) and not inside a bin at the end of the settle
    T = layout.TABLE
    out_xy = 0.0
    for r in log:
        for c in r["cables"].values():
            out_xy = max(out_xy, max(abs(c["xy_min"][0]), abs(c["xy_max"][0])) - T["size_x"] / 2,
                         max(abs(c["xy_min"][1]), abs(c["xy_max"][1])) - T["size_y"] / 2)
    put("cables_on_table", 1e3 * out_xy, 0.0, out_xy <= 0.0, "max overhang past the table edge, mm (<= 0: on the table)")
    sp = settle["success"]["per_cable"]
    in_bins = [n for n, v in sp.items() if max(v["fractions"].values()) > 0.0]
    put("pile_not_in_bins", in_bins, [], not in_bins, "cables with any vertex inside a bin after the drop")
    # settle
    i_settle = max(i for i, r in enumerate(log) if r.get("phase", "drop") == "drop")
    tail = log[max(1, i_settle - 4):i_settle + 1]
    vmax = max(r["vmax"] for r in tail)
    ke = max(r["ke"] for r in tail)
    put("settled_vmax", vmax, LIMITS["settle_vmax"], vmax < LIMITS["settle_vmax"], "m/s, last 5 frames of the settle")
    put("settled_ke", ke, LIMITS["settle_ke"], ke < LIMITS["settle_ke"], "J, last 5 frames of the settle")
    # rest-shape retention
    ret = settle["retention"]
    put("rest_shape_retention", {n: round(v["retention"], 3) for n, v in ret.items()}, LIMITS["retention"],
        all(v["retention"] >= LIMITS["retention"] for v in ret.values()),
        "per cable 1 - RMSD(settled, rest)/RMSD(straight, rest), best rigid alignment; 1 = rest shape, 0 = straight")
    put("pile_curvature_kept", {n: round(v["curvature_ratio"], 3) for n, v in ret.items()}, LIMITS["curvature_min"],
        all(v["curvature_ratio"] >= LIMITS["curvature_min"] for v in ret.values()),
        "per cable sum |turning angle| in the pile / rest: none straightened. One-sided: the softest leads drape over "
        "thicker cables and pick up extra curvature (loading, not loss of shape); the two-sided test is "
        "rest_shape_isolated")
    iso = os.path.join(S.ROOT, "out", "cables", "tests", "rest_shape_test.json")
    if os.path.exists(iso):
        J = json.load(open(iso))
        sm, ilog = J["summary"], J["log"]
        t_end = ilog[-1]["t"]
        one_s = min(ilog, key=lambda r: abs(r["t"] - (t_end - 1.0)))
        for k in sm:  # creep over the last second, after the landing transient (stacked coil turns sag ~2.5 s)
            sm[k]["drift_mm_last_s"] = ilog[-1][k]["rmsd_mm"] - one_s[k]["rmsd_mm"]
        framed = {k: v for k, v in sm.items() if not k.endswith("_KRB_only")}
        ctrl = {k: v for k, v in sm.items() if k.endswith("_KRB_only")}
        ok = all(v["final"]["retention"] >= LIMITS["iso_retention"]
                 and LIMITS["iso_curvature"][0] <= v["final"]["curvature_ratio"] <= LIMITS["iso_curvature"][1]
                 and abs(v["drift_mm_last_s"]) < LIMITS["iso_drift_mm"] for v in framed.values())
        put("rest_shape_isolated",
            {k: dict(retention=round(v["final"]["retention"], 4), curvature=round(v["final"]["curvature_ratio"], 4),
                     rmsd_mm=round(v["final"]["rmsd_mm"], 2), drift_last_s_mm=round(v["drift_mm_last_s"], 3))
             for k, v in framed.items()},
            dict(retention=LIMITS["iso_retention"], curvature=LIMITS["iso_curvature"], drift_last_s_mm=LIMITS["iso_drift_mm"]),
            ok, "each cable dropped 3 cm alone onto the floor, 4 s (tests/rest_shape_test.py)")
        put("rest_shape_control_plain_rod",
            {k: dict(retention=round(v["final"]["retention"], 3), curvature=round(v["final"]["curvature_ratio"], 3))
             for k, v in ctrl.items()}, "reported", True,
            "same cables as plain Genesis FEM.Rod (libuipc KRB, straight rest shape): they straighten")
    # tangle
    tg = settle["tangle"]
    put("tangle_crossings", tg["crossings"], "> 0", tg["crossings"] > 0, "top-view crossings between different cables")
    put("tangle_contacts", dict(segment_pairs=tg["contacts"], cable_pairs=tg["touching_pairs"]), "> 0",
        tg["contacts"] > 0, "segment pairs of different cables within 2 mm")
    put("tangle_interlocked", dict(pairs=tg["interlocked_pairs"], woven=tg["middle"]), "reported", True,
        "pairs crossing both over and under each other; cables lying over one cable and under another")
    # frozen
    worst = 0
    for n in names:
        run = 0
        for a, b in zip(log, log[1:]):
            ca, cb = a["cables"][n], b["cables"][n]
            if ca["zmin"] - S.ZT > 0.02 + 0.12 and ca["com"] == cb["com"]:  # well above the table and bins
                run += 1
                worst = max(worst, run)
            else:
                run = 0
    put("no_frozen_cable", worst, LIMITS["frozen_frames"], worst < LIMITS["frozen_frames"],
        "longest run of bit-identical frames of a cable in mid-air")
    # bins
    b0, b1 = log[0]["bins"], log[i_settle]["bins"]
    tilt = shift = 0.0
    for col in b0:
        R = quat_to_mat(np.array(log[-1]["bins"][col]["quat"]))
        tilt = max(tilt, float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))))
        shift = max(shift, 1e3 * float(np.linalg.norm(np.array(b1[col]["pos"])[:2] - np.array(b0[col]["pos"])[:2])))
    zb = min(r["bins"][c]["pos"][2] for r in log for c in r["bins"])
    put("bins_upright", tilt, LIMITS["bin_tilt_deg"], tilt < LIMITS["bin_tilt_deg"], "max bin tilt, deg")
    put("bins_stay_during_drop", shift, LIMITS["bin_shift_mm"], shift < LIMITS["bin_shift_mm"], "mm")
    put("bins_not_in_table", 1e3 * (zb - S.ZT), 0.0, zb - S.ZT >= -0.0005, "min bin bottom height above the table, mm")
    wz = layout.WORK_ZONE
    inside = all(wz["x"][0] <= p[0] - S.BIN["size"][0] / 2 and p[0] + S.BIN["size"][0] / 2 <= wz["x"][1]
                 and wz["y"][0] <= p[1] - S.BIN["size"][1] / 2 and p[1] + S.BIN["size"][1] / 2 <= wz["y"][1]
                 for p in (np.array(b0[c]["pos"]) for c in b0))
    put("bins_in_work_zone", inside, True, inside)
    # robot demo
    if demo is not None and demo.get("pick") is not None:
        pk = demo["pick"]
        fr = demo["success"]["per_cable"][pk["cable"]]
        col = S.CABLE_BY_NAME[pk["cable"]]["color"]
        put("demo_pick_in_bin", fr["fractions"][col], LIMITS["pick_in_bin"], fr["fractions"][col] >= LIMITS["pick_in_bin"],
            f"fraction of {pk['cable']} inside the {col} bin at the end")
        if demo.get("feeds") is not None:
            put("demo_feeds", [dict(vertex=f["pick"]["vertex"], before=round(f["fraction_before"], 3),
                                    after=round(f["fraction_after"], 3)) for f in demo["feeds"]], "reported", True,
                "grasps of the cable body carrying the trailing part into the bin after the plug transfer")
        carry = [r for r in log if r.get("demo_phase") == "lift_carry" and "plug_in_flange" in r]
        if carry:  # slip of the plug in the flange frame while carried (from 0.3 s into the carry)
            c0 = [r for r in carry if r["t"] >= carry[0]["t"] + 0.3]
            info = S.plug_info(S.CABLE_BY_NAME[pk["cable"]]["plug"])
            gl = np.array([0.5 * sum(info["body_x"]), 0.0, 0.0])  # grip point (body centre) in the plug frame
            gp = [np.array(r["plug_in_flange"]) + np.array(r["plug_R_in_flange"]) @ gl for r in c0]
            R0 = np.array(c0[0]["plug_R_in_flange"])
            slip = max(float(np.linalg.norm(g - gp[0])) for g in gp)
            rot = max(float(np.degrees(np.arccos(np.clip((np.trace(R0.T @ np.array(r["plug_R_in_flange"])) - 1) / 2, -1, 1))))
                      for r in c0)
            put("plug_slip_mm", 1e3 * slip, 2.0, 1e3 * slip <= 2.0,
                f"grip point of the plug in the flange frame while carried; rotation {rot:.1f} deg (as the Newton check)")
            ff = np.array([r["finger_force"] for r in carry])
            put("plug_grip_force_N", dict(median=np.median(np.abs(ff), 0).round(2).tolist(), max=float(np.abs(ff).max())),
                "cap 4 N", True, "per-finger joint force (PD command, force-capped) while carried")
        if demo.get("max_arm_joint_speed_rad_s") is not None:
            v = demo["max_arm_joint_speed_rad_s"]
            put("arm_joint_speed_rad_s", v, 3.0, v <= 3.0, "fastest arm joint between consecutive targets (no IK branch jumps)")
        if "width" in pk:  # plug grasp: how far the drawn fingers sit inside the plug while carrying it
            held = [r["width"] for r in log if r.get("demo_phase") == "lift_carry"]
            intr = [1e3 * 0.5 * (pk["width"] - w) for w in held]
            put("drawn_fingers_into_plug_mm", dict(median=float(np.median(intr)), max=float(np.max(intr))), 2.0,
                np.median(intr) <= 2.0,
                "half of (plug width - drawn pad gap) while carrying; the libuipc pads hold the plug's surface, the "
                "drawn fingers trail them by grip force / coupling stiffness")
        wrong = [n for n, v in demo["success"]["per_cable"].items() if v["bin"] is not None and not v["correct"]]
        put("demo_no_wrong_bin", wrong, [], not wrong)
        hits = robot_link_hits(out_dir, meta, tr, names)
        arm = {k: v for k, v in hits.items() if not k.endswith("_gripper")}
        put("robot_links_clear_of_cables", arm, LIMITS["robot_link_hits"], sum(arm.values()) == 0,
            "cable vertices inside the convex hull of an arm link that is not a libuipc contact body, all frames")
        grip = {k: v for k, v in hits.items() if k.endswith("_gripper")}
        put("gripper_housing_vs_cables", grip, "reported", True,
            "cable vertices inside the gripper housing's visual hull (its libuipc contact body is the palm box)")
        props = robot_vs_props(out_dir, meta, tr, log)
        put("robot_links_clear_of_table_and_bins", props, 0, sum(props.values()) == 0,
            "robot link hull vertices inside the table slab or a bin, all frames (fingertips/palm excluded)")
    # robot fingertip pads above the table, plugs not in the table
    import trimesh
    tip_min = np.inf
    for obj, o in meta["objects"].items():
        if obj.startswith("robot/") and "tip_" in obj:
            V = np.asarray(trimesh.load(os.path.join(out_dir, o["mesh"]), process=False).vertices)
            for P, Q in zip(tr[f"{obj}/pos"], tr[f"{obj}/quat"]):
                tip_min = min(tip_min, float((V @ quat_to_mat(Q).T + P)[:, 2].min()))
    if np.isfinite(tip_min):
        put("pads_above_table_mm", 1e3 * (tip_min - S.ZT), -0.5, 1e3 * (tip_min - S.ZT) >= -0.5,
            "lowest fingertip mesh point over the table top, all frames")
    plug_min = np.inf
    for obj, o in meta["objects"].items():
        if obj.startswith("plug_"):
            V = np.asarray(trimesh.load(os.path.join(out_dir, o["mesh"]), process=False).vertices)
            for P, Q in zip(tr[f"{obj}/pos"], tr[f"{obj}/quat"]):
                plug_min = min(plug_min, float((V @ quat_to_mat(Q).T + P)[:, 2].min()))
    if np.isfinite(plug_min):
        put("plugs_above_table_mm", 1e3 * (plug_min - S.ZT), -0.5, 1e3 * (plug_min - S.ZT) >= -0.5,
            "lowest plug mesh point over the table top, all frames (drawn shell / pin included)")
    if L.get("timing"):
        put("timing", dict(L["timing"], gpu=L.get("gpu")), "reported", True, "sim s, wall s per phase")
    fc = L.get("bin_frame_check") or {}
    if fc:
        err = max(v["export_mesh_vs_genesis_verts_mm"] for v in fc.values())
        put("export_pose_matches_genesis", err, 2.0, err <= 2.0,
            "bin mesh x exported pose vs Genesis's own link vertices after step 1, mm (Genesis trails libuipc ~1 step)")
    ok = all(v["pass_"] for v in res.values())
    out = dict(all_pass=ok, checks=res)
    json.dump(out, open(os.path.join(out_dir, "checks.json"), "w"), indent=1, default=float)
    return out


def robot_vs_props(out_dir, meta, tr, log):
    """Frames x vertices of robot link meshes (not fingertips / palm) inside the table slab or inside a bin's box."""
    import trimesh
    T = layout.TABLE
    lo_b = np.array([-S.BIN["size"][0] / 2, -S.BIN["size"][1] / 2, 0.0])
    hi_b = np.array([S.BIN["size"][0] / 2, S.BIN["size"][1] / 2, S.BIN["size"][2]])
    out = {}
    for obj, o in meta["objects"].items():
        if not obj.startswith("robot/") or "tip_" in obj or obj.endswith("_gripper"):
            continue
        V = np.asarray(trimesh.load(os.path.join(out_dir, o["mesh"]), process=False).vertices)
        P, Q = tr[f"{obj}/pos"], tr[f"{obj}/quat"]
        cnt = 0
        for k in range(len(P)):
            W = V @ quat_to_mat(Q[k]).T + P[k]
            slab = (np.abs(W[:, 0]) < T["size_x"] / 2) & (np.abs(W[:, 1]) < T["size_y"] / 2) & \
                   (W[:, 2] < S.ZT - 0.0005) & (W[:, 2] > S.ZT - T["top_thickness"])
            cnt += int(slab.sum())
            for b in log[min(k, len(log) - 1)]["bins"].values():
                Lb = (W - np.array(b["pos"])) @ quat_to_mat(np.array(b["quat"]))
                cnt += int(np.all((Lb > lo_b) & (Lb < hi_b), axis=1).sum())
        if cnt:
            out[obj] = cnt
    return out


def robot_link_hits(out_dir, meta, tr, names):
    """Count cable vertices inside the convex hull of every exported robot part except the fingertips (whose pads are
    the libuipc contact bodies and may hold the cable with <= 1 mm mesh overlap) and the palm (also a contact body)."""
    import trimesh
    from scipy.spatial import Delaunay
    hits = {}
    for obj, o in meta["objects"].items():
        if not obj.startswith("robot/") or "tip_" in obj:
            continue
        V = np.asarray(trimesh.load(os.path.join(out_dir, o["mesh"]), process=False).vertices)
        if len(V) < 4:
            continue
        try:
            hull = Delaunay(V)
        except Exception:
            continue
        P, Q = tr[f"{obj}/pos"], tr[f"{obj}/quat"]
        cnt = 0
        for k in range(len(P)):
            R = quat_to_mat(Q[k])
            pts = np.vstack([tr[f"{n}/pts"][k] for n in names])
            loc = (pts - P[k]) @ R
            lo, hi = V.min(0), V.max(0)
            m = np.all((loc > lo) & (loc < hi), axis=1)
            if m.any():
                cnt += int((hull.find_simplex(loc[m]) >= 0).sum())
        if cnt:
            hits[obj] = cnt
    return hits


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(S.ROOT, "out", "cables")
    out = check(d)
    print("ALL PASS" if out["all_pass"] else "FAIL")
    for k, v in out["checks"].items():
        print(f"  {'ok ' if v['pass_'] else 'FAIL'} {k}: {v['value']}  (limit {v['limit']})")
