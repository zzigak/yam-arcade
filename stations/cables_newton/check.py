"""Checks for the Newton cables run: reads out/cables/newton/log.json (+ tests/rest_shape_test.json, calib_bend.json)
and writes checks.json: per check value, limit, pass. Limits follow the Genesis station (stations/cables/check.py)
where the same quantity exists.

    python3 check.py [out_dir]
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "out", "cables", "newton")
ZT = 0.75

LIMITS = dict(
    seg_ratio=(0.9, 1.1),       # every segment of every cable, every 1/60 s frame
    below_table_mm=-0.5,        # lowest centreline point above the table top minus radius
    penetration_mm=1.0,         # any contact pair, any frame (soft contacts: 6 N on a rubber pad is ~0.3-0.6 mm)
    pad_below_table_mm=-0.5,    # lowest fingertip-pad corner above the table top
    settle_vmax=0.02,           # m/s, fastest centreline node between the last two frames of the settle
    settle_ke=2e-5,             # J, cable kinetic energy (node velocities) at the end of the settle
    retention=0.5,              # in the pile, per cable
    curvature_min=0.75,         # in the pile, per cable
    iso_retention=0.9,          # alone on the floor (tests/rest_shape_test.py)
    iso_curvature=(0.9, 1.1),
    iso_drift_mm=0.5,
    control_retention_max=0.5,  # straight-rest control must spring open
    crossings_min=1,
    interlocked_min=1,
    grip_force_rel=0.2,         # hold-phase median per finger within 20 % of the 6 N target
    grip_force_max=2.0,         # transient max <= 2 x target
    plug_slip_mm=2.0,           # grip point of the plug in the flange frame while carried
    in_bin=0.5,                 # fraction of the sorted cable's nodes inside its bin (Genesis success definition)
    bin_tilt_deg=3.0,
    bin_shift_mm=10.0,
    calib_EI_ratio=0.02,        # |EI_eff / EI - 1| at the run's budget
    arm_qd_max=3.0,             # rad/s, I2RT DM4340/DM4310 joints run well below their ~ 10 rad/s no-load speed
)
GRIP = 6.0


def quat_xyzw_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def check(out_dir):
    L = json.load(open(os.path.join(out_dir, "log.json")))
    log, settle, picks, final = L["log"], L["settle"], L["picks"], L["final"]
    res = {}

    def put(name, value, limit, ok, note=None):
        res[name] = dict(value=value, limit=limit, **{"pass": bool(ok)})
        if note:
            res[name]["note"] = note

    lo = min(r["stretch"][0] for r in log)
    hi = max(r["stretch"][1] for r in log)
    put("segment_stretch", [lo, hi], LIMITS["seg_ratio"], LIMITS["seg_ratio"][0] <= lo and hi <= LIMITS["seg_ratio"][1],
        "min/max centreline segment length / rest length, all cables, every 1/60 s frame")
    nan = any(r["nan"] for r in log)
    put("no_nan", not nan, True, not nan)
    zg = min(r["table_gap_mm"] for r in log)
    put("below_table_mm", zg, LIMITS["below_table_mm"], zg >= LIMITS["below_table_mm"],
        "min over frames of (lowest centreline z over the table top - table top - radius)")
    pen = L["pen_max"]
    worst = max(pen.values()) if pen else 0.0
    put("penetration_mm", {k: round(v, 3) for k, v in sorted(pen.items(), key=lambda kv: -kv[1])}, LIMITS["penetration_mm"],
        worst <= LIMITS["penetration_mm"],
        "max overlap per contact category over all frames, re-evaluated at the frame's body poses from the solver's "
        "contact set (soft contacts: overlap = force / ke)")
    if any("pad_zmin" in r for r in log):
        pz = 1e3 * (min(r["pad_zmin"] for r in log if "pad_zmin" in r) - ZT)
        put("pads_above_table_mm", pz, LIMITS["pad_below_table_mm"], pz >= LIMITS["pad_below_table_mm"])
    if any("arm_qd_max" in r for r in log):
        qd = max(r.get("arm_qd_max", 0.0) for r in log)
        put("arm_joint_speed_rad_s", qd, LIMITS["arm_qd_max"], qd <= LIMITS["arm_qd_max"],
            "fastest arm joint between consecutive 1/60 s targets (no jumps between IK branches)")
    jg = max(r["joint_gap_mm"] for r in log)
    put("max_joint_gap_mm", jg, "info", True, "largest opening between consecutive capsule ends (hard AL stretch slots)")
    # settle
    put("settle_vmax", settle["vmax"], LIMITS["settle_vmax"], settle["vmax"] <= LIMITS["settle_vmax"],
        f"fastest centreline node between the last two frames of the drop+settle (t = {settle['t']:.2f} s)")
    put("settle_ke", settle["ke"], LIMITS["settle_ke"], settle["ke"] <= LIMITS["settle_ke"])
    ret = settle["retention"]
    rmin = min(v["retention"] for v in ret.values())
    put("pile_retention", {k: round(v["retention"], 3) for k, v in ret.items()}, LIMITS["retention"],
        rmin >= LIMITS["retention"], "1 - RMSD(settled, rest) / RMSD(straight, rest), Kabsch-aligned, per cable")
    cmin = min(v["curvature_ratio"] for v in ret.values())
    put("pile_curvature", {k: round(v["curvature_ratio"], 3) for k, v in ret.items()}, LIMITS["curvature_min"],
        cmin >= LIMITS["curvature_min"], "summed |turning angle| settled / rest")
    iso_f = os.path.join(out_dir, "tests", "rest_shape_test.json")
    if os.path.exists(iso_f):
        iso = json.load(open(iso_f))
        cab = {k: v for k, v in iso["cables"].items() if not k.endswith("_straight_rest")}
        ctl = {k: v for k, v in iso["cables"].items() if k.endswith("_straight_rest")}
        put("iso_retention", {k: round(v["retention"], 3) for k, v in cab.items()}, LIMITS["iso_retention"],
            min(v["retention"] for v in cab.values()) >= LIMITS["iso_retention"],
            f"each cable alone, dropped 3 cm onto the floor, after {iso['sim_s']:.0f} s ({iso['sub']}x{iso['iters']})")
        put("iso_curvature", {k: round(v["curvature_ratio"], 3) for k, v in cab.items()}, LIMITS["iso_curvature"],
            all(LIMITS["iso_curvature"][0] <= v["curvature_ratio"] <= LIMITS["iso_curvature"][1] for v in cab.values()))
        put("iso_drift_mm", {k: round(v["drift_last_s_mm"], 3) for k, v in cab.items()}, LIMITS["iso_drift_mm"],
            max(v["drift_last_s_mm"] for v in cab.values()) <= LIMITS["iso_drift_mm"], "RMSD change over the last second")
        put("control_straight_rest_opens", {k: round(v["retention"], 3) for k, v in ctl.items()},
            LIMITS["control_retention_max"], all(v["retention"] < LIMITS["control_retention_max"] for v in ctl.values()),
            "the U-bend started curved but given a straight rest pose springs open: the kept shapes come from the rest "
            "curvature, not from table friction")
    tg = settle["tangle"]
    put("tangle_crossings", tg["crossings"], LIMITS["crossings_min"], tg["crossings"] >= LIMITS["crossings_min"],
        "top-view crossings between different cables in the settled pile")
    put("tangle_interlocked_pairs", len(tg["interlocked_pairs"]), LIMITS["interlocked_min"],
        len(tg["interlocked_pairs"]) >= LIMITS["interlocked_min"], ", ".join(tg["interlocked_pairs"]))
    # robot
    for k, p in enumerate(picks):
        n = p["cable"]
        for j, s in enumerate(p["steps"]):
            if s.get("missed"):
                put(f"sort{k}_{n}_step{j}_missed_grasp", True, "info", True,
                    "the fingers closed on nothing (the cable had moved since planning); detected, backed off, re-planned")
                continue
            if not s.get("ok"):
                put(f"sort{k}_{n}_step{j}_reachable", False, True, False, s.get("why"))
                continue
            if s.get("dropped"):
                put(f"sort{k}_{n}_step{j}_dropped", s.get("dropped_at_t"), "info", True,
                    "the grasped cable was pulled out of the fingers during the carry (it is trapped in the pile; "
                    "6 N per finger x mu 0.89 x 2 pads ~ 10.7 N of friction); detected, re-planned")
                continue
            med = s["hold_force_median"]
            ok = all(abs(f - GRIP) <= LIMITS["grip_force_rel"] * GRIP for f in med) and \
                s["hold_force_max"] <= LIMITS["grip_force_max"] * GRIP
            tag = "plug" if s["kind"] == "plug" else f"feed{j - 1}"
            put(f"sort{k}_{n}_{tag}_grip_force_N", dict(median=[round(f, 2) for f in med],
                                                                min=round(s["hold_force_min"], 2),
                                                                max=round(s["hold_force_max"], 2)),
                dict(target=GRIP, median_tol=LIMITS["grip_force_rel"], max=LIMITS["grip_force_max"] * GRIP), ok,
                "per-finger squeeze from VBD's contact law (avg ke x overlap) on the pad boxes, while carried")
            if s["kind"] == "plug":
                put(f"sort{k}_{n}_plug_slip_mm", s["slip_max_mm"], LIMITS["plug_slip_mm"],
                    s["slip_max_mm"] <= LIMITS["plug_slip_mm"],
                    f"grip point of the plug in the flange frame while carried; rotation {s['slip_max_deg']:.1f} deg")
            else:
                put(f"sort{k}_{n}_{tag}_cable_slip_mm", s["slip_max_mm"], "info", True,
                    "grasped cable node in the flange frame (the cable may slide through the fingers)")
        fr = final["success"]["per_cable"][n]
        put(f"sort{k}_{n}_in_{p['color']}_bin", round(fr["fractions"][p["color"]], 3), LIMITS["in_bin"],
            fr["correct"], "fraction of the cable's centreline nodes inside the cavity of the bin of its colour, at the end")
    bins0 = [r for r in log if "bin_orange" in r][0]
    for col in ("orange", "blue"):
        b0 = np.array(bins0[f"bin_{col}"])
        sh, tl = 0.0, 0.0
        for r in log:
            b = np.array(r[f"bin_{col}"])
            sh = max(sh, 1e3 * float(np.linalg.norm(b[:2] - b0[:2])))
            z = quat_xyzw_to_mat(b[3:7])[:, 2]
            tl = max(tl, math.degrees(math.acos(min(1.0, z[2]))))
        put(f"bin_{col}_shift_mm", sh, LIMITS["bin_shift_mm"], sh <= LIMITS["bin_shift_mm"])
        put(f"bin_{col}_tilt_deg", tl, LIMITS["bin_tilt_deg"], tl <= LIMITS["bin_tilt_deg"])
    # calibration at the run's budget
    key = f"{L['args']['sub']}x{L['args']['iters']}"
    cal = json.load(open(os.path.join(HERE, "calib_bend.json")))[key]
    rr = {k: round(v["EI_ratio"], 4) for k, v in cal.items() if not k.startswith("_")}
    put("bend_calibration_EI_ratio", rr, LIMITS["calib_EI_ratio"],
        all(abs(v - 1) <= LIMITS["calib_EI_ratio"] for v in rr.values()),
        f"clamped cantilever vs heavy elastica at {key}: effective EI / specified EI")
    put("init_plug_clearance_mm", {k: round(v, 1) for k, v in L["plug_clear_mm"].items()}, 0.0,
        min(L["plug_clear_mm"].values()) > 0, "plugs vs other cables, plugs, bins, table at t = 0")
    gaps = [v for k, v in L["init_gaps_mm"].items()]
    put("init_cable_clearance_mm", round(min(gaps), 2), 0.0, min(gaps) > 0, "cable-cable and cable self gaps at t = 0")
    tm = L["timing"]
    put("timing", {k: dict(sim_s=round(v["sim_s"], 2), wall_s=round(v["wall_s"], 1),
                           wall_per_sim_s=round(v.get("wall_per_sim_s", v["wall_s"] / max(v["sim_s"], 1e-9)), 2))
                   for k, v in tm.items() if isinstance(v, dict)}, "info", True,
        f"60 fps x {L['args']['sub']} substeps x {L['args']['iters']} iterations; GPU {L['gpu']}")
    allp = all(v["pass"] for v in res.values())
    out = dict(all_pass=allp, checks=res)
    json.dump(out, open(os.path.join(out_dir, "checks.json"), "w"), indent=1, default=float)
    for k, v in res.items():
        if k == "timing":
            continue
        print(f"{'PASS' if v['pass'] else 'FAIL'}  {k}: {v['value']}  (limit {v['limit']})")
    print("ALL PASS" if allp else "SOME CHECKS FAIL")
    return allp


if __name__ == "__main__":
    sys.exit(0 if check(sys.argv[1] if len(sys.argv) > 1 else OUT) else 1)
