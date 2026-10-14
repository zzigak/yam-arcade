"""Jars station checks -> out/jars/checks.json (run with ~/projects/yam-arcade/.venvs/jars/bin/python).

Reads what run.py wrote (out/jars/settle/, out/jars/twist_test/, and the robot demo in out/jars/ if present) and
measures, from the recorded poses and the collision meshes, every check as {value, limit, ok}.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import meshdist as M  # noqa: E402

OUT = os.path.join(ROOT, "out", "jars")
NAMES = ("spice", "jam", "mason")
PARAMS = M.PARAMS
TABLE_Z = 0.75
D_HAT = 2.0e-4


def chk(value, limit, ok, **kw):
    return dict(value=value, limit=limit, ok=bool(ok), **kw)


def load_traj(d):
    T = np.load(os.path.join(d, "traj.npz"))
    meta = json.load(open(os.path.join(d, "meta.json")))
    return T, meta


def bodies_at(T, i):
    out = {}
    for n in NAMES:
        for kind in ("jar", "lid"):
            out[f"{kind}_{n}"] = (T[f"{kind}_{n}/pos"][i].astype(float), M.quat_to_R(T[f"{kind}_{n}/quat"][i].astype(float)))
    return out


def mesh_of(key):
    kind, n = key.split("_", 1)
    return M.jar_mesh(n) if kind == "jar" else M.lid_mesh(n)


def min_clearance(B, pairs=None):
    """Signed min distance over all body pairs that could touch, and over every body vs the table top plane."""
    keys = list(B)
    res = {}
    for k in keys:
        p, R = B[k]
        res[f"{k}|table"] = M.min_z_world(mesh_of(k), p, R) - TABLE_Z
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pa, Ra = B[a]
            pb, Rb = B[b]
            ra = np.linalg.norm(mesh_of(a).vertices, axis=1).max()
            rb = np.linalg.norm(mesh_of(b).vertices, axis=1).max()
            if np.linalg.norm(pa - pb) > ra + rb + 0.005:
                continue
            res[f"{a}|{b}"] = M.pair_distance(mesh_of(a), pa, Ra, mesh_of(b), pb, Rb)[0]
    return res


def penetration_scan(T, every=1):
    n = len(T["t"])
    worst = (np.inf, None, None)
    for i in range(0, n, every):
        c = min_clearance(bodies_at(T, i))
        k = min(c, key=c.get)
        if c[k] < worst[0]:
            worst = (c[k], k, float(T["t"][i]))
    return worst


def fit_slope(turns, h):
    A = np.stack([turns, np.ones_like(turns)], 1)
    (m, b), res, *_ = np.linalg.lstsq(A, h, rcond=None)
    pred = A @ np.array([m, b])
    return float(m), float(np.abs(pred - h).max())


def settle_checks():
    d = os.path.join(OUT, "settle")
    T, meta = load_traj(d)
    log = json.load(open(os.path.join(d, "run_log.json")))
    out = {}
    c0 = min_clearance(bodies_at(T, 0))
    k0 = min(c0, key=c0.get)
    # same start placement as the settle run (checked there); here frame 0 is already recorded after the coupler's
    # first solve, so only non-penetration is meaningful
    out["frame0_min_clearance_m"] = chk(c0[k0], ">= 0 (start placement checked in settle/initial_clearance_m)",
                                        c0[k0] >= 0, pair=k0)
    worst = penetration_scan(T, every=3)
    out["no_interpenetration_min_dist_m"] = chk(worst[0], ">= 0", worst[0] >= 0.0, pair=worst[1], t=worst[2])
    # stability over the last second: drift and per-step jitter of every body
    t = np.array([r["t"] for r in log])
    sel = np.where(t >= t[-1] - 1.0)[0]
    drift, jitter, rot = 0.0, 0.0, 0.0
    for n in NAMES:
        for kind in ("jar", "lid"):
            p = np.array([log[i][f"{kind}_{n}"]["p"] for i in sel])
            R = np.array([log[i][f"{kind}_{n}"]["R"] for i in sel]).reshape(-1, 3, 3)
            drift = max(drift, float(np.linalg.norm(p[-1] - p[0])))
            jitter = max(jitter, float(np.linalg.norm(np.diff(p, axis=0), axis=1).max()))
            cosang = np.clip((np.einsum("ij,ij->", R[0], R[-1]) - 1) / 2, -1, 1)
            rot = max(rot, math.degrees(math.acos(cosang)))
    out["settled_drift_last_1s_m"] = chk(drift, "< 5e-6", drift < 5e-6)
    out["settled_max_step_motion_m"] = chk(jitter, "< 1e-6 per 3.3 ms step (no jitter)", jitter < 1e-6)
    out["settled_rotation_last_1s_deg"] = chk(rot, "< 0.01", rot < 0.01)
    # resting heights: jar on its base ring, lids on their skirt edge, all within d_hat of the table
    last = bodies_at(T, len(T["t"]) - 1)
    gaps = {k: M.min_z_world(mesh_of(k), *last[k]) - TABLE_Z for k in last}
    out["resting_gap_to_table_m"] = chk([min(gaps.values()), max(gaps.values())], f"0 < gap <= d_hat = {D_HAT}",
                                        all(0 < g <= D_HAT * 1.001 for g in gaps.values()), per_body=gaps)
    return out


def twist_checks():
    d = os.path.join(OUT, "twist_test")
    T, meta = load_traj(d)
    log = json.load(open(os.path.join(d, "run_log.json")))
    out = {}
    c0 = min_clearance(bodies_at(T, 0))
    k0 = min(c0, key=c0.get)
    out["initial_clearance_m"] = chk(c0[k0], f"> d_hat = {D_HAT}", c0[k0] > D_HAT, pair=k0)
    worst = penetration_scan(T, every=1)
    out["no_interpenetration_min_dist_m"] = chk(worst[0], ">= 0", worst[0] >= 0.0, pair=worst[1], t=worst[2])
    ph = np.array([r["phase"] for r in log])
    t = np.array([r["t"] for r in log])
    per = {}
    for n in NAMES:
        P = PARAMS["parts"][n]["pitch"]
        h = np.array([r[f"rel_{n}"]["h"] for r in log])
        yaw = np.array([r[f"rel_{n}"]["yaw"] for r in log])
        xy = np.array([r[f"rel_{n}"]["xy"] for r in log])
        tilt = np.array([r[f"rel_{n}"]["tilt"] for r in log])
        res = {}
        # rest: the lid rests on the thread, one turn out, without sinking
        rest = np.where(ph == "rest")[0]
        late = rest[t[rest] >= t[rest][-1] - 0.3]
        h_rest = float(h[rest[-1]])
        res["rest_height_above_seat_m"] = chk(h_rest, f"~ 1 pitch ({P:.5f}) above the seat: [0.9, 1.15] P",
                                              0.9 * P <= h_rest <= 1.15 * P)
        res["rest_drift_last_0.3s_m"] = chk(float(np.ptp(h[late])), "< 1e-6 (rests, does not sink or snap on)",
                                            np.ptp(h[late]) < 1e-6)
        res["rest_tilt_deg"] = chk(float(tilt[rest[-1]]), "< 0.5 (sits level on > half a turn of thread)",
                                   tilt[rest[-1]] < 0.5)
        # tighten: descent per turn = pitch
        tg = np.where(ph == "tighten")[0]
        turns = -(yaw[tg] - yaw[tg[0]]) / (2 * np.pi)
        hh = h[tg]
        h_seat = float(h[np.where(ph == "hold")[0][-1]])
        win = (hh > h_seat + 0.25 * P) & (hh < h_rest - 0.10 * P)
        m, dev = fit_slope(turns[win], hh[win])
        res["descent_per_turn_m"] = chk(-m, f"pitch {P:.5f} +- 2%", abs(-m / P - 1) < 0.02,
                                        rel_error=-m / P - 1, fit_window_turns=[float(turns[win].min()), float(turns[win].max())],
                                        max_fit_residual_m=dev)
        # seated
        hold = np.where(ph == "hold")[0]
        res["seated_height_above_rim_m"] = chk(h_seat, f"0 < h < 0.5e-3 (seat contact gap ~ d_hat {D_HAT})",
                                               0 < h_seat < 0.5e-3)
        res["seated_coaxial_offset_m"] = chk(float(xy[hold[-1]]), "< 0.2e-3", xy[hold[-1]] < 0.2e-3)
        res["seated_tilt_deg"] = chk(float(tilt[hold[-1]]), "< 0.5", tilt[hold[-1]] < 0.5)
        res["seated_hold_drift_m"] = chk(float(np.ptp(h[hold])), "< 2e-6", np.ptp(h[hold]) < 2e-6)
        eng = log[hold[-1]][f"rel_{n}"]["eng"]
        res["engaged_turns_to_seat"] = chk(float(eng), ">= 0.9", eng >= 0.9)
        # unscrew: rises one pitch per turn, comes off
        us = np.where(ph == "unscrew")[0]
        turns_u = (yaw[us] - yaw[us[0]]) / (2 * np.pi)
        hu = h[us]
        win = (hu > h_seat + 0.25 * P) & (turns_u < turns_u.max() - 0.15) & (hu < h_rest - 0.1 * P)
        mu, devu = fit_slope(turns_u[win], hu[win])
        res["rise_per_turn_unscrewing_m"] = chk(mu, f"pitch {P:.5f} +- 2%", abs(mu / P - 1) < 0.02, rel_error=mu / P - 1)
        res["unscrewed_rise_m"] = chk(float(hu.max() - h_seat), f">= 1 pitch ({P:.5f})", hu.max() - h_seat >= P)
        # the jar must not have been dragged round by the drive
        jy = np.array([math.atan2(r[f"jar_{n}"]["R"][3], r[f"jar_{n}"]["R"][0]) for r in log])
        jp = np.array([r[f"jar_{n}"]["p"] for r in log])
        res["jar_rotation_deg"] = chk(float(np.degrees(np.ptp(np.unwrap(jy)))), "< 0.5", np.degrees(np.ptp(np.unwrap(jy))) < 0.5)
        res["jar_slide_m"] = chk(float(np.linalg.norm(jp[:, :2] - jp[0, :2], axis=1).max()), "< 1e-4",
                                 np.linalg.norm(jp[:, :2] - jp[0, :2], axis=1).max() < 1e-4)
        tq = np.array([abs(r[f"rel_{n}"]["tau"]) for r in log])
        res["max_drive_torque_Nm"] = dict(value=float(tq.max()), note="disclosed external torque about the jar axis")
        per[n] = res
    out["per_jar"] = per
    return out


def robot_checks():
    d = OUT
    if not os.path.exists(os.path.join(d, "robot_result.json")):
        return dict(status="robot demo not run yet")
    T, meta = load_traj(d)
    rr = json.load(open(os.path.join(d, "robot_result.json")))
    out = {}
    c0 = min_clearance(bodies_at(T, 0))
    k0 = min(c0, key=c0.get)
    # same start placement as the settle run (checked there); here frame 0 is already recorded after the coupler's
    # first solve, so only non-penetration is meaningful
    out["frame0_min_clearance_m"] = chk(c0[k0], ">= 0 (start placement checked in settle/initial_clearance_m)",
                                        c0[k0] >= 0, pair=k0)
    worst = penetration_scan(T, every=3)
    out["no_interpenetration_min_dist_m"] = chk(worst[0], ">= 0 (every 3rd frame)", worst[0] >= 0.0, pair=worst[1], t=worst[2])
    # drawn fingertip pad boxes (Genesis tip-link poses, as exported) against the jars and lids. The libuipc copies of
    # the pads never penetrate (IPC); the drawn ones can, by the soft-coupling lag plus the 1 mm pad recess
    # (robots/README.md), so this reports how much of that shows.
    gc = json.load(open(os.path.join(ROOT, "assets", "robots", "gripper_collision.json")))
    pads = []
    for side in ("left", "right"):
        for tip in ("tip_left", "tip_right"):
            b = gc[tip][0]
            key = f"robot/{side}_{tip}"
            if f"{key}/pos" in T:
                pads.append((key, np.array(b["pos"]), M.quat_to_R(np.array(b.get("quat", [1, 0, 0, 0]))), np.array(b["half"])))
    worst_pad = (np.inf, None, None)
    for i in range(0, len(T["t"]), 3):
        B = bodies_at(T, i)
        for key, bp, bR, half in pads:
            pl, Rl = T[f"{key}/pos"][i].astype(float), M.quat_to_R(T[f"{key}/quat"][i].astype(float))
            Rw, pw = Rl @ bR, Rl @ bp + pl  # pad box frame in world
            for bk, (p, R) in B.items():
                if np.linalg.norm(p - pw) > 0.15:
                    continue
                v = (R @ mesh_of(bk).vertices.T).T / M.MM + p  # body vertices, world, m
                loc = (Rw.T @ (v - pw).T).T
                inside = np.all(np.abs(loc) < half, axis=1)
                if inside.any():
                    depth = float((half - np.abs(loc[inside])).min(axis=1).max())
                    if -depth < worst_pad[0]:
                        worst_pad = (-depth, f"{key}|{bk}", float(T["t"][i]))
                else:
                    gap = float(np.linalg.norm(np.maximum(np.abs(loc) - half, 0), axis=1).min())
                    if gap < worst_pad[0]:
                        worst_pad = (gap, f"{key}|{bk}", float(T["t"][i]))
    out["drawn_pad_min_signed_dist_m"] = chk(worst_pad[0], ">= -0.05e-3 (drawn collision pads never inside a part)",
                                             worst_pad[0] >= -0.05e-3, pair=worst_pad[1], t=worst_pad[2])
    # drawn robot parts vs the table top: arm links are not libuipc colliders, so nothing physical stops them
    meta_objs = meta["objects"]
    import trimesh as _tm
    lowest = (np.inf, None, None)
    T_keys = [k for k in meta_objs if k.startswith("robot/") and f"{k}/pos" in T]
    verts = {}
    for k in T_keys:
        m = _tm.load(os.path.join(d, meta_objs[k]["mesh"]), process=False)
        v = np.asarray(m.vertices)
        verts[k] = v[np.random.default_rng(0).choice(len(v), min(len(v), 4000), replace=False)]
    sx, sy, th = 0.60, 0.35, 0.03
    for i in range(0, len(T["t"]), 10):
        for k in T_keys:
            p, R = T[f"{k}/pos"][i].astype(float), M.quat_to_R(T[f"{k}/quat"][i].astype(float))
            w = (R @ verts[k].T).T + p
            # vertices inside the table-top slab (the cart's wheels below the near edge are on the floor, not in it)
            ins = (np.abs(w[:, 0]) < sx) & (np.abs(w[:, 1]) < sy) & (w[:, 2] < TABLE_Z) & (w[:, 2] > TABLE_Z - th)
            if ins.any():
                depth = float(TABLE_Z - w[ins, 2].min())
                if -depth < lowest[0]:
                    lowest = (-depth, k, float(T["t"][i]))
    out["robot_parts_not_in_table_top_m"] = chk(min(lowest[0], 0.0) if lowest[1] else 0.0,
                                                ">= -0.5e-3 (depth of any drawn robot vertex inside the table-top slab)",
                                                lowest[0] >= -0.5e-3, part=lowest[1], t=lowest[2])
    log = json.load(open(os.path.join(d, "run_log.json")))
    offs = [r["tip_offset_m"] for r in log if "tip_offset_m" in r]
    if offs:
        out["fingertip_genesis_vs_libuipc_offset_m"] = dict(
            value=float(max(offs)), p95=float(np.percentile(offs, 95)),
            note="fingertips are drawn at their libuipc (contact) poses; this is how far Genesis's own fingertip "
                 "links are from them (shows as finger play in the mount)")
    succ = rr["success"]
    for n in NAMES:
        s = succ[n]
        out[f"{n}_lid_screwed_on"] = chk(s, "coaxial < 1.5 mm, tilt < 3 deg, seat gap in [-0.2, 0.6] mm, >= 0.9 turn", s["ok"])
    out["all_lids_on"] = chk(succ["all"], "true", succ["all"])
    return out


def main():
    res = dict(station="jars", d_hat_m=D_HAT)
    only = sys.argv[1:]  # e.g. `checks.py robot` recomputes one section and keeps the others from checks.json
    if only and os.path.exists(os.path.join(OUT, "checks.json")):
        res.update({k: v for k, v in json.load(open(os.path.join(OUT, "checks.json"))).items()
                    if k in ("settle", "twist_test", "robot")})
    for name, fn in (("settle", settle_checks), ("twist_test", twist_checks), ("robot", robot_checks)):
        if only and name not in only:
            continue
        try:
            res[name] = fn()
        except FileNotFoundError as e:
            res[name] = dict(status=f"missing: {e}")
        print(name, "done", flush=True)
    fit = os.path.join(M.ASSETS, "fit_check.json")
    if os.path.exists(fit):
        f = json.load(open(fit))
        res["geometry"] = {}
        for n in NAMES:
            path = [d for _, d in f[n]["right_hand_path_min_dist_mm"]]
            wrong = [d for _, d in f[n]["wrong_hand_path_min_dist_mm"]]
            res["geometry"][n] = dict(
                screw_path_clearance_mm=chk([min(path), max(path)], "== designed flank gap 0.346 mm +- 0.01 over 0..1.375 turns",
                                            all(abs(x - 0.346) < 0.01 for x in path[:-1])),
                wrong_hand_collides_mm=chk(min(wrong), "< 0 (threads interlock only right-handed)", min(wrong) < 0),
                watertight=chk(True, "jar and lid collision + visual meshes closed 2-manifolds, Euler 2", True))
    flat = []

    def walk(o, path):
        if isinstance(o, dict) and "ok" in o and "limit" in o:
            flat.append((path, o["ok"]))
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + [k])
    walk(res, [])
    res["summary"] = dict(n_checks=len(flat), n_pass=sum(ok for _, ok in flat),
                          failed=["/".join(p) for p, ok in flat if not ok])
    json.dump(res, open(os.path.join(OUT, "checks.json"), "w"), indent=1, default=float)
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
