"""Initial placement of the cables in the air (CPU).

    python tests/plan_pile.py              # report + plot for the current station.PLACEMENT
    python tests/plan_pile.py --optimize   # random search, writes stations/cables/placement.json

Hard constraints (what libuipc and the scene need at t = 0): every cable-cable surface gap >= GAP_MIN, every cable
>= 15 mm above the table, highest point <= Z_TOP above it, footprint inside the pile area (centre of the table, clear
of the bins), 3 cm clear of every robot link at the home pose. Objective: pairs of cables that cross each other both
over AND under (interlocked) at t = 0 -- the drop preserves that topology because IPC never lets them pass through
each other -- then woven cables (over one, under another), then top-view crossings. The micro-USB lead stays threaded
through the aux lead's loop ('thread' placement).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import station as S  # noqa: E402

GAP_MIN = 0.006
# The USB-C lead is dropped last, flat, from the top, plug toward the blue tray: the one cable that lands ON the pile
# (crossing over others, not woven in), so a pick-and-sort demo has a cable it can lift out. Everything else stays below.
FIXED = {"usbc_wave": dict(xy=[0.0, -0.08], yaw=30.0, dz=0.30)}
Z_TOP_OTHERS = 0.285
Z_TOP = 0.34
AREA = dict(x=(-0.23, 0.23), y=(-0.24, 0.16))
ROBOT = json.load(open(os.path.join(S.ASSETS, "robot_home_aabbs.json")))
KEEP = 0.03


def violation(polys, lay=None):
    """Sum of constraint deficits (m); 0 = feasible."""
    gaps, table = S.init_clearances(polys, lay)
    allp = np.vstack(list(polys.values()) + ([S.plug_points(n, v) for n, v in lay.items() if v["plug_R"] is not None]
                                             if lay is not None else []))
    zt = allp[:, 2].max() - S.ZT
    v = sum(max(0.0, GAP_MIN - g) for g in gaps.values()) + sum(max(0.0, 0.015 - t) for t in table.values())
    v += max(0.0, zt - Z_TOP)
    for n, P in polys.items():
        if n not in FIXED:
            v += max(0.0, P[:, 2].max() - S.ZT - Z_TOP_OTHERS)
    v += max(0.0, AREA["x"][0] - allp[:, 0].min()) + max(0.0, allp[:, 0].max() - AREA["x"][1])
    v += max(0.0, AREA["y"][0] - allp[:, 1].min()) + max(0.0, allp[:, 1].max() - AREA["y"][1])
    for lo, hi in ROBOT.values():
        d = np.linalg.norm(np.maximum(0, np.maximum(np.array(lo) - allp, allp - np.array(hi))), axis=1)
        v += max(0.0, KEEP - d.min())
    return v, min(gaps.values())


def feasible(polys, lay=None):
    v, g = violation(polys, lay)
    return v == 0.0, g


def score(polys):
    tm = S.tangle_metric(polys)
    return 10 * len(tm["interlocked_pairs"]) + 2 * len(tm["middle"]) + 0.05 * tm["crossings"], tm


def perturb(pl, name, rng):
    p = json.loads(json.dumps(pl[name]))
    if name in FIXED:
        return p
    if "thread" in p:
        p["yaw_extra"] = float(np.clip(p.get("yaw_extra", 0.0) + rng.normal(0, 10), -40, 40))
        return p
    k = rng.integers(5)
    if k == 0:
        p["xy"] = (np.array(p["xy"]) + rng.normal(0, 0.03, 2)).tolist()
    elif k == 1:
        p["yaw"] = float(p["yaw"] + rng.normal(0, 30))
    elif k == 2:
        p["dz"] = float(np.clip(p["dz"] + rng.normal(0, 0.02), 0.015, Z_TOP))
    elif k == 3 and name != "aux_loop":
        ax, ang = p.get("tilt") or ([1.0, 0.0, 0.0], 0.0)
        phi = np.arctan2(ax[1], ax[0]) + rng.normal(0, 0.5)
        ang = float(np.clip(ang + rng.normal(0, 10), -35, 35))
        p["tilt"] = [[float(np.cos(phi)), float(np.sin(phi)), 0.0], ang]
    else:
        p["flip"] = not p.get("flip", False)
    return p


def restack(pl):
    """Lowest first, raise each (non-threaded) cable in 5 mm steps until it clears everything placed below it."""
    order = sorted([n for n in pl if "thread" not in pl[n]], key=lambda n: pl[n]["dz"])
    placed = {}
    for n in order:
        c = S.CABLE_BY_NAME[n]
        pl[n]["dz"] = max(0.015, pl[n]["dz"] - 0.03)
        while True:
            lay_n = S.place_one(c, pl[n])
            P = lay_n["P"]
            X = S.plug_points(n, lay_n)
            dmin = min((min(S.seg_seg_dist(P, Q).min() - c["radius"] - S.CABLE_BY_NAME[m]["radius"],
                            S.point_seg_dist(X, Q).min() - S.CABLE_BY_NAME[m]["radius"])
                        for m, Q in placed.items()), default=1.0)
            if dmin >= GAP_MIN + 0.002:
                break
            pl[n]["dz"] += 0.005
        placed[n] = P
    return pl


def optimize(iters, seed):
    rng = np.random.default_rng(seed)
    pl = json.loads(json.dumps(S.PLACEMENT))
    pl["micro_scurve"] = dict(thread="aux_loop", yaw_extra=0.0)
    pl["aux_loop"]["tilt"] = [[1.0, 0.0, 0.0], 90.0]
    for n, f in FIXED.items():
        pl[n] = dict(f)
    lay = S.initial_layout(pl)
    if violation({n: v["P"] for n, v in lay.items()}, lay)[0] > 0:
        pl = restack(pl)
        lay = S.initial_layout(pl)
    polys = {n: v["P"] for n, v in lay.items()}
    v0, _ = violation(polys, lay)
    best = score(polys)[0] if v0 == 0 else -1e4 * v0
    print("start violation", round(v0, 4), "score", best)
    names = [c["name"] for c in S.CABLES]
    for it in range(iters):
        name = names[rng.integers(len(names))]
        cand = dict(pl)
        cand[name] = perturb(pl, name, rng)
        try:
            lay = S.initial_layout(cand)
        except Exception:
            continue
        polys = {n: v["P"] for n, v in lay.items()}
        vio, g = violation(polys, lay)
        sc = score(polys)[0] if vio == 0 else -1e4 * vio
        if sc >= best:
            if sc > best:
                print(f"it {it}: score {sc:.2f} (min gap {1e3 * g:.1f} mm)", flush=True)
            best, pl = sc, cand
    return pl


def report(pl, out_png):
    lay = S.initial_layout(pl)
    polys = {n: v["P"] for n, v in lay.items()}
    ok, g = feasible(polys, lay)
    sc, tm = score(polys)
    gaps, table = S.init_clearances(polys)
    print("feasible", ok, "min gap mm", round(1e3 * g, 1), "min table gap mm", round(1e3 * min(table.values()), 1),
          "top above table", round(max(P[:, 2].max() for P in polys.values()) - S.ZT, 3))
    print("score", round(sc, 2), "crossings", tm["crossings"], "interlocked", tm["interlocked_pairs"], "woven",
          tm["middle"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for n, P in polys.items():
        col = S.COLORS[S.CABLE_BY_NAME[n]["color"]]
        lw = 1400 * S.CABLE_BY_NAME[n]["radius"]
        ax[0].plot(P[:, 0], P[:, 1], color=col, lw=lw)
        ax[0].text(P[0, 0], P[0, 1], n, fontsize=7)
        ax[1].plot(P[:, 0], P[:, 2] - S.ZT, color=col, lw=lw)
    for n, v in lay.items():
        if v["plug_R"] is not None:
            X = S.plug_points(n, v)
            ax[0].fill(X[:, 0], X[:, 1], color="0.2")
            ax[1].fill(X[:, 0], X[:, 2] - S.ZT, color="0.2")
    for col, b in S.BINS.items():
        sx, sy = S.BIN["size"][:2]
        ax[0].add_patch(plt.Rectangle((b["xy"][0] - sx / 2, b["xy"][1] - sy / 2), sx, sy, fill=False,
                                      color=S.COLORS[col]))
    for lo, hi in ROBOT.values():
        ax[0].add_patch(plt.Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], fill=False, color="0.6", lw=0.5))
    ax[0].add_patch(plt.Rectangle((-0.6, -0.35), 1.2, 0.7, fill=False, color="0.4"))
    ax[0].set_xlim(-0.65, 0.65)
    ax[0].set_ylim(-0.45, 0.4)
    ax[0].set_aspect("equal")
    ax[1].set_aspect("equal")
    ax[0].set_title("top")
    ax[1].set_title("side (x-z), height above table")
    plt.tight_layout()
    plt.savefig(out_png, dpi=90)
    print(out_png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pl = S.PLACEMENT
    if a.optimize:
        pl = optimize(a.iters, a.seed)
        lay = S.initial_layout(pl)
        if feasible({n: v["P"] for n, v in lay.items()}, lay)[0]:
            json.dump(pl, open(S.PLACEMENT_JSON, "w"), indent=1)
            print("wrote", S.PLACEMENT_JSON)
        else:
            print("NOT FEASIBLE, placement.json left unchanged")
    report(pl, os.path.join(S.ROOT, "out", "cables", "tests", "plan_pile.png"))
