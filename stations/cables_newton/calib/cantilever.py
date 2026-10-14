"""Bend calibration of the Newton cables: every station cable as a straight clamped cantilever vs the heavy elastica,
at the substep x iteration budget the station runs.

Each cable is a straight VBD rod with its own radius, segment length and line density (no plug). Its first two
segments are kinematic (the clamp); the overhang is ~2 bending lengths (Lb = (EI / w)^(1/3)), where the tip drop is
most sensitive to EI. Gravity ramps in over 2 s and is held 3 s; the tip drop is averaged over the last 0.5 s and
inverted through the heavy elastica (elastica.py) to the effective EI. Per-joint bend stiffness is
scale * EI / segment; with an unconverged block Gauss-Seidel the effective EI comes out lower than EI, so the scale
that gives EI_eff = EI is found by a secant iteration (starting from 1.0) and written to ../calib_bend.json under
"<sub>x<iters>".

    cd ~/sims/newton && .venv/bin/python ~/projects/yam-arcade/stations/cables_newton/calib/cantilever.py --sub 10 --iters 20
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import elastica  # noqa: E402
import ncable as NC  # noqa: E402
import warp as wp  # noqa: E402
import newton  # noqa: E402
from newton.solvers import SolverVBD  # noqa: E402

G = 9.81


def overhang(c):
    w = c["rho_line"] * G
    Lb = (c["EI"] / w) ** (1.0 / 3.0)
    n = int(round(min(2.0 * Lb, 0.8 * c["length"]) / c["seg"]))
    return n * c["seg"], n, Lb


def run(cables, scales, sub, iters, hold=3.0, ramp=2.0):
    b = newton.ModelBuilder(gravity=-G)
    NC.register(b)
    rods = {}
    for k, c in enumerate(cables):
        L, n_over, _ = overhang(c)
        seg = c["seg"]
        n_clamp = 2
        z0 = 1.0
        y = 0.3 * k
        pts = [(-(n_clamp) * seg + i * seg, y, z0) for i in range(n_clamp + n_over + 1)]
        cab = NC.add_cable(b, c, np.array(pts), bend_scale=scales[c["name"]], plug=False)
        NC.make_kinematic(b, cab["bodies"][:n_clamp])
        rods[c["name"]] = dict(cab=cab, L=L, z0=z0, n_clamp=n_clamp)
    b.color()
    model = b.finalize()
    solver = SolverVBD(model, iterations=iters, rigid_contact_hard=False)
    NC.set_hard_stretch(solver, [j for r in rods.values() for j in r["cab"]["joints"]])
    s0, s1 = model.state(), model.state()
    ctrl = model.control()
    contacts = model.contacts()
    fps = 60
    dt = 1.0 / fps / sub
    n_frames = int((hold + ramp) * fps)
    hist = {k: [] for k in rods}
    assert sub % 2 == 0, "CUDA graph replay needs an even substep count (state ping-pong)"
    st = [s0, s1]

    def simulate():
        for k in range(sub):
            a, b_ = st[k % 2], st[(k + 1) % 2]
            a.clear_forces()
            model.collide(a, contacts)
            solver.step(a, b_, ctrl, contacts, dt)

    model.set_gravity((0.0, 0.0, 0.0))
    simulate()  # first step allocates; capture afterwards
    with wp.ScopedCapture() as cap:
        simulate()
    for f in range(n_frames):
        model.set_gravity((0.0, 0.0, -G * min(1.0, (f + 1) / (ramp * fps))))
        wp.capture_launch(cap.graph)
        if f >= n_frames - 30 and f % 3 == 0:
            bq = s0.body_q.numpy()
            for k, r in rods.items():
                X, gaps = NC.cable_nodes(bq, r["cab"])
                hist[k].append((r["z0"] - X[-1, 2], gaps.max()))
    out = {}
    for k, r in rods.items():
        c = NC.CABLE_BY_NAME[k]
        drop = float(np.mean([h[0] for h in hist[k]]))
        W = c["rho_line"] * G
        target = elastica.tip_drop(c["EI"], W, r["L"])[0]
        try:
            from scipy.optimize import brentq
            EI_eff = math.exp(brentq(lambda lb: elastica.tip_drop(math.exp(lb), W, r["L"])[0] - drop,
                                     math.log(c["EI"] / 8), math.log(c["EI"] * 8)))
        except Exception:
            EI_eff = float("nan")
        out[k] = dict(scale=scales[k], overhang=r["L"], tip_drop=drop, target_drop=target, EI_eff=EI_eff,
                      ratio=EI_eff / c["EI"], max_gap_mm=1e3 * max(h[1] for h in hist[k]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    cables = NC.CABLES
    scales = {c["name"]: 1.0 for c in cables}
    rounds = []
    t0 = time.time()
    prev = None
    for it in range(args.rounds):
        res = run(cables, scales, args.sub, args.iters)
        rounds.append(res)
        print(f"round {it}: " + ", ".join(f"{k} s={v['scale']:.3f} r={v['ratio']:.3f}" for k, v in res.items()), flush=True)
        new = {}
        for k, v in res.items():
            if prev is not None and abs(v["ratio"] - prev[k]["ratio"]) > 1e-3 and abs(v["scale"] - prev[k]["scale"]) > 1e-4:
                # secant on ratio(scale) = 1
                sl = (v["ratio"] - prev[k]["ratio"]) / (v["scale"] - prev[k]["scale"])
                new[k] = float(np.clip(v["scale"] + (1.0 - v["ratio"]) / sl, 0.3, 10.0))
            else:
                new[k] = float(np.clip(v["scale"] / v["ratio"], 0.3, 10.0))
        prev = res
        scales = new
    final = run(cables, scales, args.sub, args.iters)
    print("final: " + ", ".join(f"{k} s={v['scale']:.3f} r={v['ratio']:.3f}" for k, v in final.items()), flush=True)
    key = f"{args.sub}x{args.iters}"
    allc = json.load(open(NC.CALIB_JSON)) if os.path.exists(NC.CALIB_JSON) else {}
    allc[key] = {k: dict(scale=v["scale"], EI_ratio=v["ratio"], overhang=v["overhang"], tip_drop=v["tip_drop"],
                         target_drop=v["target_drop"], max_gap_mm=v["max_gap_mm"],
                         Lb=overhang(NC.CABLE_BY_NAME[k])[2]) for k, v in final.items()}
    allc[key]["_note"] = (f"clamped cantilever vs heavy elastica, 60 fps x {args.sub} substeps x {args.iters} iterations, "
                          f"gravity ramp 2 s + hold 3 s; scale = per-joint bend stiffness / (EI / segment); "
                          f"rounds: {[{k: round(v['ratio'], 4) for k, v in r.items()} for r in rounds]}")
    json.dump(allc, open(NC.CALIB_JSON, "w"), indent=1)
    print(f"wrote {NC.CALIB_JSON} [{key}] in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
