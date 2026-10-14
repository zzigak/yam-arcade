"""Does a pre-curved Newton rod keep its rest shape? Each station cable (with its plug) is dropped 3 cm onto the floor
on its own (0.8 m apart, nothing else to touch); 4 s. Per cable: retention (1 - RMSD(now, rest) / RMSD(straight,
rest)), summed-curvature ratio, and RMSD drift over the last second (creep). Control: the same U-bend built as a
straight rod and bent into the U by placing its bodies -- no, VBD's rest pose IS model.body_q -- so the control is a
U-bend whose joints have zero bend stiffness (it must collapse), showing that the retained shape comes from the
bend springs and not from friction.

    cd ~/sims/newton && .venv/bin/python ~/projects/yam-arcade/stations/cables_newton/tests/rest_shape_test.py \
        [--sub 10 --iters 10 --scale 1.0]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ncable as NC  # noqa: E402
import warp as wp  # noqa: E402
import newton  # noqa: E402
from newton.solvers import SolverVBD  # noqa: E402

OUT = os.path.join(NC.ROOT, "out", "cables", "newton", "tests")


def place_flat(P0, x, lift):
    P = np.asarray(P0, float).copy()
    P[:, :2] += np.array([x, 0.0]) - 0.5 * (P[:, :2].min(0) + P[:, :2].max(0))
    P[:, 2] += lift - P[:, 2].min()
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=40)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--scale", type=float, default=None, help="bend scale (default: calibrated, else 1.0)")
    ap.add_argument("--time", type=float, default=4.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cal = NC.bend_scales().get(f"{args.sub}x{args.iters}", {})

    b = newton.ModelBuilder(gravity=-9.81)
    NC.register(b)
    b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=NC.TABLE_KE, kd=0.0, mu=NC.TABLE_MU, margin=0.0,
                                                           gap=NC.CONTACT_GAP))
    runs = [(c, False) for c in NC.CABLES] + [(NC.CABLE_BY_NAME["hdmi_ubend"], True)]
    straight = {}
    cabs = {}
    for k, (c, ctrl) in enumerate(runs):
        sc = args.scale if args.scale is not None else cal.get(c["name"], {}).get("scale", 1.0)
        lo, hi = NC.plug_extent(c["name"])
        P = place_flat(np.array(c["rest_local"]), -3.2 + 0.8 * k, c["radius"] + 0.03 + max(0.0, -lo[2] - c["radius"]))
        name = c["name"] + ("_straight_rest" if ctrl else "")
        cabs[name] = NC.add_cable(b, c, P, bend_scale=sc, plug=not ctrl)
        cabs[name]["scale"] = sc
        if ctrl:  # same segments laid out straight: the rest pose the control gets
            seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
            S = np.c_[np.concatenate([[0.0], np.cumsum(seg)]), np.zeros((len(P), 1)), np.zeros((len(P), 1))] + P[0]
            b2 = newton.ModelBuilder()
            NC.register(b2)
            straight[name] = (cabs[name]["bodies"], NC.add_cable(b2, c, S, bend_scale=sc, plug=False)["bodies"], b2)
    b.color()
    model = b.finalize()
    s0, s1 = model.state(), model.state()  # start from the drawn (curved) poses
    bq_rest = model.body_q.numpy()
    for name, (bods, bods2, b2) in straight.items():
        for i, i2 in zip(bods, bods2):
            bq_rest[i] = np.array(b2.body_q[i2], float)
    model.body_q.assign(bq_rest)  # rest pose for the solver
    solver = SolverVBD(model, iterations=args.iters, rigid_contact_hard=False, rigid_body_contact_buffer_size=512,
                       friction_epsilon=1e-3)
    NC.set_hard_stretch(solver, [j for c in cabs.values() for j in c["joints"]])
    ctrl = model.control()
    contacts = model.contacts()
    fps = 60
    dt = 1.0 / fps / args.sub

    def simulate():
        nonlocal s0, s1
        for _ in range(args.sub):
            s0.clear_forces()
            model.collide(s0, contacts)
            solver.step(s0, s1, ctrl, contacts, dt)
            s0, s1 = s1, s0

    simulate()  # warm-up (allocations) before capture
    with wp.ScopedCapture() as cap:
        simulate()
    graph = cap.graph
    n = int(round(args.time * fps))
    hist = {k: [] for k in cabs}
    stretch = [np.inf, 0.0]
    t0 = time.time()
    for f in range(2, n + 1):
        wp.capture_launch(graph)
        if f % 6 == 0 or f == n:
            bq = s0.body_q.numpy()
            for k, c in cabs.items():
                X, gaps = NC.cable_nodes(bq, c)
                ratio = np.linalg.norm(np.diff(X, axis=0), axis=1) / c["seg"]
                stretch = [min(stretch[0], ratio.min()), max(stretch[1], ratio.max())]
                m = NC.retention(X, c["P0"])
                m.update(t=f / fps, zmin=float(X[:, 2].min() - c["spec"]["radius"]), gap_mm=1e3 * float(gaps.max()))
                hist[k].append(m)
    wp.synchronize()
    wall = time.time() - t0
    bq = s0.body_q.numpy()
    res = {}
    for k, c in cabs.items():
        h = hist[k]
        last = h[-1]
        one_s = [m for m in h if m["t"] >= h[-1]["t"] - 1.0 - 1e-9]
        X, _ = NC.cable_nodes(bq, c)
        res[k] = dict(scale=c["scale"], retention=last["retention"], curvature_ratio=last["curvature_ratio"],
                      rmsd_mm=last["rmsd_mm"], drift_last_s_mm=abs(one_s[-1]["rmsd_mm"] - one_s[0]["rmsd_mm"]),
                      zmin_mm=1e3 * min(m["zmin"] for m in h), max_joint_gap_mm=max(m["gap_mm"] for m in h),
                      final=X.tolist(), rest=c["P0"].tolist())
        print(f"{k:20s} scale {c['scale']:.2f} retention {last['retention']:.3f} curv {last['curvature_ratio']:.3f} "
              f"rmsd {last['rmsd_mm']:.1f} mm drift(last 1 s) {res[k]['drift_last_s_mm']:.3f} mm "
              f"zmin {res[k]['zmin_mm']:.2f} mm gap {res[k]['max_joint_gap_mm']:.4f} mm")
    print(f"stretch {stretch}, wall {wall:.1f} s for {args.time:.1f} s ({args.sub}x{args.iters})")
    json.dump(dict(sub=args.sub, iters=args.iters, wall_s=wall, sim_s=args.time, stretch=stretch, cables=res,
                   hist={k: [{kk: v for kk, v in m.items()} for m in h] for k, h in hist.items()}),
              open(os.path.join(OUT, f"rest_shape_test{args.tag}.json"), "w"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(res), figsize=(2.6 * len(res), 2.8))
        for ax, (k, r) in zip(axs, res.items()):
            R, F = np.array(r["rest"]), np.array(r["final"])
            R = R - R.mean(0)
            Fc = F - F.mean(0)
            U, _, Vt = np.linalg.svd(Fc.T @ R)
            d = np.sign(np.linalg.det(U @ Vt))
            Fa = Fc @ (U @ np.diag([1, 1, d]) @ Vt)
            ax.plot(R[:, 0], R[:, 1], color="0.7", lw=4, label="rest")
            ax.plot(Fa[:, 0], Fa[:, 1], color="C0", lw=1.5, label="after 4 s")
            ax.set_aspect("equal")
            ax.set_title(f"{k}\nret {r['retention']:.2f} curv {r['curvature_ratio']:.2f}", fontsize=8)
            ax.axis("off")
        axs[0].legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"rest_shape_test{args.tag}.png"), dpi=110)
    except Exception as e:  # plotting is a convenience
        print("plot failed:", e)


if __name__ == "__main__":
    main()
