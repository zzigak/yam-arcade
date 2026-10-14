"""Does each station cable keep its rest shape? Every cable of station.CABLES is dropped 3 cm onto the floor on its
own (0.7 m apart, nothing else to touch), plus two controls built as plain FEM.Rod (libuipc KRB carrying the full EI,
straight rest shape): the HDMI U-bend and the power coil. 4 s; per-cable retention, curvature ratio and RMSD vs the
rest shape over time. Writes out/cables/tests/rest_shape_test.json (read by check.py) and a still.

    stations/cables/gpu_run.sh out/cables/tests/rest_shape_test.log stations/cables/tests/rest_shape_test.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import cable_model as CM  # noqa: E402
import station as S  # noqa: E402
import layout  # noqa: E402

OUT = os.path.join(S.ROOT, "out", "cables", "tests")


def kabsch_rmsd(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return float(np.sqrt(((Xc @ R - Yc) ** 2).sum(1).mean()))


def straight_rmsd(P):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return kabsch_rmsd(np.stack([s, 0 * s, 0 * s], 1), P)


def metrics(X, P):
    th0, th = CM.turning_angles(P), CM.turning_angles(X)
    rm = kabsch_rmsd(X, P)
    return dict(rmsd_mm=1e3 * rm, retention=1 - rm / straight_rmsd(P), curvature_ratio=float(th.sum() / th0.sum()),
                seg=[float(v) for v in (lambda r: (r.min(), r.max()))(
                    np.linalg.norm(np.diff(X, axis=0), axis=1) / np.linalg.norm(np.diff(P, axis=0), axis=1))],
                zgap_mm=None)


def main():
    import genesis as gs
    import genesis.utils.mesh as mu
    import trimesh
    from genesis.utils.misc import tensor_to_array as A
    T = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=S.DT, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(enable_collision=False),
        coupler_options=gs.options.IPCCouplerOptions(contact_d_hat=CM.D_HAT, contact_resistance=1e7,
                                                     newton_semi_implicit_enable=False),
        vis_options=gs.options.VisOptions(show_world_frame=False, background_color=layout.BACKGROUND_RGB,
                                          ambient_light=(0.45, 0.45, 0.45)),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=0.5),
                     surface=gs.surfaces.Default(color=(0.80, 0.70, 0.55)))
    runs = [(c, False) for c in S.CABLES] + [(S.CABLE_BY_NAME["hdmi_ubend"], True),
                                             (S.CABLE_BY_NAME["power_coil"], True)]
    ents = {}
    for k, (c, krb) in enumerate(runs):
        P0, _ = CM.resample(CM.rest_curve(c["shape"]), c["seg"])
        P = CM.place(P0, (-3.5 + 0.7 * k, 0.0), 0.0, c["radius"] + 0.03)
        fr = CM.Framed(P, c["radius"], c["EI"])
        h = CM.add_framed_cable(scene, gs, fr, c["rho_line"], c["friction"], S.COLORS[c["color"]], krb_only=krb)
        ents[c["name"] + ("_KRB_only" if krb else "")] = (fr, h, c)
    cam = scene.add_camera(res=(1600, 500), pos=(0.0, -2.6, 1.6), lookat=(0.0, 0.0, 0.0), fov=60, GUI=False,
                           debug=True)
    scene.build()
    log, t0 = [], time.time()
    n = int(round(T / S.DT))
    for i in range(n + 1):
        if i:
            scene.step()
        if i % 30 == 0 or i == n:
            rec = dict(t=round(i * S.DT, 3))
            for name, (fr, h, c) in ents.items():
                X = A(h["cable"].get_state().pos).reshape(-1, 3).astype(np.float64)
                m = metrics(X, fr.P)
                m["zgap_mm"] = 1e3 * float(X[:, 2].min() - c["radius"])
                rec[name] = m
            log.append(rec)
            if i % 90 == 0:
                print(rec["t"], {k: (round(v["retention"], 3), round(v["curvature_ratio"], 3)) for k, v in rec.items()
                                 if k != "t"}, f"wall {time.time() - t0:.0f}s", flush=True)
    nodes = []
    for name, (fr, h, c) in ents.items():
        X = A(h["cable"].get_state().pos).reshape(-1, 3).astype(np.float64)
        v, f = mu.create_tube(X, c["radius"], 14, False)
        tm = trimesh.Trimesh(v, f, process=False)
        tm.visual.vertex_colors = (np.array([*S.COLORS[c["color"]], 1.0]) * 255).astype(np.uint8)
        nodes.append(scene.draw_debug_mesh(tm))
        v, f = mu.create_tube(fr.P - np.array([0, 0, fr.P[:, 2].min() - 0.0005]), 0.0012, 8, False)
        tm = trimesh.Trimesh(v, f, process=False)
        tm.visual.vertex_colors = np.array([30, 30, 30, 255], np.uint8)
        nodes.append(scene.draw_debug_mesh(tm))
    import imageio.v2 as imageio
    imageio.imwrite(os.path.join(OUT, "rest_shape_test.png"), np.asarray(A(cam.render(rgb=True)[0]), np.uint8))
    final = log[-1]
    early = min(log, key=lambda r: abs(r["t"] - 1.0))
    summary = {k: dict(final=v, drift_mm_1s_to_end=v["rmsd_mm"] - early[k]["rmsd_mm"]) for k, v in final.items()
               if k != "t"}
    shapes = {name: dict(rest=fr.P.tolist(), final=A(h["cable"].get_state().pos).reshape(-1, 3).tolist(),
                         color=c["color"], radius=c["radius"]) for name, (fr, h, c) in ents.items()}
    json.dump(dict(summary=summary, log=log, wall=time.time() - t0, shapes=shapes),
              open(os.path.join(OUT, "rest_shape_test.json"), "w"), indent=1)
    figure(shapes, summary)
    for k, v in summary.items():
        f = v["final"]
        print(f"{k:22s} retention {f['retention']:.3f} curvature {f['curvature_ratio']:.3f} rmsd {f['rmsd_mm']:.2f} mm "
              f"drift {v['drift_mm_1s_to_end']:+.2f} mm seg [{f['seg'][0]:.4f}, {f['seg'][1]:.4f}] zgap {f['zgap_mm']:.2f}")


def figure(shapes, summary, out=os.path.join(OUT, "rest_shape_test_panels.png")):
    """Top views: rest shape (thin black, rigidly aligned) vs the cable after 4 s on the floor (colour)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(shapes)
    fig, axs = plt.subplots(2, 5, figsize=(15, 6.4))
    for ax, name in zip(axs.ravel(), names):
        d = shapes[name]
        P, X = np.array(d["rest"]), np.array(d["final"])
        Xc, Pc = X - X.mean(0), P - P.mean(0)
        U, _, Vt = np.linalg.svd(Xc.T @ Pc)
        R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        Xa = Xc @ R
        ax.plot(Xa[:, 0], Xa[:, 1], color=S.COLORS[d["color"]], lw=max(1.0, 1400 * d["radius"]), solid_capstyle="round")
        ax.plot(Pc[:, 0], Pc[:, 1], color="k", lw=0.8)
        f = summary[name]["final"]
        ax.set_title(f"{name}\nretention {f['retention']:.2f}, curvature x{f['curvature_ratio']:.2f}", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("each cable dropped alone, 4 s: black = rest shape, colour = cable (best rigid fit); "
                 "last two = plain FEM.Rod controls", fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=100)


if __name__ == "__main__":
    main()
