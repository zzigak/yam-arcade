"""Jars station runs (Genesis venv, gs.gpu).

    PY=~/simquarry/genesis-world/.venv/bin/python
    $PY run.py settle   # puzzle layout (jars + loose lids) settles: stability       -> out/jars/settle/
    $PY run.py twist    # disclosed torque drive: rest, screw on, seat, unscrew       -> out/jars/twist_test/
    $PY run.py robot    # YAM Pro cart screws the lids on (see robot_demo.py)        -> out/jars/ (main export)
    ~/projects/yam-arcade/.venvs/jars/bin/python checks.py                            -> out/jars/checks.json

Every part writes, under its output directory: the Recorder export (meta.json, traj.npz, meshes/), run_log.json
(per-step lid/jar state for the checks), preview.mp4 and PNG stills from a Genesis camera.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out", "jars")

MATERIALS = dict(
    jar=dict(kind="glass", color=(0.94, 0.97, 0.98), roughness=0.02),
    spice=dict(kind="plastic", color=(0.78, 0.10, 0.08), roughness=0.35),
    jam=dict(kind="metal", color=(0.83, 0.66, 0.30), roughness=0.25),
    mason=dict(kind="metal", color=(0.74, 0.75, 0.77), roughness=0.30),
)


class Session:
    """Steps an env, logs every step, records the export and the preview at 30 fps."""

    def __init__(self, env, out_dir, video=True, still_cams=("wide", "close")):
        import layout
        from export import Recorder
        import station as S

        self.S, self.env, self.out_dir = S, env, out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.rec = Recorder(out_dir, station="jars", fps=layout.RENDER_FPS)
        for n in S.NAMES:
            self.rec.add_rigid(f"jar_{n}", os.path.join(S.ASSETS, S.part(n)["jar_vis"]["file"]), material=MATERIALS["jar"])
            self.rec.add_rigid(f"lid_{n}", os.path.join(S.ASSETS, S.part(n)["lid_vis"]["file"]), material=MATERIALS[n])
        if env.robot is not None:
            env.robot.register(self.rec)
        self.every = int(round(1.0 / (layout.RENDER_FPS * env.dt)))
        self.k = 0
        self.log = []
        self.frames = []
        self.video = video
        self.phase = "start"
        self.extra = {}
        self.wall0 = time.time()
        self._frame()

    def _state(self):
        env, S = self.env, self.S
        P = env.poses()
        row = dict(t=env.t, phase=self.phase)
        for name, (p, R) in P.items():
            row[name] = dict(p=p.tolist(), R=R.reshape(-1).tolist())
        for n in S.NAMES:
            s = env.lid_on_jar(n, n, P)
            row[f"rel_{n}"] = dict(h=s["height_above_rim"], xy=s["offset_xy"], tilt=s["tilt_deg"],
                                   yaw=env.rel_yaw[(n, n)], eng=env.engaged_turn[(n, n)],
                                   tau=(float(env.drive.tau[n] @ P[f"jar_{n}"][1][:, 2]) if env.drive else 0.0))
        row.update(self.extra)
        return row

    def _frame(self):
        env = self.env
        rigid = {}
        for n in self.S.NAMES:
            for kind, ent in (("jar", env.jars[n]), ("lid", env.lids[n])):
                p, R = env.pose(ent)  # libuipc state, mesh-file frame (see JarsEnv.pose)
                rigid[f"{kind}_{n}"] = (p, self.S.R_to_quat(R))
        if env.robot is not None:
            self.rec.frame(env.t, rigid={**self.robot_poses(), **rigid})
        else:
            self.rec.frame(env.t, rigid=rigid)
        if self.video and "wide" in env.cams:
            self.frames.append(np.asarray(env.cams["wide"].render()[0]))

    TIP_LINKS = ("left_tip_left", "left_tip_right", "right_tip_left", "right_tip_right")

    def robot_poses(self):
        """Robot parts for the export. The four fingertip links are drawn at their libuipc (contact) poses: those are
        what touch the lids and jars. Genesis's own fingertip links sit 1-3 mm away from them while holding a part
        (two-way soft-coupling offset), which drew the fingertips up to 3.3 mm inside a lid. Everything else is drawn
        at Genesis's link poses; the difference shows as finger play in its mount (logged as tip_offset_m)."""
        poses = self.env.robot.poses()
        co = self.env.scene.sim.coupler
        for obj, link_name, _, _ in self.env.robot._parts:
            if link_name in self.TIP_LINKS:
                T = np.asarray(co._abd_data_by_link[self.env.robot.entity.get_link(link_name)][0].transform, float)
                U, _, Vt = np.linalg.svd(T[:3, :3])
                poses[obj] = (T[:3, 3].copy(), self.S.R_to_quat(U @ Vt))
        return poses

    def tip_offset(self):
        co = self.env.scene.sim.coupler
        worst = 0.0
        for ln in self.TIP_LINKS:
            link = self.env.robot.entity.get_link(ln)
            T = np.asarray(co._abd_data_by_link[link][0].transform, float)
            p = link.get_pos()
            p = np.asarray(p.cpu() if hasattr(p, "cpu") else p, float).reshape(3)
            worst = max(worst, float(np.linalg.norm(T[:3, 3] - p)))
        return worst

    def step(self, n=1):
        for _ in range(n):
            self.env.step()
            self.k += 1
            row = self._state()
            if self.env.robot is not None:
                row["tip_offset_m"] = self.tip_offset()
            self.log.append(row)
            if self.k % self.every == 0:
                self._frame()
            if self.k % 300 == 0:
                print(f"  t={self.env.t:6.2f}s phase {self.phase:10s} wall {time.time() - self.wall0:6.0f}s", flush=True)

    def still(self, name, cam="close"):
        import imageio

        img = np.asarray(self.env.cams[cam].render()[0])
        path = os.path.join(self.out_dir, f"{name}.png")
        imageio.imwrite(path, img)
        return path

    def frame_check(self):
        """Recorded pose x mesh file == libuipc's own world-space collision vertices (the compositor relies on it).
        The *_vis.obj files share the *_col.obj frame by construction (same lathe origin), so this covers both."""
        import trimesh

        out = {}
        co = self.env.scene.sim.coupler
        for n in self.S.NAMES:
            for kind, ent in (("jar", self.env.jars[n]), ("lid", self.env.lids[n])):
                p, R = self.env.pose(ent)
                m = trimesh.load(os.path.join(self.S.ASSETS, self.S.part(n)[f"{kind}_col"]["file"]), process=False)
                geo = co._abd_slots_by_link[ent.base_link][0].geometry()
                lv = np.asarray(geo.positions().view()).reshape(-1, 3)
                T = np.asarray(co._abd_data_by_link[ent.base_link][0].transform)
                v_ipc = (T[:3, :3] @ lv.T).T + T[:3, 3]  # libuipc's own world vertices (same order as the file)
                out[f"{kind}_{n}"] = float(np.abs(v_ipc - ((R @ m.vertices.T).T + p)).max())
        return out

    def close(self, extra_meta=None):
        import imageio

        extra_meta = dict(extra_meta or {})
        extra_meta["frame_check_max_vertex_error_m"] = self.frame_check()
        extra_meta.setdefault("camera", dict(eye=[0.30, 0.62, 1.00], target=[0.0, -0.07, 0.82], lens=50))
        extra_meta.setdefault("table", dict(top="white"))
        print("frame check (max |Genesis vertex - pose x mesh|, m):", extra_meta["frame_check_max_vertex_error_m"])
        self.rec.close(extra_meta=extra_meta)
        json.dump(self.log, open(os.path.join(self.out_dir, "run_log.json"), "w"))
        if self.frames:
            imageio.mimwrite(os.path.join(self.out_dir, "preview.mp4"), self.frames, fps=30, quality=8,
                             macro_block_size=8)


def run_settle(args):
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    import station as S

    env = S.build(gs, kind="puzzle", drive=False)
    out = args.out or os.path.join(OUT, "settle")
    ses = Session(env, out, video=not args.no_video)
    ses.still("start_close", "close")
    ses.still("start_wide", "wide")
    ses.phase = "settle"
    ses.step(int(round(args.settle / env.dt)))
    ses.still("settled_close", "close")
    ses.still("settled_wide", "wide")
    ses.close(extra_meta=dict(part="settle", note="puzzle layout settling under gravity, no drive",
                              dt=env.dt, d_hat=S.D_HAT))


def run_twist(args):
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    import station as S

    env = S.build(gs, kind="twist", drive=True)
    out = args.out or os.path.join(OUT, "twist_test")
    ses = Session(env, out, video=not args.no_video)
    ses.still("t0_on_mouth", "close")
    omega = args.omega
    tau0 = dict(env.drive.tau_max)
    ses.phase = "rest"
    ses.step(int(round(args.rest / env.dt)))
    ses.still("after_rest", "close")
    ses.phase = "tighten"
    for n in S.NAMES:
        env.drive.set_twist(n, -omega)
    ses.step(int(round(args.tighten / env.dt)))
    ses.still("after_tighten", "close")
    ses.phase = "hold"
    for n in S.NAMES:
        env.drive.set_twist(n, None)
    ses.step(int(round(args.hold / env.dt)))
    ses.still("after_hold", "close")
    ses.extra["success_after_hold"] = env.success()
    print("success after hold:", json.dumps(env.success()), flush=True)
    # unscrew each lid (opening clutch 1.5x the closing one, still below the jar's slip torque on the table) until
    # it has turned back args.unscrew_turns from seated, then let it go
    ses.phase = "unscrew"
    yaw_seated = {n: env.rel_yaw[(n, n)] for n in S.NAMES}
    for n in S.NAMES:
        env.drive.set_twist(n, omega, tau_max=1.5 * tau0[n])
    t_end = env.t + args.unscrew
    while env.t < t_end and any(env.drive.active.values()):
        for n in S.NAMES:
            if env.drive.active[n] and (env.rel_yaw[(n, n)] - yaw_seated[n]) / (2 * math.pi) >= args.unscrew_turns:
                env.drive.set_twist(n, None)
        ses.step(1)
    ses.still("after_unscrew", "close")
    ses.phase = "release"
    for n in S.NAMES:
        env.drive.set_twist(n, None)
    ses.step(int(round(args.release / env.dt)))
    ses.still("after_release", "close")
    ses.still("after_release_wide", "wide")
    phases = [("rest", args.rest), ("tighten", args.tighten), ("hold", args.hold),
              ("unscrew", f"<= {args.unscrew} s, until {args.unscrew_turns} turns back"), ("release", args.release)]
    meta = dict(part="twist_test", dt=env.dt, d_hat=S.D_HAT,
                drive=("FEASIBILITY TEST: an external torque about each jar axis is applied to the lid "
                       "(libuipc AffineBodyExternalBodyForce), from a torque-limited angle servo; no force, "
                       "no position/velocity is ever set. Phases: " + ", ".join(f"{p[0]} {p[1]} s" for p in phases)),
                omega_rad_s=omega, tau_max_close_Nm=tau0, tau_max_open_Nm={n: 1.5 * tau0[n] for n in tau0})
    ses.close(extra_meta=meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["settle", "twist", "robot"])
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--omega", type=float, default=math.pi, help="twist rate, rad/s")
    ap.add_argument("--rest", type=float, default=0.6)
    ap.add_argument("--tighten", type=float, default=2.8)
    ap.add_argument("--hold", type=float, default=0.6)
    ap.add_argument("--unscrew", type=float, default=5.0)
    ap.add_argument("--unscrew-turns", type=float, default=1.4)
    ap.add_argument("--jars", default=None, help="robot: comma list of jars to screw (default all, mason,jam,spice)")
    ap.add_argument("--out", default=None, help="output directory (default out/jars/<settle|twist_test>, robot: out/jars)")
    ap.add_argument("--dt", type=float, default=1.0 / 150.0, help="robot: sim step (settle/twist use station.DT)")
    ap.add_argument("--release", type=float, default=0.8)
    args = ap.parse_args()
    if args.part == "settle":
        run_settle(args)
    elif args.part == "twist":
        run_twist(args)
    else:
        import robot_demo

        robot_demo.main(args)


if __name__ == "__main__":
    main()
