"""Cables station in Newton: eight cables with plugs (VBD rods with curved rest shapes) dropped from the air into a
tangled pile on the table, two free bins, and the wheeled bimanual YAM Ultra (kinematic arms, force-servoed gripper)
picking cables out of the pile by their plugs and dropping them into the bin of their colour.

    cd ~/sims/newton && systemd-run --user --scope -q -p MemoryMax=24G .venv/bin/python \
        ~/projects/yam-arcade/stations/cables_newton/run.py [--sub 20 --iters 10] [--no-robot] [--picks 1]

Writes out/cables/newton/: meta.json + traj.npz + meshes/ (export.Recorder, station "cables", 30 fps), log.json
(per-frame physics log), timing.json, preview.mp4, still_*.png. check.py turns log.json into checks.json.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ncable as NC  # noqa: E402
import nrobot as NR  # noqa: E402
import layout  # noqa: E402
import export  # noqa: E402
import warp as wp  # noqa: E402
import newton  # noqa: E402
from newton.solvers import SolverVBD  # noqa: E402

OUT = os.path.join(NC.ROOT, "out", "cables", "newton")
ZT = NC.ZT
FPS = 60
GRIP_FORCE = 6.0  # N per finger (same cap as the Genesis station's GRIP_FORCE)
SIDE_FOR_COLOR = dict(orange="left", blue="right")  # bins: orange at -x (left arm), blue at +x (right arm)
CAMS = {
    "video": dict(pos=(0.78, 0.92, 1.48), lookat=(0.0, -0.12, 0.80), fov=46),  # facing the robot across the table
    "front": dict(pos=(0.62, -1.05, 1.42), lookat=(0.0, -0.03, 0.76), fov=42),  # the Genesis station's camera
    "top": dict(pos=(0.0, -0.06, 1.95), lookat=(0.0, -0.05, 0.75), fov=40),
    "close": dict(pos=(0.30, -0.62, 1.05), lookat=(0.0, -0.05, 0.76), fov=40),
}

_CURVE = json.load(open(os.path.join(HERE, "assets", "newton_display_curve.json")))
_CIN = np.array(sorted(float(k) for k in _CURVE))
_COUT = np.array([_CURVE[k][0] for k in sorted(_CURVE, key=float)])


def display_rgb(rgb):
    """Newton's GL viewer tone-maps colours (0.5 grey shows as 0.80); invert its measured response
    (franka-deformables calib/newton_display_curve.json) so previews show the specified colours."""
    return tuple(float(np.interp(c, _COUT, _CIN)) for c in rgb)


# ------------------------------------------------------------------------------------------------- kernels
@wp.kernel
def lerp_q(q0: wp.array[float], q1: wp.array[float], a: float, inv_dt: float, q: wp.array[float], qd: wp.array[float]):
    i = wp.tid()
    q[i] = q0[i] + a * (q1[i] - q0[i])
    qd[i] = (q1[i] - q0[i]) * inv_dt


@wp.kernel
def copy_robot(src_q: wp.array[wp.transform], src_qd: wp.array[wp.spatial_vector],
               q0: wp.array[wp.transform], q1: wp.array[wp.transform],
               qd0: wp.array[wp.spatial_vector], qd1: wp.array[wp.spatial_vector]):
    i = wp.tid()
    q0[i] = src_q[i]
    q1[i] = src_q[i]
    qd0[i] = src_qd[i]
    qd1[i] = src_qd[i]


# ------------------------------------------------------------------------------------------- initial layout
def pts_seg_dist(X, P):
    """Distance from each point of X (m, 3) to polyline P (n, 3): (m,)."""
    a, b = P[:-1][None], P[1:][None]
    d = b - a
    t = np.clip(((X[:, None] - a) * d).sum(-1) / (d * d).sum(-1), 0, 1)
    return np.linalg.norm(X[:, None] - (a + t[..., None] * d), axis=-1).min(1)


def plug_world_points(c, P):
    """Plug surface points in the world for a plug fixed on P[0] (same frame construction as ncable.add_cable)."""
    L_in = NC.PLUG_INNER * NC.plug_len(c["name"])
    t0 = (P[0] - P[1]) / np.linalg.norm(P[0] - P[1])
    x = t0
    up = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    z = up - up.dot(x) * x
    z /= np.linalg.norm(z)
    R = np.stack([x, np.cross(z, x), z], 1)
    return NC.plug_points(c["name"]) @ R.T + P[0], L_in


def choose_plug_ends(polys):
    """For each cable, put the plug on whichever end has more clearance from the other cables, the plugs placed so
    far and the table (the drop must start with nothing touching). Returns {name: polyline starting at the plug},
    {name: clearance in m}."""
    out, clear, plug_pts = {}, {}, {}
    for c in sorted(NC.CABLES, key=lambda c: -NC.PLUGS[c["name"]]["mass"]):  # big plugs first
        n = c["name"]
        best = None
        for P in (polys[n], polys[n][::-1]):
            X, _ = plug_world_points(c, P)
            d = [X[:, 2].min() - ZT] + [NC.bin_sdf(X, col).min() for col in NC.BINS]
            for m, Q in polys.items():
                if m != n:
                    d.append(pts_seg_dist(X, Q).min() - NC.CABLE_BY_NAME[m]["radius"])
            # own cable beyond the first 3 segments
            d.append(pts_seg_dist(X, P[3:]).min() - c["radius"])
            for m, Y in plug_pts.items():
                d.append(np.linalg.norm(X[:, None] - Y[None], axis=-1).min())
            dm = float(min(d))
            if best is None or dm > best[0]:
                best = (dm, P, X)
        out[n], clear[n], plug_pts[n] = best[1], best[0], best[2]
    return {c["name"]: out[c["name"]] for c in NC.CABLES}, clear


# ------------------------------------------------------------------------------------------------- station
class Station:
    def __init__(self, sub, iters, robot=True, res=(1280, 720), video=True, friction_eps=1e-3, cam="video",
                 solver_kw=None, calib_key=None, link_hulls=True):
        self.sub, self.iters, self.has_robot = sub, iters, robot
        assert sub % 2 == 0, "CUDA graph replay needs an even substep count"
        self.frame_dt = 1.0 / FPS
        self.dt = self.frame_dt / sub
        self.t = 0.0
        cal = NC.bend_scales().get(calib_key or f"{sub}x{iters}")
        assert cal is not None, f"no bend calibration for {sub}x{iters} (calib/cantilever.py)"
        self.cal = {k: v["scale"] for k, v in cal.items() if not k.startswith("_")}

        polys0 = {c["name"]: np.array(c["initial"]) for c in NC.CABLES}
        self.polys, self.plug_clear = choose_plug_ends(polys0)

        b = newton.ModelBuilder(gravity=-9.81)
        NC.register(b)
        self.rinfo = None
        if robot:
            self.rmodel, _ = NR.robot_only_model()
            self.rstate = self.rmodel.state()
            self.rinfo = NR.add_robot(b, link_hulls=link_hulls)
            self.kin = NR.Kin()
        self.table_shape, self.floor_shape = NC.add_table(b)
        self.bins = NC.add_bins(b)
        self.cabs = {}
        for c in NC.CABLES:
            self.cabs[c["name"]] = NC.add_cable(b, c, self.polys[c["name"]], bend_scale=self.cal[c["name"]], plug=True,
                                                color=display_rgb(NC.COLORS[c["color"]]))
        # preview colours (the viewer tone-maps): plugs, bins, table, robot
        for c in self.cabs.values():
            for s, m in zip(c["plug_shapes"], c["plug_mats"]):
                if m != "body":  # body parts already got the cable's display colour
                    b.shape_color[s] = display_rgb(b.shape_color[s])
        for col, bn in self.bins.items():
            for s in bn["shapes"]:
                b.shape_color[s] = display_rgb(NC.BIN_TINT[col])
        b.shape_color[self.table_shape] = display_rgb((0.78, 0.66, 0.50))
        if robot:
            for s in self.rinfo["robot_shapes"]:
                b.shape_color[s] = display_rgb(b.shape_color[s])
        b.color()
        self.model = b.finalize(requires_grad=False)
        flags = self.model.shape_flags.numpy()
        flags[self.floor_shape] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags.assign(flags)
        kw = dict(rigid_contact_hard=False, rigid_body_contact_buffer_size=512, friction_epsilon=friction_eps)
        kw.update(solver_kw or {})
        self.solver = SolverVBD(self.model, iterations=iters, **kw)
        NC.set_hard_stretch(self.solver, [j for c in self.cabs.values() for j in c["joints"]])
        self.s = [self.model.state(), self.model.state()]
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # shape categories for penetration bookkeeping
        nshape = self.model.shape_count
        self.shape_cat = np.full(nshape, "", dtype=object)
        self.shape_owner = np.full(nshape, "", dtype=object)
        self.shape_cat[self.table_shape] = "table"
        self.shape_cat[self.floor_shape] = "floor"
        for col, bn in self.bins.items():
            for s in bn["shapes"]:
                self.shape_cat[s], self.shape_owner[s] = "bin", col
        for n, c in self.cabs.items():
            for s in c["cable_shapes"]:
                self.shape_cat[s], self.shape_owner[s] = "cable", n
            for s in c["plug_shapes"]:
                self.shape_cat[s], self.shape_owner[s] = "plug", n
        self.pad_shapes = {}
        if robot:
            for side in NR.SIDES:
                for f, s in self.rinfo["pads"][side].items():
                    self.shape_cat[s], self.shape_owner[s] = "pad", f"{side}/{f}"
                    self.pad_shapes[s] = (side, f)
                self.shape_cat[self.rinfo["palms"][side]] = "palm"
            for s in self.rinfo["hulls"]:
                self.shape_cat[s] = "link"
        self.cat_names = ["", "table", "floor", "bin", "cable", "plug", "pad", "palm", "link"]
        self.cat_code = np.array([self.cat_names.index(x) for x in self.shape_cat])
        owners = sorted(set(self.shape_owner) - {""})
        self.owner_code = np.array([owners.index(x) + 1 if x else 0 for x in self.shape_owner])
        self.owner_names = [""] + owners
        self.ke = self.model.shape_material_ke.numpy()
        self.shape_body = self.model.shape_body.numpy()
        self.body_mass = self.model.body_mass.numpy()
        self.body_inertia = self.model.body_inertia.numpy()
        self.cable_bodies = np.concatenate([c["bodies"] for c in self.cabs.values()])

        # robot state
        if robot:
            self.nq = self.rmodel.joint_coord_count
            self.q = self.kin.q_home()
            self.q_prev = wp.array(self.q, dtype=float)
            self.q_tgt = wp.array(self.q, dtype=float)
            self.q_sub = wp.array(self.q, dtype=float)
            self.qd_sub = wp.zeros(self.rmodel.joint_dof_count, dtype=float)
            newton.eval_fk(self.rmodel, self.q_sub, self.qd_sub, self.rstate)
            wp.launch(copy_robot, dim=self.rinfo["n_bodies"], inputs=[self.rstate.body_q, self.rstate.body_qd],
                      outputs=[self.s[0].body_q, self.s[1].body_q, self.s[0].body_qd, self.s[1].body_qd])
            self.solver.body_q_prev = wp.clone(self.s[0].body_q)
            self.finger = {s_: NR.FINGER_Q_MAX for s_ in NR.SIDES}
            self.servo = {s_: dict(mode="open", k_est=5e3, prev=None, q0_est=None, q_free=None) for s_ in NR.SIDES}
            self.grip = {s_: (0.0, 0.0) for s_ in NR.SIDES}
        self.init_read = self.read()  # t = 0, before anything moves
        self.simulate()  # one uncaptured frame allocates every buffer; then capture (recording does not execute)
        self.t += self.frame_dt
        with wp.ScopedCapture() as cap:
            self.simulate()
        self.graph = cap.graph
        self.viewer = None
        if video:
            self._init_viewer(res, cam)

    # ---------------------------------------------------------------------------------------------- stepping
    def simulate(self):
        for k in range(self.sub):
            a, b_ = self.s[k % 2], self.s[(k + 1) % 2]
            if self.has_robot:
                wp.launch(lerp_q, dim=self.nq, inputs=[self.q_prev, self.q_tgt, (k + 1) / self.sub, 1.0 / self.frame_dt],
                          outputs=[self.q_sub, self.qd_sub])
                newton.eval_fk(self.rmodel, self.q_sub, self.qd_sub, self.rstate)
                wp.launch(copy_robot, dim=self.rinfo["n_bodies"], inputs=[self.rstate.body_q, self.rstate.body_qd],
                          outputs=[a.body_q, b_.body_q, a.body_qd, b_.body_qd])
            a.clear_forces()
            self.model.collide(a, self.contacts)
            self.solver.step(a, b_, self.control, self.contacts, self.dt)
        if self.has_robot:
            wp.copy(self.q_prev, self.q_tgt)

    def step(self, q_next=None):
        """Advance one 1/60 s frame; q_next = full robot joint target at the end of the frame."""
        if self.has_robot:
            q = self.q.copy() if q_next is None else np.array(q_next, float)
            for side in NR.SIDES:
                q[self.kin.finger_q[side]] = self.finger[side]
            self.q = q
            self.q_tgt.assign(q)
        wp.capture_launch(self.graph)
        self.t += self.frame_dt

    @property
    def state(self):
        return self.s[0]

    # ---------------------------------------------------------------------------------------------- reading
    def read(self):
        """Per-frame readback: body poses/velocities, cable centrelines, plug poses, contact penetrations, grip."""
        bq = self.state.body_q.numpy()
        bqd = self.state.body_qd.numpy()
        pts, gaps = {}, {}
        for n, c in self.cabs.items():
            pts[n], gaps[n] = NC.cable_nodes(bq, c)
        return bq, bqd, pts, gaps

    def contact_table(self, bq):
        """Current rigid contacts with their penetration re-evaluated at the current body poses:
        (s0, s1, pen, normal, world point on shape 0). Positive pen = overlap."""
        c = self.contacts
        n = int(c.rigid_contact_count.numpy()[0])
        if n == 0:
            z = np.zeros(0)
            return z.astype(int), z.astype(int), z, np.zeros((0, 3)), np.zeros((0, 3))
        s0 = c.rigid_contact_shape0.numpy()[:n]
        s1 = c.rigid_contact_shape1.numpy()[:n]
        p0 = c.rigid_contact_point0.numpy()[:n]
        p1 = c.rigid_contact_point1.numpy()[:n]
        nrm = c.rigid_contact_normal.numpy()[:n]
        m0 = c.rigid_contact_margin0.numpy()[:n]
        m1 = c.rigid_contact_margin1.numpy()[:n]
        ok = (s0 >= 0) & (s1 >= 0)
        s0, s1, p0, p1, nrm, m0, m1 = s0[ok], s1[ok], p0[ok], p1[ok], nrm[ok], m0[ok], m1[ok]
        b0, b1 = self.shape_body[s0], self.shape_body[s1]

        def world(b, p):
            out = p.copy()
            m = b >= 0
            if m.any():
                T = bq[b[m]]
                R = np.stack([NC.quat_to_mat_xyzw(t[3:7]) for t in T])
                out[m] = np.einsum("nij,nj->ni", R, p[m]) + T[:, :3]
            return out

        w0, w1 = world(b0, p0), world(b1, p1)
        pen = (m0 + m1) - np.einsum("ni,ni->n", nrm, w1 - w0)
        return s0, s1, pen, nrm, w0

    def penetration_summary(self, s0, s1, pen):
        """Max penetration (mm) per category pair ('cable-cable' between different cables, 'cable-self' within one)."""
        m = pen > 0
        if not m.any():
            return {}
        a, b_ = self.cat_code[s0[m]], self.cat_code[s1[m]]
        lo, hi = np.minimum(a, b_), np.maximum(a, b_)
        key = lo * 16 + hi
        cab = self.cat_names.index("cable")
        same = (lo == cab) & (hi == cab) & (self.owner_code[s0[m]] == self.owner_code[s1[m]])
        key = np.where(same, 255, key)
        out = {}
        for k in np.unique(key):
            name = "cable-self" if k == 255 else f"{self.cat_names[k // 16]}-{self.cat_names[k % 16]}"
            out[name] = 1e3 * float(pen[m][key == k].max())
        return out

    def grip_forces(self, bq, s0, s1, pen, nrm):
        """Inward squeeze (N) on each fingertip pad of each arm from VBD's soft contact law (avg ke x penetration),
        counting all non-robot shapes the pad touches; and which cables' shapes each arm touches."""
        out = {side: [0.0, 0.0] for side in NR.SIDES}
        touched = {side: set() for side in NR.SIDES}
        if not self.has_robot:
            return out, touched
        stf = self.model.shape_transform.numpy() if not hasattr(self, "_stf") else self._stf
        self._stf = stf
        sel = np.where((pen > 0) & (np.isin(s0, list(self.pad_shapes)) | np.isin(s1, list(self.pad_shapes))))[0]
        if len(sel) == 0:
            return out, touched
        s0, s1, pen, nrm = s0[sel], s1[sel], pen[sel], nrm[sel]
        for side in NR.SIDES:
            pl, pr = self.rinfo["pads"][side]["tip_left"], self.rinfo["pads"][side]["tip_right"]
            cl = NC.tf_mat(bq[self.shape_body[pl]]) @ NC.tf_mat(stf[pl])
            cr = NC.tf_mat(bq[self.shape_body[pr]]) @ NC.tf_mat(stf[pr])
            axis = cr[:3, 3] - cl[:3, 3]
            axis /= np.linalg.norm(axis)
            for i in range(len(pen)):
                if pen[i] <= 0:
                    continue
                for k, (ps, inward) in enumerate(((pl, axis), (pr, -axis))):
                    if s0[i] == ps:
                        other, sgn = s1[i], -1.0  # force on shape 0 is -n F
                    elif s1[i] == ps:
                        other, sgn = s0[i], 1.0
                    else:
                        continue
                    if self.shape_cat[other] in ("pad", "palm", "link"):
                        continue
                    F = 0.5 * (self.ke[s0[i]] + self.ke[s1[i]]) * pen[i]
                    # reaction on the pad points away from the object: its outward (opening) component = squeeze
                    out[side][k] += F * max(0.0, -sgn * float(np.dot(nrm[i], inward)))
                    if self.shape_owner[other]:
                        touched[side].add(self.shape_owner[other])
        return out, touched

    def pad_zmin(self, bq):
        """Lowest corner of the four fingertip pad boxes (m)."""
        if not hasattr(self, "_pad_geo"):
            stf, ssc = self.model.shape_transform.numpy(), self.model.shape_scale.numpy()
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
            self._pad_geo = [(self.shape_body[s], NC.tf_mat(stf[s]), corners * ssc[s]) for s in self.pad_shapes]
        z = np.inf
        for b_, T, C in self._pad_geo:
            M = NC.tf_mat(bq[b_]) @ T
            z = min(z, float((C @ M[:3, :3].T + M[:3, 3])[:, 2].min()))
        return z

    # ---------------------------------------------------------------------------------------------- gripper
    def servo_step(self, side, cmd):
        """Gripper force servo (franka cable lesson). cmd 'open' | 'close' | 'hold'. Pad contact is a spring,
        F = k (q0 - q), past the touch point q0: close at full speed until the first touch, then aim at q0 - G/k each
        frame; a 0 N reading at q bounds q0 <= q; the first touch bounds k >= F / (q_free - q); secant updates."""
        S = self.servo[side]
        step = NR.FINGER_SPEED * self.frame_dt
        # one motor + rack drives both fingers (I2RT linear_4310, joint7 = joint8): it balances the SUM of the two
        # finger loads, so the servo regulates the mean squeeze; one finger alone on an anchored object stalls at 2 G
        f = 0.5 * sum(self.grip[side])
        q = self.finger[side]
        if cmd == "open":
            S.update(prev=None, q0_est=None, q_free=None)
            self.finger[side] = min(NR.FINGER_Q_MAX, q + step)
            return
        if cmd == "width":
            return
        if f > 0.0:
            p = S["prev"]
            if p is not None and p[1] > 0.0 and p[0] - q > 1e-6 and f > p[1]:
                S["k_est"] = float(np.clip(0.5 * S["k_est"] + 0.5 * (f - p[1]) / (p[0] - q), 5e2, 5e5))
            elif S["q_free"] is not None and S["q_free"] - q > 1e-6:
                S["k_est"] = float(np.clip(max(S["k_est"], f / (S["q_free"] - q)), 5e2, 5e5))
            S["q0_est"] = q + f / S["k_est"] if S["q_free"] is None else min(q + f / S["k_est"], S["q_free"])
        else:
            S["q_free"] = q
            if S["q0_est"] is not None:
                S["q0_est"] = min(S["q0_est"], q)
        S["prev"] = (q, f)
        if S["q0_est"] is None:
            q_new = q - step
        else:
            q_new = q + float(np.clip(S["q0_est"] - GRIP_FORCE / S["k_est"] - q, -step, step))
        self.finger[side] = float(np.clip(q_new, 0.0, NR.FINGER_Q_MAX))

    # ---------------------------------------------------------------------------------------------- viewer
    def _init_viewer(self, res, cam):
        self.viewer = newton.viewer.ViewerGL(width=res[0], height=res[1], headless=True)
        self.viewer.set_model(self.model)
        r = self.viewer.renderer
        r.draw_sky = False
        r.sky_upper = tuple(layout.BACKGROUND_RGB)
        r.sky_lower = tuple(layout.BACKGROUND_RGB)
        fl = 6.0
        self.floor_pts = wp.array(np.array([[-fl, -fl, 0], [fl, -fl, 0], [fl, fl, 0], [-fl, fl, 0]], np.float32),
                                  dtype=wp.vec3)
        self.floor_idx = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32)
        self.set_cam(cam)

    def set_cam(self, name):
        c = CAMS[name]
        (cx, cy, cz), (lx, ly, lz) = c["pos"], c["lookat"]
        dx, dy, dz = lx - cx, ly - cy, lz - cz
        self.viewer.camera.fov = c["fov"]
        self.viewer.set_camera(wp.vec3(cx, cy, cz), math.degrees(math.atan2(dz, math.hypot(dx, dy))),
                               math.degrees(math.atan2(dy, dx)))

    def render(self):
        v = self.viewer
        v.begin_frame(self.t)
        v.log_state(self.state)
        v.log_mesh("/floor", self.floor_pts, self.floor_idx, color=display_rgb(layout.FLOOR_RGB), backface_culling=False)
        v.end_frame()
        return v.get_frame().numpy()


def write_mp4(path, frames, fps):
    h, w, _ = frames[0].shape
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()


def write_png(path, img):
    from PIL import Image
    Image.fromarray(np.ascontiguousarray(img)).save(path)


# ------------------------------------------------------------------------------------------------- pick planning
def min_jerk(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def plug_grip_geometry(name):
    """Grip point (plug frame), half width across the grip (plug y) and half thickness (plug z) of the gripped part."""
    kind, gx = NC.PLUGS[name]["grip"]
    for p in NC.PLUGS[name]["parts"]:
        if p[-1] == "metal":
            continue
        if p[0] == "box" and kind == "box" and abs(p[1][0] - gx) <= p[2][0] + 1e-9:
            return np.array([gx, 0.0, 0.0]), p[2][1], p[2][2]
        if p[0] == "cap" and kind == "cap" and p[1] <= gx <= p[2]:
            return np.array([gx, p[4], 0.0]), p[3], p[3]
    raise ValueError(name)


def cart_path(kin, side, keys, q_start):
    """Joint targets per frame through TCP keyframes [dict(pos, yaw, tilt, t)] (the first is the start), min-jerk in
    Cartesian space, IK every frame seeded with the previous solution. Returns (list of full q, worst IK error)."""
    q = np.array(q_start, float).copy()
    out, worst = [], 0.0
    for k0, k1 in zip(keys, keys[1:]):
        p0, p1 = np.array(k0["pos"]), np.array(k1["pos"])
        # duration: the keyframe's, or longer so that the TCP stays under 0.25 m/s and the wrist under ~1 rad/s
        tmin = max(np.linalg.norm(p1 - p0) / 0.25, abs(k1["yaw"] - k0["yaw"]) / 1.0, abs(k1["tilt"] - k0["tilt"]) / 0.8)
        n = max(1, int(math.ceil(max(k1["t"], 1.875 * tmin) * FPS)))  # min-jerk peak speed = 1.875 x mean
        for j in range(1, n + 1):
            s = min_jerk(j / n)
            R = NR.grasp_R(k0["yaw"] + s * (k1["yaw"] - k0["yaw"]), k0["tilt"] + s * (k1["tilt"] - k0["tilt"]))
            qa, err = kin.ik(side, p0 + s * (p1 - p0), R, q)
            worst = max(worst, err)
            q = q.copy()
            q[kin.arm_q[side]] = qa
            out.append(q)
    return out, worst


# ------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=40)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--no-robot", action="store_true")
    ap.add_argument("--picks", type=int, default=2, help="cables to sort (one per arm, alternating)")
    ap.add_argument("--t-settle", type=float, default=4.0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--res", type=int, nargs=2, default=(1280, 720))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--friction-eps", type=float, default=1e-3)
    ap.add_argument("--max-time", type=float, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    print(f"GPU at start: {smi}; {len(apps)} compute processes", flush=True)

    t_build = time.time()
    st = Station(args.sub, args.iters, robot=not args.no_robot, res=tuple(args.res), video=not args.no_video,
                 friction_eps=args.friction_eps)
    wp.synchronize()
    t_build = time.time() - t_build
    print(f"built in {t_build:.1f} s: {st.model.body_count} bodies, {st.model.shape_count} shapes, "
          f"{st.model.joint_count} joints; plug clearances (mm) "
          f"{ {k: round(1e3 * v, 1) for k, v in st.plug_clear.items()} }", flush=True)

    # ---- export registration
    rec = export.Recorder(args.out, station="cables", fps=layout.RENDER_FPS)
    os.makedirs(NC.ASSETS, exist_ok=True)
    bm = os.path.join(NC.ASSETS, "bin.obj")
    NC.bin_mesh().export(bm)
    lm = os.path.join(NC.ASSETS, "bin_label.obj")
    NC.label_mesh().export(lm)
    for col in NC.BINS:
        rec.add_rigid(f"bin_{col}", bm, material=dict(kind="plastic", color=NC.BIN_TINT[col], roughness=0.45))
        rec.add_rigid(f"bin_{col}_label", lm, material=dict(kind="paint", color=NC.COLORS[col], roughness=0.6))
    plug_parts = {}
    for n, cab in st.cabs.items():
        c = cab["spec"]
        rec.add_rod(n, radius=c["radius"], material=dict(kind="rubber", color=NC.COLORS[c["color"]], roughness=0.55))
        for mat, mesh in NC.plug_meshes(n).items():
            f = os.path.join(NC.ASSETS, f"plug_{n}_{mat}.obj")
            mesh.export(f)
            col = dict(body=dict(kind="rubber", color=NC.COLORS[c["color"]], roughness=0.5),
                       metal=dict(kind="metal", color=NC.METAL_RGB, roughness=0.3),
                       clear=dict(kind="plastic", color=NC.CLEAR_RGB, roughness=0.2))[mat]
            rec.add_rigid(f"plug_{n}/{mat}", f, material=col)
            plug_parts.setdefault(n, []).append(f"plug_{n}/{mat}")
    robot_parts = []
    if st.has_robot:
        for obj, link, mesh, mat in NR.export_parts():
            rec.add_rigid(obj, mesh, material=mat)
            robot_parts.append((obj, st.rinfo["bodies"][link]))

    def rigid_poses(bq):
        out = {}
        for col, bn in st.bins.items():
            M = NC.tf_mat(bq[bn["body"]])
            pq = (M[:3, 3], NR_quat(M[:3, :3]))
            out[f"bin_{col}"] = pq
            out[f"bin_{col}_label"] = pq
        for n, cab in st.cabs.items():
            M = NC.plug_pose(bq, cab)
            pq = (M[:3, 3], NR_quat(M[:3, :3]))
            for o in plug_parts[n]:
                out[o] = pq
        for obj, bi in robot_parts:
            M = NC.tf_mat(bq[bi])
            out[obj] = (M[:3, 3], NR_quat(M[:3, :3]))
        return out

    # ---- per-frame bookkeeping
    seg_rest = {n: c["seg"] for n, c in st.cabs.items()}
    log, frames, timing = [], [], {}
    pen_max_all = {}
    prev_pts = None
    prev_q = [None]

    def bookkeeping(phase, extra=None, record_frame=False):
        nonlocal prev_pts
        bq, bqd, pts, gaps = st.read()
        s0, s1, pen, nrm, _ = st.contact_table(bq)
        pens = st.penetration_summary(s0, s1, pen)
        for k, v in pens.items():
            pen_max_all[k] = max(pen_max_all.get(k, 0.0), v)
        if st.has_robot:
            g, touched = st.grip_forces(bq, s0, s1, pen, nrm)
            st.grip = {k: tuple(v) for k, v in g.items()}
        lo, hi, zg = np.inf, 0.0, np.inf
        on_table = {}
        for n, P in pts.items():
            ratio = np.linalg.norm(np.diff(P, axis=0), axis=1) / seg_rest[n]
            lo, hi = min(lo, float(ratio.min())), max(hi, float(ratio.max()))
            r = NC.CABLE_BY_NAME[n]["radius"]
            inside = (np.abs(P[:, 0]) < 0.5 * layout.TABLE["size_x"]) & (np.abs(P[:, 1]) < 0.5 * layout.TABLE["size_y"])
            if inside.any():
                zg = min(zg, float((P[inside, 2] - ZT - r).min()))
            on_table[n] = bool(inside.all())
        cb = st.cable_bodies
        v = bqd[cb]
        ke = 0.5 * float((st.body_mass[cb] * (v[:, :3] ** 2).sum(1)).sum())
        ke += 0.5 * float(np.einsum("ni,nij,nj->", v[:, 3:], st.body_inertia[cb], v[:, 3:]))
        # centreline velocity between consecutive frames (the Genesis station's definition); `vmax_body` / `ke` are the
        # solver's instantaneous body velocities of the last 1/2400 s substep, which include micrometre contact noise
        vf, kef = 0.0, 0.0
        if prev_pts is not None:
            for n, P in pts.items():
                u = np.linalg.norm(P - prev_pts[n], axis=1) / st.frame_dt
                vf = max(vf, float(u.max()))
                kef += 0.5 * NC.CABLE_BY_NAME[n]["rho_line"] * seg_rest[n].mean() * float((u ** 2).sum())
        r = dict(t=round(st.t, 5), phase=phase, stretch=[lo, hi], table_gap_mm=1e3 * zg, ke=kef, vmax=vf,
                 ke_body=ke, vmax_body=float(np.linalg.norm(v[:, :3], axis=1).max()), pen_mm=pens,
                 joint_gap_mm=1e3 * max(float(g_.max()) for g_ in gaps.values()), nan=bool(np.isnan(bq).any()))
        if st.has_robot:
            r["pad_zmin"] = st.pad_zmin(bq)
            arm = np.concatenate([st.kin.arm_q[s_] for s_ in NR.SIDES])
            if prev_q[0] is not None:
                r["arm_qd_max"] = float(np.abs(st.q[arm] - prev_q[0][arm]).max() / st.frame_dt)
            prev_q[0] = st.q.copy()
        if st.has_robot:
            r["grip"] = {k: [round(x, 3) for x in v_] for k, v_ in st.grip.items()}
            r["finger_q"] = dict(st.finger)
            r["touched"] = {k: sorted(v_) for k, v_ in touched.items()}
        for col, bn in st.bins.items():
            r[f"bin_{col}"] = bq[bn["body"]].tolist()
        if extra:
            r.update(extra)
        log.append(r)
        if record_frame:
            rec.frame(st.t, rigid=rigid_poses(bq), rods=pts)
            if st.viewer is not None:
                frames.append(st.render())
        prev_pts = pts
        return bq, pts

    # t = 0 record (initial layout) from the pre-warm-up readback
    bq0, _, pts0, _ = st.init_read
    rec.frame(0.0, rigid=rigid_poses(bq0), rods=pts0)
    init_gaps = NC.min_cable_gaps(pts0)
    nframe = 1  # the warm-up frame ran in Station()

    def run_frames(n, phase, q_list=None, cmd=None, extra_fn=None):
        nonlocal nframe
        for i in range(n):
            if st.has_robot and cmd is not None:
                for side, c_ in cmd.items():
                    st.servo_step(side, c_)
            st.step(None if q_list is None else q_list[min(i, len(q_list) - 1)])
            nframe += 1
            ex = extra_fn() if extra_fn else None
            bookkeeping(phase, ex, record_frame=(nframe % (FPS // layout.RENDER_FPS) == 0))
            if args.max_time is not None and st.t > args.max_time:
                return False
        return True

    # ---- drop + settle
    t0 = time.time()
    n_settle = int(round(args.t_settle * FPS))
    run_frames(n_settle - 1, "drop_settle")
    wp.synchronize()
    timing["drop_settle"] = dict(sim_s=args.t_settle, wall_s=time.time() - t0)
    timing["drop_settle"]["wall_per_sim_s"] = timing["drop_settle"]["wall_s"] / args.t_settle
    bq, _, pts, _ = st.read()
    settled = {n: P.copy() for n, P in pts.items()}
    settle = dict(t=st.t, ke=log[-1]["ke"], vmax=log[-1]["vmax"], tangle=NC.tangle_metric(settled),
                  retention={n: NC.retention(P, st.cabs[n]["P0"]) for n, P in settled.items()},
                  plugs={n: NC.plug_pose(bq, c).tolist() for n, c in st.cabs.items()})
    print(f"settled at t={st.t:.2f}: ke {settle['ke']:.2e} J vmax {settle['vmax']:.4f} m/s; wall "
          f"{timing['drop_settle']['wall_s']:.1f} s ({timing['drop_settle']['wall_per_sim_s']:.2f} s/s); tangle "
          f"{ {k: settle['tangle'][k] for k in ('crossings', 'contacts', 'touching_pairs', 'interlocked_pairs')} }",
          flush=True)
    print("retention", {n: round(v["retention"], 3) for n, v in settle["retention"].items()}, flush=True)
    status_note(args, st, timing, settle, smi)

    # ---- robot demo
    picks = []
    if st.has_robot and args.picks > 0:
        picker = Picker(st, run_frames, timing)
        picker.still_dir = os.path.join(args.out, "stills")
        os.makedirs(picker.still_dir, exist_ok=True)
        done, last_side = [], None
        for k in range(args.picks):
            bq, _, pts, _ = st.read()
            cands = plug_candidates(st, bq, pts, exclude=done,
                                    sides=None if last_side is None else [s_ for s_ in NR.SIDES if s_ != last_side])
            print("candidates:", [(c["cable"], c["side"], c["blocked"], round(c["score"], 3)) for c in cands], flush=True)
            res = None
            for cand in cands:
                res = sort_cable(picker, k, cand)
                if res is not None:
                    break
            if res is None:
                break
            res["candidates"] = cands
            picks.append(res)
            done.append(res["cable"])
            last_side = res["side"]
            if args.max_time is not None and st.t > args.max_time:
                break
    wp.synchronize()
    bq, _, pts, _ = st.read()
    final = dict(t=st.t, success=success(st, bq, pts), tangle=NC.tangle_metric(pts))
    timing["total"] = dict(sim_s=st.t, wall_s=sum(v["wall_s"] for k, v in timing.items() if k != "total"))
    timing["build_s"] = t_build
    rec.close(extra_meta=dict(engine="newton", solver=f"SolverVBD {args.sub} substeps x {args.iters} iterations at 60 fps",
                              camera=dict(eye=[0.42, -0.78, 1.18], target=[0.0, -0.06, 0.77], lens=35.0),
                              cables={n: dict(color=c["spec"]["color"], EI=c["spec"]["EI"], length=float(c["seg"].sum()),
                                              plug=NC.PLUGS[n]["kind"]) for n, c in st.cabs.items()}))
    smi_end = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    json.dump(dict(log=log, settle=settle, picks=picks, final=final, timing=timing, pen_max=pen_max_all,
                   init_gaps_mm={k: 1e3 * v for k, v in init_gaps.items()},
                   plug_clear_mm={k: 1e3 * v for k, v in st.plug_clear.items()}, args=vars(args),
                   bend_scale=st.cal, gpu=dict(start=smi, end=smi_end, n_apps_start=len(apps)),
                   counts=dict(bodies=st.model.body_count, shapes=st.model.shape_count, joints=st.model.joint_count)),
              open(os.path.join(args.out, "log.json"), "w"), default=float)
    json.dump(timing, open(os.path.join(args.out, "timing.json"), "w"), indent=1)
    if frames:
        write_mp4(os.path.join(args.out, "preview.mp4"), frames, layout.RENDER_FPS)
        write_png(os.path.join(args.out, "still_start.png"), frames[0])
        write_png(os.path.join(args.out, "still_settled.png"), frames[min(len(frames) - 1, int(args.t_settle * 30) - 1)])
        write_png(os.path.join(args.out, "still_end.png"), frames[-1])
        for name in ("top", "close", "front"):
            st.set_cam(name)
            write_png(os.path.join(args.out, f"still_{name}.png"), st.render())
    print("final:", json.dumps(final["success"]), flush=True)
    print("timing:", json.dumps(timing), flush=True)


def NR_quat(R):
    """3x3 -> (w, x, y, z)."""
    return NC.xyzw_to_wxyz(NC.mat_to_quat_xyzw(R))


def success(st, bq, pts):
    """Per cable: fraction of its centreline nodes inside each bin's cavity (Genesis station's definition)."""
    lo, hi = NC.bin_inner_box()
    per = {}
    for n, P in pts.items():
        fr = {}
        for col, bn in st.bins.items():
            M = NC.tf_mat(bq[bn["body"]])
            L = (P - M[:3, 3]) @ M[:3, :3]
            fr[col] = float(np.all((L > lo) & (L < hi), axis=1).mean())
        best = max(fr, key=fr.get)
        in_bin = best if fr[best] > 0.5 else None
        per[n] = dict(fractions=fr, bin=in_bin, correct=in_bin == NC.CABLE_BY_NAME[n]["color"])
    return dict(per_cable=per, sorted_fraction=float(np.mean([v["correct"] for v in per.values()])))


def wrap_yaw(yaw):
    return (yaw + math.pi / 2) % math.pi - math.pi / 2


def sweep_hits(st, bq, pts, side, q, finger_q, own=None, own_keep=None, rise=0.09):
    """Collision test of a planned grasp: the two pad boxes at joint targets q with the fingers at finger_q, swept
    back up along the approach by `rise` (the descent). Counts cable nodes (inflated by their radius) and plug surface
    points of anything but the grasped object inside the swept pads, and returns the lowest pad corner z."""
    qq = np.array(q, float).copy()
    qq[st.kin.finger_q[side]] = finger_q
    boxes = st.kin.pad_boxes(side, qq)
    a = st.kin.tcp(side, qq)[1][:, 2]  # approach (flange +z)
    X, rad = [], []
    for m, P in pts.items():
        keep = np.ones(len(P), bool) if (m != own or own_keep is None) else ~own_keep
        X.append(P[keep])
        rad.append(np.full(keep.sum(), NC.CABLE_BY_NAME[m]["radius"]))
    for m, cab2 in st.cabs.items():
        if m == own and own_keep is None:
            continue
        Y = (NC.plug_pose(bq, cab2) @ np.c_[NC.plug_points(m), np.ones(len(NC.plug_points(m)))].T).T[:, :3]
        X.append(Y)
        rad.append(np.full(len(Y), 0.0005))
    X, rad = np.vstack(X), np.concatenate(rad)
    hits, zmin = 0, np.inf
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    for c, R, h in boxes:
        zmin = min(zmin, float(((corners * h) @ R.T + c)[:, 2].min()))
        hit = np.zeros(len(X), bool)
        for sft in np.linspace(0.0, rise, 10):
            L = (X - (c - sft * a)) @ R
            hit |= np.all(np.abs(L) < h + rad[:, None], axis=1)
        hits += int(hit.sum())
    return hits, zmin


def plan_grasp(st, bq, pts, side, g, lat, width_open, z_want, z_floor, own, own_keep, tilts=(0.0, 0.2, 0.35, 0.5)):
    """Top-down grasp if reachable, else pitched (tilt about the closing axis). TCP (pad bottom centre) at z_want,
    raised if a tilted pad corner would come within 2 mm of z_floor, but never above the object's centre g[2] (the
    pads must cover at least the lower half of its sides). Returns (yaw, tilt, tcp z, hits) for the first reachable
    tilt with a clear swept volume, or None."""
    yaw = wrap_yaw(math.atan2(lat[1], lat[0]))
    fq = (width_open - NR.PAD_GAP0) / 2
    for tl in tilts:
        z = z_want
        for _ in range(2):
            q_arm, err = st.kin.ik(side, (g[0], g[1], z), NR.grasp_R(yaw, tl), st.q)
            if err > 1e-3:
                break
            q = st.q.copy()
            q[st.kin.arm_q[side]] = q_arm
            hits, zmin = sweep_hits(st, bq, pts, side, q, fq, own=own, own_keep=own_keep)
            if zmin >= z_floor + 0.0015:
                break
            z += z_floor + 0.002 - zmin
        if err > 1e-3 or zmin < z_floor + 0.0015 or z > g[2]:
            continue
        if hits == 0:
            return yaw, tl, z, 0
    return None


def plug_candidates(st, bq, pts, exclude=(), sides=None):
    """Plug grasps: each cable's plug lying roughly flat with its axis roughly horizontal, reachable by the arm on its
    bin's side (top-down, or pitched about the closing axis for plugs beyond the top-down reach), the swept pads
    clear of everything else. Best first: clear of the pile (fewest cables lying over it), short, close to its bin."""
    cands = []
    for n, cab in st.cabs.items():
        if n in exclude:
            continue
        c = cab["spec"]
        side = SIDE_FOR_COLOR[c["color"]]
        if sides is not None and side not in sides:
            continue
        Pw = NC.plug_pose(bq, cab)
        gp, hw, ht = plug_grip_geometry(n)
        g = (Pw @ np.r_[gp, 1.0])[:3]
        xw, yw, zw = Pw[:3, 0], Pw[:3, 1], Pw[:3, 2]
        flat = abs(zw[2]) if NC.PLUGS[n]["grip"][0] == "box" else 1.0
        if abs(xw[2]) > 0.35 or flat < 0.85:
            continue
        lat = np.array([yw[0], yw[1], 0.0])
        lat /= np.linalg.norm(lat)
        width_open = 2 * hw + 0.024
        zbot = g[2] - ht
        z_want = max(ZT + 0.002, g[2] - 0.6 * ht)
        z_floor = ZT  # the table; whatever else lies under the pads is caught by the swept-volume test
        gr = plan_grasp(st, bq, pts, side, g, lat, width_open, z_want, z_floor, own=n,
                        own_keep=np.arange(len(pts[n])) <= 2)
        if gr is None:
            continue
        yaw, tilt, z_tcp, _ = gr
        bx = NC.BINS[c["color"]]["xy"][0]
        uc = under_crossings(pts, n)
        score = -0.5 * abs(g[0] - bx) - 0.4 * c["length"] - 0.08 * uc - 0.3 * tilt
        cands.append(dict(kind="plug", cable=n, side=side, grip=g.tolist(), yaw=yaw, tilt=tilt, pad_bottom=z_tcp,
                          width_open=width_open, blocked=0, score=score, plug_bottom=zbot, under_crossings=uc))
    cands.sort(key=lambda d: -d["score"])
    return cands


def bin_local(st, bq, col, P):
    M = NC.tf_mat(bq[st.bins[col]["body"]])
    return (P - M[:3, 3]) @ M[:3, :3]


def feed_candidates(st, bq, pts, n, side):
    """Cable-body grasps that feed the part of cable n still outside its bin into it. The outside part is split into
    runs of consecutive nodes; the longest run is fed first. A run that ends at the cable's free end (still caught in
    the pile) is fed from that end: lifting a middle node instead would pull the plug back out over the rim. A run
    bounded by the bin on both sides (a loop hanging over the rim) is fed from its middle. Grasp nodes: >= 4 cm outside
    the bin walls, pad zones free, reachable top-down."""
    c = NC.CABLE_BY_NAME[n]
    col = c["color"]
    P = pts[n]
    N = len(P)
    L = bin_local(st, bq, col, P)
    a, bh = 0.5 * NC.BIN["size"][0], 0.5 * NC.BIN["size"][1]
    inside_xy = (np.abs(L[:, 0]) < a) & (np.abs(L[:, 1]) < bh)
    clear_of_bin = (np.abs(L[:, 0]) > a + 0.04) | (np.abs(L[:, 1]) > bh + 0.04)
    runs, i = [], 0
    while i < N:
        if inside_xy[i]:
            i += 1
            continue
        j = i
        while j + 1 < N and not inside_xy[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    if not runs:
        return []
    i0, i1 = max(runs, key=lambda r: r[1] - r[0])
    # crossings where this run lies under another cable: is its free end trapped in the pile?
    under = 0
    for m, Q in pts.items():
        if m == n:
            continue
        for ii, jj, s_, t_ in NC._seg_intersections_2d(P[i0:i1 + 1], Q):
            za = P[i0 + ii, 2] + s_ * (P[i0 + ii + 1, 2] - P[i0 + ii, 2])
            zb = Q[jj, 2] + t_ * (Q[jj + 1, 2] - Q[jj, 2])
            under += int(za < zb)
    if i1 == N - 1 and i0 > 0:
        # tail run: a free tail is lifted in by its end; a trapped one is pulled out hand over hand, grasped next to
        # the bin, so it slides out of the pile lengthwise instead of being lifted against the cables on top of it
        target = (N - 3) if under == 0 else i0
    elif i0 == 0 and i1 < N - 1:
        target = i1 if under else 2
    else:
        target = 0.5 * (i0 + i1)
    r = c["radius"]
    cands = []
    for i in range(max(2, i0), min(N - 2, i1 + 1)):
        if not clear_of_bin[i]:
            continue
        t = P[i + 1] - P[i - 1]
        lat = np.array([-t[1], t[0], 0.0])
        lat /= np.linalg.norm(lat)
        width_open = 2 * r + 0.024
        if abs(i - target) > 12:  # only grasp near the planned point (each plan_grasp costs a few IK solves)
            continue
        keep = np.abs(np.arange(N) - i) <= 3
        z_floor = ZT
        gr = plan_grasp(st, bq, pts, side, P[i], lat, width_open, max(ZT + 0.0015, P[i, 2] - r + 0.0005), z_floor,
                        own=n, own_keep=keep, tilts=(0.0, 0.2, 0.35))
        if gr is None:
            continue
        yaw, tilt, z_tcp, _ = gr
        score = -abs(i - target) - 10.0 * max(0.0, P[i, 2] - r - ZT - 0.002) / 0.01 - 2.0 * tilt
        cands.append(dict(kind="cable", cable=n, side=side, node=i, run=[i0, i1], run_under=under,
                          grip=P[i].tolist(), yaw=yaw, tilt=tilt, pad_bottom=z_tcp, width_open=width_open, blocked=0,
                          score=score))
    cands.sort(key=lambda d: -d["score"])
    return cands


def under_crossings(pts, n):
    """Top-view crossings where cable n lies under another cable (how trapped it is in the pile)."""
    k = 0
    P = pts[n]
    for m, Q in pts.items():
        if m == n:
            continue
        for ii, jj, s_, t_ in NC._seg_intersections_2d(P, Q):
            za = P[ii, 2] + s_ * (P[ii + 1, 2] - P[ii, 2])
            zb = Q[jj, 2] + t_ * (Q[jj + 1, 2] - Q[jj, 2])
            k += int(za < zb)
    return k


def drag_target(st, col, g, margin=0.07):
    """Point just outside the bin of colour col (margin from its outer walls) nearest to g, at g's side."""
    bq = st.state.body_q.numpy()
    M = NC.tf_mat(bq[st.bins[col]["body"]])
    gl = M[:3, :3].T @ (np.asarray(g) - M[:3, 3])
    a, bh = 0.5 * NC.BIN["size"][0] + margin, 0.5 * NC.BIN["size"][1] + margin
    d = np.array([np.clip(gl[0], -a, a), np.clip(gl[1], -bh, bh), gl[2]])
    if abs(d[0]) < a and abs(d[1]) < bh:  # g is within the margin: no drag
        d = gl
    return (M[:3, :3] @ d + M[:3, 3])[:2]


class Picker:
    """Grasp-and-transfer primitives on top of the Station; joint targets come from MuJoCo IK on Cartesian TCP
    keyframes; every grasp closes under the force servo."""

    def __init__(self, st, run_frames, timing):
        self.st, self.run_frames, self.timing = st, run_frames, timing
        self.pose = {s: (st.kin.tcp(s, st.q)[0], 0.0, 0.0) for s in NR.SIDES}  # TCP pos, yaw, tilt per arm
        self.home_tcp = {s: self.pose[s][0].copy() for s in NR.SIDES}
        self.still_dir = None

    def path(self, side, keys):
        q, err = cart_path(self.st.kin, side, [dict(pos=self.pose[side][0], yaw=self.pose[side][1],
                                                    tilt=self.pose[side][2], t=0.0)] + keys, self.st.q)
        return q, err

    def best_path(self, side, key_options):
        """First key list (of several alternatives) whose whole Cartesian path has IK error < 2 mm."""
        worst = None
        for keys in key_options:
            q, err = self.path(side, keys)
            if err < 2e-3:
                return q, keys, err
            worst = err if worst is None else min(worst, err)
        return None, None, worst

    def timed(self, name, fn):
        st = self.st
        t0, ts = time.time(), st.t
        fn()
        wp.synchronize()
        dw, ds = time.time() - t0, st.t - ts
        prev = self.timing.get(name, dict(sim_s=0.0, wall_s=0.0))
        self.timing[name] = dict(sim_s=prev["sim_s"] + ds, wall_s=prev["wall_s"] + dw)
        self.timing[name]["wall_per_sim_s"] = self.timing[name]["wall_s"] / max(self.timing[name]["sim_s"], 1e-9)

    def transfer(self, tag, cand, release_xy):
        """Approach (fingers pre-closing), descend, close under the force servo, lift + carry (tilted: the YAM Ultra's
        top-down reach ends ~0.15 m above the table near the pile), lower over the bin, open, rise."""
        st, side, n = self.st, cand["side"], cand["cable"]
        g, yaw, zg = np.array(cand["grip"]), cand["yaw"], cand["pad_bottom"]
        z_top = ZT + 0.18
        gt = cand.get("tilt", 0.0)
        a_dir = NR.grasp_R(yaw, gt)[:, 2]
        above = np.array([g[0], g[1], zg]) - 0.09 * a_dir  # descend along the approach axis
        approach = [[dict(pos=tuple(above), yaw=yaw, tilt=max(tl, gt), t=2.2),
                     dict(pos=(g[0], g[1], zg), yaw=yaw, tilt=gt, t=1.2)] for tl in (0.0, 0.15, 0.3)]
        qa, ka, ea = self.best_path(side, approach)
        if qa is None:
            return dict(ok=False, why=f"approach unreachable ({ea:.3g})")
        mid = 0.5 * (g[:2] + np.asarray(release_xy))
        if cand["kind"] == "cable":
            # drag the cable low along the table to just outside the bin wall first (a trapped cable slides out from
            # under the others instead of lifting them), then lift it over the rim
            D = drag_target(st, NC.CABLE_BY_NAME[n]["color"], g)
            carry = [[dict(pos=(g[0], g[1], zg + 0.03), yaw=yaw, tilt=gt, t=0.6),
                      dict(pos=(D[0], D[1], zg + 0.03), yaw=yaw, tilt=gt, t=1.2),
                      dict(pos=(D[0], D[1], z_top), yaw=0.5 * yaw, tilt=tl, t=1.0),
                      dict(pos=(release_xy[0], release_xy[1], z_top), yaw=0.0, tilt=tl, t=1.0),
                      dict(pos=(release_xy[0], release_xy[1], ZT + 0.12), yaw=0.0, tilt=0.0, t=0.8)]
                     for tl in (0.3, 0.45)]
        else:
            carry = [[dict(pos=(g[0], g[1], zg + 0.08), yaw=yaw, tilt=tl, t=1.2),
                      dict(pos=(mid[0], mid[1], z_top), yaw=0.5 * yaw, tilt=tl, t=1.5),
                      dict(pos=(release_xy[0], release_xy[1], z_top), yaw=0.0, tilt=tl, t=1.3),
                      dict(pos=(release_xy[0], release_xy[1], ZT + 0.12), yaw=0.0, tilt=0.0, t=1.0)]
                     for tl in (0.3, 0.45)]
        save = self.pose[side]
        self.pose[side] = (np.array(ka[-1]["pos"]), yaw, gt)
        qc, kc, ec = self.best_path(side, carry)
        self.pose[side] = save
        if qc is None:
            return dict(ok=False, why=f"carry unreachable ({ec:.3g})")
        res = dict(ok=True, kind=cand["kind"], cable=n, side=side, cand={k: v for k, v in cand.items()},
                   ik_err=[ea, ec], release_xy=list(map(float, release_xy)))
        w_open = (cand["width_open"] - NR.PAD_GAP0) / 2
        cab = st.cabs[n]
        flange_b = st.rinfo["bodies"][f"{side}_gripper"]
        slip = dict(ref=None, max_mm=0.0, max_deg=0.0, on=False)
        hold_f = []

        def extra():
            bq = st.state.body_q.numpy()
            F = NC.tf_mat(bq[flange_b])
            if cand["kind"] == "plug":
                rel = np.linalg.inv(F) @ NC.plug_pose(bq, cab)
                rel[:3, 3] = rel[:3, :3] @ plug_grip_geometry(n)[0] + rel[:3, 3]  # track the grip point
            else:
                X, _ = NC.cable_nodes(bq, cab)
                rel = np.eye(4)
                rel[:3, 3] = F[:3, :3].T @ (X[cand["node"]] - F[:3, 3])
            out = dict(task=tag, side=side, cable=n, grasp=cand["kind"])
            if slip["on"]:
                if slip["ref"] is None:
                    slip["ref"] = rel
                d = 1e3 * float(np.linalg.norm(rel[:3, 3] - slip["ref"][:3, 3]))
                Rr = slip["ref"][:3, :3].T @ rel[:3, :3]
                ang = math.degrees(math.acos(float(np.clip((np.trace(Rr) - 1) / 2, -1, 1))))
                slip["max_mm"], slip["max_deg"] = max(slip["max_mm"], d), max(slip["max_deg"], ang)
                out.update(slip_mm=d, slip_deg=ang)
                hold_f.append(list(st.grip[side]))
            return out

        def approach_fn():
            for q in qa:
                st.finger[side] = float(np.clip(w_open, st.finger[side] - NR.FINGER_SPEED / FPS,
                                                st.finger[side] + NR.FINGER_SPEED / FPS))
                self.run_frames(1, f"{tag}_approach", [q], extra_fn=extra)
            self.run_frames(10, f"{tag}_approach", [qa[-1]], extra_fn=extra)

        def close_fn():
            ok = 0
            for i in range(int(2.0 * FPS)):
                self.run_frames(1, f"{tag}_close", [qa[-1]], cmd={side: "close"}, extra_fn=extra)
                f = 0.5 * sum(st.grip[side])
                if st.finger[side] <= 1e-4 and max(st.grip[side]) == 0.0:
                    res["missed"] = True  # fingers closed on nothing: the target moved away
                    break
                ok = ok + 1 if abs(f - GRIP_FORCE) < 0.15 * GRIP_FORCE else 0
                if ok >= int(0.3 * FPS):
                    break
            res["grip_closed_N"] = [float(x) for x in st.grip[side]]

        def carry_fn():
            slip["on"] = True
            lost = 0
            for q in list(qc) + [qc[-1]] * int(0.3 * FPS):
                self.run_frames(1, f"{tag}_carry", [q], cmd={side: "close"}, extra_fn=extra)
                lost = lost + 1 if max(st.grip[side]) == 0.0 else 0
                if lost >= int(0.2 * FPS):  # the object was pulled out of the fingers
                    res["dropped"] = True
                    res["dropped_at_t"] = st.t
                    break
            slip["on"] = False

        def release_fn():
            q_here = st.q.copy()
            self.run_frames(int(0.8 * FPS), f"{tag}_release", [q_here], cmd={side: "open"}, extra_fn=extra)

        self.timed("approach", approach_fn)
        self.timed("grasp", close_fn)
        self.snapshot(f"{tag}_grasp", g)
        if res.get("missed"):  # open, back up, let the caller re-plan from the new state
            self.pose[side] = (np.array(ka[-1]["pos"]), yaw, gt)
            qb, kb, eb = self.best_path(side, [[dict(pos=tuple(above), yaw=yaw, tilt=max(tl, gt), t=1.0)]
                                               for tl in (0.0, 0.15, 0.3)])
            self.timed("release", lambda: self.run_frames(int(0.5 * FPS), f"{tag}_missed", [qa[-1]],
                                                          cmd={side: "open"}, extra_fn=extra))
            if qb is not None:
                self.timed("retreat", lambda: self.run_frames(len(qb), f"{tag}_missed", qb, cmd={side: "open"},
                                                              extra_fn=extra))
                self.pose[side] = (np.array(kb[-1]["pos"]), yaw, kb[-1]["tilt"])
            res.update(ok=False, why="missed: fingers closed on nothing", slip_max_mm=0.0, slip_max_deg=0.0)
            print(f"  {tag}: missed grasp (fingers closed on nothing), backing off", flush=True)
            return res
        self.timed("carry", carry_fn)
        self.timed("release", release_fn)
        if res.get("dropped"):
            p_now, R_now = st.kin.tcp(side, st.q)
            tilt = math.atan2(R_now[1, 2], -R_now[2, 2])  # flange z = Rx(tilt) (0, 0, -1)
            c_, s_ = math.cos(-tilt), math.sin(-tilt)
            x0 = np.array([[1, 0, 0], [0, c_, -s_], [0, s_, c_]]) @ R_now[:, 0]
            self.pose[side] = (p_now, float(math.atan2(x0[1], x0[0])), float(tilt))
            print(f"  {tag}: lost the grasp during the carry (pulled out of the fingers) at t={res['dropped_at_t']:.2f}",
                  flush=True)
        else:
            self.pose[side] = (np.array(kc[-1]["pos"]), 0.0, 0.0)
        # rise clear of the bin before the next move
        qr, kr, er = self.best_path(side, [[dict(pos=(release_xy[0], release_xy[1], ZT + 0.16), yaw=0.0, tilt=tl,
                                                 t=0.7)] for tl in (0.0, 0.15, 0.3)])
        if qr is not None:
            self.timed("release", lambda: self.run_frames(len(qr), f"{tag}_rise", qr, cmd={side: "open"},
                                                          extra_fn=extra))
            self.pose[side] = (np.array(kr[-1]["pos"]), 0.0, kr[-1]["tilt"])
        hf = np.array(hold_f) if hold_f else np.zeros((1, 2))
        res.update(slip_max_mm=slip["max_mm"], slip_max_deg=slip["max_deg"],
                   hold_force_median=np.median(hf, 0).tolist(), hold_force_max=float(hf.max()),
                   hold_force_min=float(hf.min()))
        return res

    def snapshot(self, name, g):
        """Close-up still of a grasp (out/.../stills/), then back to the video camera."""
        st = self.st
        if st.viewer is None or self.still_dir is None:
            return
        g = np.asarray(g, float)
        CAMS["_snap"] = dict(pos=tuple(g + np.array([0.09, 0.13, 0.07])), lookat=tuple(g + np.array([0.0, 0.0, 0.012])),
                             fov=32)
        st.set_cam("_snap")
        write_png(os.path.join(self.still_dir, f"{name}.png"), st.render())
        st.set_cam("video")

    def go_home(self, tag, side):
        st = self.st
        q, k, e = self.best_path(side, [[dict(pos=self.home_tcp[side], yaw=0.0, tilt=0.0, t=2.0)],
                                        [dict(pos=self.home_tcp[side] + np.array([0, 0, -0.05]), yaw=0.0, tilt=0.2,
                                              t=1.2), dict(pos=self.home_tcp[side], yaw=0.0, tilt=0.0, t=1.0)]])
        if q is None:
            return False
        self.timed("retreat", lambda: self.run_frames(len(q), f"{tag}_home", q, cmd={side: "open"}))
        self.pose[side] = (self.home_tcp[side].copy(), 0.0, 0.0)
        return True


def sort_cable(picker, k, cand, max_feeds=5, target=0.75):
    """Pick a cable by its plug, drop the plug into its bin (far half, away from the pile), then feed the part still
    outside the bin with cable-body grasps until >= target of it is inside (or max_feeds)."""
    st = picker.st
    n, side = cand["cable"], cand["side"]
    col = NC.CABLE_BY_NAME[n]["color"]
    bx, by = NC.BINS[col]["xy"]
    steps, skipped = [], []
    print(f"sort {k}: {n} ({col}) with the {side} arm", flush=True)
    r = picker.transfer(f"sort{k}_plug", cand, (bx + np.sign(bx) * 0.045, by + 0.02))
    if not r["ok"]:
        print(f"  plug grasp failed ({r['why']})", flush=True)
        return None
    steps.append(r)
    print(f"  plug transfer: ok={r['ok']} {r.get('why', '')} slip {r.get('slip_max_mm', 0):.2f} mm "
          f"hold force median {r.get('hold_force_median')}", flush=True)
    for f in range(max_feeds if r["ok"] else 0):
        bq, _, pts, _ = st.read()
        fr = success(st, bq, pts)["per_cable"][n]["fractions"][col]
        print(f"  inside {fr:.2f}", flush=True)
        if fr >= target:
            break
        fc = feed_candidates(st, bq, pts, n, side)
        r = None
        for cnd in fc[:6]:  # the first whose approach and carry paths are reachable (planning only, nothing moves)
            away = np.array([bx, by]) - np.asarray(cnd["grip"][:2])
            rel = np.array([bx, by]) + 0.05 * away / max(np.linalg.norm(away), 1e-9)  # bin's far half from the grasp
            r = picker.transfer(f"sort{k}_feed{f}", cnd, rel)
            if r["ok"] or r.get("missed"):
                break
            skipped.append(dict(feed=f, node=cnd["node"], why=r["why"]))
        if r is None or not (r["ok"] or r.get("missed")):
            print("  no reachable feed grasp", flush=True)
            break
        steps.append(r)
        print(f"  feed {f}: node {r['cand']['node']} of run {r['cand']['run']} (under {r['cand']['run_under']})"
              f"{' MISSED' if r.get('missed') else ''}",
              flush=True)
    picker.go_home(f"sort{k}", side)
    bq, _, pts, _ = st.read()
    res = dict(cable=n, side=side, color=col, steps=steps, skipped_unreachable=skipped,
               success=success(st, bq, pts)["per_cable"][n])
    print(f"  result: {res['success']}", flush=True)
    return res


def status_note(args, st, timing, settle, smi):
    """Interim note for the orchestrator (out/cables/newton/STATUS.md)."""
    if os.path.abspath(args.out) != os.path.abspath(OUT):
        return
    d = timing["drop_settle"]
    with open(os.path.join(OUT, "STATUS.md"), "w") as f:
        f.write(f"# cables / newton: status ({time.strftime('%Y-%m-%d %H:%M')})\n\n"
                f"8-cable pile drop + settle ({'with' if st.has_robot else 'without'} the robot in the scene): "
                f"{d['sim_s']:.1f} s simulated in {d['wall_s']:.1f} s wall = **{d['wall_per_sim_s']:.2f} s wall per "
                f"simulated s** (SolverVBD 60 fps x {args.sub} substeps x {args.iters} iterations, "
                f"{st.model.body_count} bodies; GPU shared: {smi}).\n\n"
                f"Settled: KE {settle['ke']:.2e} J, vmax {settle['vmax']:.4f} m/s, crossings "
                f"{settle['tangle']['crossings']}, interlocked pairs {len(settle['tangle']['interlocked_pairs'])}.\n"
                f"Rest-shape retention: { {n: round(v['retention'], 3) for n, v in settle['retention'].items()} }\n\n"
                f"Genesis (stations/cables/NOTES.md, run.log): ~100 s wall per simulated s for drop+settle, 600-800 s/s "
                f"while grasping.\n")


if __name__ == "__main__":
    main()
