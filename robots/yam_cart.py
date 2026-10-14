"""Wheeled bimanual I2RT YAM cart for the YAM arcade stations (see robots/README.md).

    import sys; sys.path.insert(0, os.path.expanduser("~/projects/yam-arcade")); sys.path.insert(0, ".../robots")
    from yam_cart import add_cart
    cart = add_cart(scene, gs, "yam_pro", ipc=False)       # before scene.build()
    scene.build()
    cart.after_build()                                     # gains, force limits, home pose
    q = cart.ik("left", pos=(0.1, 0.0, 0.80), yaw=0.0)     # top-down TCP target -> 6 arm joint angles
    for qk in cart.joint_path("left", q, duration=1.5, dt=0.01):
        cart.set_arm_target("left", qk); cart.update(); scene.step()
    cart.grasp("left", force=10.0)                         # speed-limited close, then force-capped squeeze
    ...
    rec = Recorder(...); cart.register(rec); each frame: rec.frame(t, rigid={**cart.poses(), **mine})

Station-local frame: origin on the floor under the table-top centre, +z up; the cart stands at the -y edge facing +y.
The robot's left arm is at -x, right arm at +x.
"""
import json
import math
import os
import sys

import numpy as np
from genesis.utils.misc import tensor_to_array

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import layout  # noqa: E402

ASSETS = os.path.join(ROOT, "assets", "robots")
VARIANTS = ("yam", "yam_pro", "yam_ultra", "big_yam")
SIDES = ("left", "right")
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
FINGER_JOINTS = ("joint7", "joint8")
BASE_JOINTS = ("base_x", "base_y", "base_yaw")

# linear_4310 grasp_site (I2RT gripper MJCF) in the arm's `<side>_gripper` (flange) link frame: 144.65 mm along the
# flange +z (approach axis), at the midpoint of the two fingertip end facets.
TCP_OFFSET = np.array([3.8898e-05, 9.681e-05, 0.14465])
FINGER_Q_MAX = 0.0475  # each finger travels 0 (closed) .. 47.5 mm (open), along the flange x axis
# collision-pad gap between the two pads = 2 q + PAD_GAP0. The fingertip meshes overlap by 1 mm at q = 0; each box pad
# is recessed 1 mm behind its mesh face so the pads never touch (libuipc rejects touching link geometry at init).
# See robots/measure_gripper.py. A grasped object therefore shows <= 1 mm of fingertip-mesh overlap.
PAD_GAP0 = 0.001
MAX_WIDTH = 2 * FINGER_Q_MAX + PAD_GAP0

# Peak motor torques (i2rt MotorConstants.TORQUE_MAX) per arm joint, from robots/config/<arm>_v1.yml motor lists.
_T = {"DM4310": 10.0, "DM4340": 28.0, "DM6248": 120.0}
ARM_TORQUE = {
    "yam": [_T["DM4340"]] * 3 + [_T["DM4310"]] * 3,
    "yam_pro": [_T["DM4340"]] * 3 + [_T["DM4310"]] * 3,
    "yam_ultra": [_T["DM4340"]] * 3 + [_T["DM4310"]] * 3,
    "big_yam": [_T["DM6248"]] * 2 + [_T["DM4340"]] * 2 + [_T["DM4310"]] * 2,
}
ARM_KP = np.array([600.0, 600.0, 500.0, 150.0, 120.0, 60.0])
ARM_KV = np.array([40.0, 40.0, 30.0, 8.0, 6.0, 3.0])
FINGER_KP, FINGER_KV = 2000.0, 20.0  # finger mass 0.07 kg + 0.5 kg armature, plus 50 N s/m joint damping (MJCF)
OPEN_FORCE = 20.0  # N per finger when opening / holding a set width
BASE_KP, BASE_KV = (4.0e4, 4.0e4, 2.0e4), (4.0e3, 4.0e3, 2.0e3)

GRASP_TILTS = (0.0, 0.25, 0.5, 0.7, 0.9)  # rad, tried in order by Cart.ik_grasp

# Ready pose: both TCPs 20 cm above the table, 13 cm in from the near edge, pointing down, fingers open.
HOME_TCP = dict(left=(-0.26, -0.22, layout.TABLE_TOP_Z + 0.20), right=(0.26, -0.22, layout.TABLE_TOP_Z + 0.20))

# MJCF rgba (set by build_models.PALETTE) -> compositor material
MATERIALS = {
    "anodised_black": dict(kind="metal", color=(0.045, 0.045, 0.05), roughness=0.45),
    "graphite": dict(kind="metal", color=(0.20, 0.20, 0.22), roughness=0.4),
    "silver": dict(kind="metal", color=(0.70, 0.71, 0.73), roughness=0.3),
    "white_shell": dict(kind="plastic", color=(0.90, 0.90, 0.88), roughness=0.5),
    "top_plate": dict(kind="plastic", color=(0.16, 0.16, 0.18), roughness=0.6),
    "rubber": dict(kind="rubber", color=(0.03, 0.03, 0.03), roughness=0.8),
    "aluminium": dict(kind="metal", color=(0.78, 0.79, 0.80), roughness=0.35),
    "estop_red": dict(kind="plastic", color=(0.80, 0.08, 0.06), roughness=0.4),
    "camera": dict(kind="plastic", color=(0.12, 0.12, 0.13), roughness=0.5),
}


def model_path(variant):
    return os.path.join(ASSETS, f"yam_cart_{variant}.xml")


def quat_from_mat(R):
    """3x3 rotation -> (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        q = [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def mat_from_quat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def down_quat(yaw=0.0, tilt=0.0):
    """Flange orientation for a top-down grasp: approach (flange +z) along world -z, fingers closing along
    (cos yaw, sin yaw, 0). `tilt` (rad) then pitches the approach axis towards +y (away from the cart, about world x),
    which extends the reach to far targets."""
    xa = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    za = np.array([0.0, 0.0, -1.0])
    R = np.stack([xa, np.cross(za, xa), za], axis=1)
    c, s = math.cos(tilt), math.sin(tilt)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])  # about world x: approach (0,0,-1) -> (0, sin, -cos)
    return quat_from_mat(Rx @ R)


def min_jerk(n):
    s = np.linspace(0.0, 1.0, n + 1)[1:]
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class Cart:
    """One wheeled bimanual YAM. Create with add_cart(); call after_build() once after scene.build()."""

    def __init__(self, scene, gs, variant, gripper="linear_4310", ipc=False, pos=None, yaw_deg=0.0,
                 coup_links=None, coup_friction=1.0, name="robot", surface=None, gravity_compensation=1.0):
        assert variant in VARIANTS, variant
        assert gripper == "linear_4310", "only the linear_4310 gripper is modelled"
        self.scene, self.gs, self.variant, self.ipc, self.name = scene, gs, variant, ipc, name
        m = layout.ROBOT_MOUNT
        self.base_pos = np.array(pos if pos is not None else (0.0, m["cart_center_y"], 0.0), dtype=float)
        if ipc:
            patch_ipc_only_coupling_forces()
            self.coup_link_names = tuple(coup_links) if coup_links is not None else self.default_coup_links()
            material = gs.materials.Rigid(coup_type="two_way_soft_constraint", coup_links=self.coup_link_names,
                                          coup_friction=coup_friction,
                                          gravity_compensation=gravity_compensation)
        else:
            self.coup_link_names = ()
            material = gs.materials.Rigid(gravity_compensation=gravity_compensation)
        kw = dict(file=model_path(variant), pos=tuple(self.base_pos), euler=(0.0, 0.0, yaw_deg))
        self.entity = scene.add_entity(gs.morphs.MJCF(**kw), material=material,
                                       **({} if surface is None else dict(surface=surface)))
        self._built = False

    # ------------------------------------------------------------------ structure
    @staticmethod
    def default_coup_links():
        """Links that libuipc sees in IPC scenes: the four fingertips (box pads) and the two palms (box)."""
        return tuple(f"{s}_{n}" for s in SIDES for n in ("tip_left", "tip_right", "gripper"))

    def coup_links(self):
        return self.coup_link_names

    def _dofs(self, joint_names):
        return np.array([self.entity.get_joint(j).dofs_idx_local[0] for j in joint_names])

    def after_build(self, home=True):
        e = self.entity
        self.base_dofs = self._dofs(BASE_JOINTS)
        self.arm_dofs = {s: self._dofs([f"{s}_{j}" for j in ARM_JOINTS]) for s in SIDES}
        self.finger_dofs = {s: self._dofs([f"{s}_{j}" for j in FINGER_JOINTS]) for s in SIDES}
        self.ee_link = {s: e.get_link(f"{s}_gripper") for s in SIDES}
        self.tcp_offset = TCP_OFFSET.copy()
        kp, kv = np.zeros(e.n_dofs), np.zeros(e.n_dofs)
        lo, hi = np.full(e.n_dofs, -1e4), np.full(e.n_dofs, 1e4)
        kp[self.base_dofs], kv[self.base_dofs] = BASE_KP, BASE_KV
        for s in SIDES:
            kp[self.arm_dofs[s]], kv[self.arm_dofs[s]] = ARM_KP, ARM_KV
            lo[self.arm_dofs[s]], hi[self.arm_dofs[s]] = -np.array(ARM_TORQUE[self.variant]), ARM_TORQUE[self.variant]
            kp[self.finger_dofs[s]], kv[self.finger_dofs[s]] = FINGER_KP, FINGER_KV
            lo[self.finger_dofs[s]], hi[self.finger_dofs[s]] = -OPEN_FORCE, OPEN_FORCE
        e.set_dofs_kp(kp)
        e.set_dofs_kv(kv)
        e.set_dofs_force_range(lo, hi)
        self._built = True
        self.q_target = np.zeros(e.n_dofs)
        self._grip = {s: dict(mode="hold", target=FINGER_Q_MAX, force=OPEN_FORCE, speed=0.05, stalled=False)
                      for s in SIDES}
        self.home_qpos = self.compute_home()
        if home:
            self.set_qpos(self.home_qpos)

    def compute_home(self):
        q = np.zeros(self.entity.n_dofs)
        q[self.base_dofs] = 0.0
        seed = {"left": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0], "right": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]}
        for s in SIDES:
            q[self.arm_dofs[s]] = seed[s]
            q[self.finger_dofs[s]] = FINGER_Q_MAX
        self.entity.set_qpos(q)
        for s in SIDES:
            qa, err = self.ik(s, HOME_TCP[s], yaw=0.0, init_qpos=q, return_error=True)
            q[self.arm_dofs[s]] = qa
        return q

    # ------------------------------------------------------------------ state
    def set_qpos(self, q):
        """Teleport the whole robot (all dofs) and make it the PD target. Use only before the run starts."""
        q = np.asarray(q, dtype=float)
        self.entity.set_qpos(q)
        self.entity.zero_all_dofs_velocity()
        self.q_target = q.copy()
        self.entity.control_dofs_position(self.q_target)
        for s in SIDES:
            self._grip[s].update(mode="hold", target=float(q[self.finger_dofs[s]].mean()), force=OPEN_FORCE)

    def qpos(self):
        return self._a(self.entity.get_qpos())

    def arm_q(self, side):
        return self.qpos()[self.arm_dofs[side]]

    def finger_q(self, side):
        return self.qpos()[self.finger_dofs[side]]

    def gripper_width(self, side):
        """Current gap between the two collision pads, m."""
        return float(self.finger_q(side).sum() + PAD_GAP0)

    def tcp_pose(self, side):
        """World TCP position and flange quaternion (w, x, y, z)."""
        link = self.ee_link[side]
        p, q = self._a(link.get_pos()), self._a(link.get_quat())
        return p + mat_from_quat(q) @ self.tcp_offset, q

    @staticmethod
    def _a(x):
        return tensor_to_array(x)

    # ------------------------------------------------------------------ kinematics
    def ik(self, side, pos, quat=None, yaw=0.0, tilt=0.0, init_qpos=None, return_error=False, max_samples=30):
        """6 arm joint angles putting the TCP at `pos` (world) with flange orientation `quat` (default: top-down,
        fingers closing along (cos yaw, sin yaw, 0)). Only this arm's dofs move. init_qpos: full-robot seed
        (default: current targets)."""
        quat = down_quat(yaw, tilt) if quat is None else np.asarray(quat, dtype=float)
        init = self.q_target if init_qpos is None else np.asarray(init_qpos, dtype=float)
        q, err = self.entity.inverse_kinematics(
            link=self.ee_link[side], pos=np.asarray(pos, dtype=float), quat=quat, local_point=self.tcp_offset,
            init_qpos=init, dofs_idx_local=self.arm_dofs[side], return_error=True, max_samples=max_samples)
        qa = self._a(q)[self.arm_dofs[side]]
        if return_error:
            return qa, self._a(err)
        return qa

    def ik_grasp(self, side, pos, yaw=0.0, tilts=GRASP_TILTS, init_qpos=None, pos_tol=0.003, rot_tol=0.03):
        """Grasp IK that degrades gracefully with distance: tries a pure top-down approach first, then approach axes
        pitched progressively towards +y (see down_quat). Returns (q_arm, tilt) for the first converged solution,
        or (None, None) if `pos` is out of reach."""
        for tilt in tilts:
            q, err = self.ik(side, pos, yaw=yaw, tilt=tilt, init_qpos=init_qpos, return_error=True)
            if np.linalg.norm(err[:3]) < pos_tol and np.linalg.norm(err[3:]) < rot_tol:
                return q, tilt
        return None, None

    def fk_tcp(self, side, q_arm):
        """TCP world position for arm joints q_arm (restores the current state afterwards)."""
        q0 = self.qpos()
        q = q0.copy()
        q[self.arm_dofs[side]] = q_arm
        self.entity.set_qpos(q)
        p, quat = self.tcp_pose(side)
        self.entity.set_qpos(q0)
        return p, quat

    # ------------------------------------------------------------------ control
    def set_arm_target(self, side, q_arm):
        self.q_target[self.arm_dofs[side]] = q_arm
        self.entity.control_dofs_position(np.asarray(q_arm, dtype=float), self.arm_dofs[side])

    def set_base_target(self, x=0.0, y=0.0, yaw=0.0):
        """Base offset from its parked pose (station frame metres, rad); held by stiff PD."""
        self.q_target[self.base_dofs] = (x, y, yaw)
        self.entity.control_dofs_position(np.array([x, y, yaw], dtype=float), self.base_dofs)

    def joint_path(self, side, q_goal, duration, dt, q_start=None):
        """Minimum-jerk joint-space waypoints (one per sim step) from the current arm target to q_goal."""
        q0 = self.q_target[self.arm_dofs[side]].copy() if q_start is None else np.asarray(q_start, dtype=float)
        n = max(1, int(round(duration / dt)))
        return [q0 + s * (np.asarray(q_goal) - q0) for s in min_jerk(n)]

    def both_paths(self, q_goals, duration, dt):
        """Min-jerk paths for several arms at once: {side: q_goal} -> list of {side: q}."""
        paths = {s: self.joint_path(s, q, duration, dt) for s, q in q_goals.items()}
        return [{s: paths[s][k] for s in paths} for k in range(len(next(iter(paths.values()))))]

    def set_gripper(self, side, width, force=OPEN_FORCE, speed=0.05):
        """Move the fingers to a pad gap `width` (m, 0 .. MAX_WIDTH) at `speed` m/s per finger, force cap `force` N."""
        q = float(np.clip((width - PAD_GAP0) / 2, 0.0, FINGER_Q_MAX))
        self._grip[side].update(mode="width", goal=q, force=force, speed=speed, stalled=False)

    def open(self, side, speed=0.05):
        self.set_gripper(side, MAX_WIDTH, speed=speed)

    def grasp(self, side, force=10.0, speed=0.05, squeeze_q=-0.02):
        """Close at `speed` until the fingers stall on the object (3 mm behind their target), then command a target
        past closed with a per-finger force cap `force` N (the Franka lesson: an unthrottled capped actuator slams
        the light fingers shut)."""
        g = self._grip[side]
        g.update(mode="grasp", force=force, speed=speed, stalled=False, squeeze_q=squeeze_q,
                 target=float(self.finger_q(side).mean()))

    def grasped(self, side):
        return self._grip[side]["stalled"]

    def finger_force(self, side):
        return self._a(self.entity.get_dofs_control_force(self.finger_dofs[side]))

    def update(self, dt=None):
        """Advance the gripper targets by one sim step. Call once per step, before scene.step()."""
        dt = self.scene.dt if dt is None else dt
        q = self.qpos()
        for s in SIDES:
            g = self._grip[s]
            qf = float(q[self.finger_dofs[s]].mean())
            if g["mode"] == "width":
                step = g["speed"] * dt
                g["target"] = float(np.clip(g["goal"], g["target"] - step, g["target"] + step))
                lo, hi = -g["force"], g["force"]
            elif g["mode"] == "grasp":
                if not g["stalled"] and qf - g["target"] > 0.003 and g["target"] > 0.0005:
                    g["stalled"] = True
                if g["stalled"]:
                    g["target"] = g["squeeze_q"]
                else:
                    g["target"] = max(-0.001, g["target"] - g["speed"] * dt)
                    if g["target"] <= -0.0009:  # closed on nothing
                        g["target"] = g["squeeze_q"]
                lo, hi = -g["force"], OPEN_FORCE
            else:
                lo, hi = -g["force"], g["force"]
            self.q_target[self.finger_dofs[s]] = g["target"]
            self.entity.control_dofs_position(np.full(2, g["target"]), self.finger_dofs[s])
            self.entity.set_dofs_force_range(np.full(2, lo), np.full(2, hi), self.finger_dofs[s])

    # ------------------------------------------------------------------ export
    def export_parts(self):
        """[(object_name, link_name, mesh_path, material)] — one OBJ per (link, material), in the link frame."""
        return export_parts(self.variant, self.name)

    def register(self, rec):
        """Add every robot link (per material) to an export.Recorder as a rigid named robot/<link>[/<material>]."""
        self._parts = self.export_parts()
        for obj, link, mesh, mat in self._parts:
            rec.add_rigid(obj, mesh, material=mat)

    def poses(self):
        """{object_name: (pos, quat_wxyz)} for every registered part, for Recorder.frame(rigid=...)."""
        out = {}
        cache = {}
        for obj, link, _, _ in self._parts:
            if link not in cache:
                l = self.entity.get_link(link)
                cache[link] = (self._a(l.get_pos()), self._a(l.get_quat()))
            out[obj] = cache[link]
        return out

    def record(self, rec, t, rigid=None, **kw):
        """rec.frame(t, rigid={robot parts + rigid}, **kw)."""
        rec.frame(t, rigid={**self.poses(), **(rigid or {})}, **kw)


_PATCHED = []


def patch_ipc_only_coupling_forces():
    """Genesis 1.3.1 (ipc_rod_and_mpm) bug workaround, from ~/projects/franka-deformables/genesis_cable.py:
    IPCCoupler._apply_abd_coupling_forces applies the two-way soft-constraint reaction to EVERY affine-body link,
    including free `ipc_only` props, which have no soft constraint. Next to a two_way_soft_constraint robot a free
    ipc_only cube's Genesis copy got a spurious force every step and flew to 1e30 m / NaN within ~2 s (seen in
    robots/test_grasp.py --ipc). This re-implements the method with the forces on non-robot links zeroed; libuipc
    sets those links' poses directly. Applied automatically by add_cart(ipc=True); idempotent."""
    if _PATCHED:
        return
    from genesis.engine.couplers.ipc_coupler import coupler as C

    masks = {}

    def _apply(self):
        if (not self.options.two_way_coupling
                or C.COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT not in self._entities_by_coup_type
                or not self._abd_data_by_link):
            return
        d = self._coupling_data
        C.update_coupling_forces(d.ipc_transforms, d.aim_transforms, d.links_mass, d.links_inertia_i,
                                 self._constraint_strength_translation_scaled,
                                 self._constraint_strength_rotation_scaled, d.out_forces, d.out_torques)
        if id(self) not in masks:
            soft = set(self._entities_by_coup_type[C.COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT])
            masks[id(self)] = np.array([link.entity in soft for link in d.links])
        m = masks[id(self)]
        d.out_forces[:, ~m] = 0.0
        d.out_torques[:, ~m] = 0.0
        if np.isnan(d.out_forces).any() or np.isnan(d.out_torques).any():
            raise RuntimeError("NaN IPC coupling forces")
        f, tq = (d.out_forces, d.out_torques) if self.sim.n_envs > 0 else (d.out_forces[0], d.out_torques[0])
        self.rigid_solver.apply_links_external_force(f, links_idx=d.links_idx, local=False)
        self.rigid_solver.apply_links_external_torque(tq, links_idx=d.links_idx, local=False)

    C.IPCCoupler._apply_abd_coupling_forces = _apply
    _PATCHED.append(True)


def add_cart(scene, gs, variant, gripper="linear_4310", ipc=False, **kw):
    """Add a wheeled bimanual YAM cart of `variant` (yam | yam_pro | yam_ultra | big_yam) to `scene`, parked per
    layout.ROBOT_MOUNT. ipc=True couples the fingertips + palms to libuipc (two_way_soft_constraint). Call
    cart.after_build() after scene.build()."""
    return Cart(scene, gs, variant, gripper=gripper, ipc=ipc, **kw)


# ---------------------------------------------------------------------- export meshes (MuJoCo-side, no Genesis)
def export_parts(variant, name="robot"):
    import mujoco
    import trimesh

    out_dir = os.path.join(ASSETS, "export", variant)
    index = os.path.join(out_dir, "index.json")
    src = model_path(variant)
    if os.path.exists(index) and os.path.getmtime(index) >= os.path.getmtime(src):
        parts = json.load(open(index))
    else:
        os.makedirs(out_dir, exist_ok=True)
        m = mujoco.MjModel.from_xml_path(src)
        names = list(MATERIALS)
        cols = np.array([MATERIALS[k]["color"] for k in names])
        groups = {}
        for gi in range(m.ngeom):
            if m.geom_group[gi] > 2 or m.geom_rgba[gi, 3] < 0.99:
                continue
            body = m.body(m.geom_bodyid[gi]).name
            dist = np.abs(cols - m.geom_rgba[gi, :3]).max(1)
            assert dist.min() < 2e-3, f"geom {m.geom(gi).name}: rgba {m.geom_rgba[gi]} is not a MATERIALS colour"
            mat = names[int(dist.argmin())]
            t = m.geom_type[gi]
            if t == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[gi]
                V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].copy()
                F = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]].copy()
                tm = trimesh.Trimesh(V, F, process=False)
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                tm = trimesh.creation.box(extents=2 * m.geom_size[gi])
            elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                tm = trimesh.creation.cylinder(radius=m.geom_size[gi, 0], height=2 * m.geom_size[gi, 1], sections=48)
            else:
                raise ValueError(f"geom type {t}")
            T = np.eye(4)
            T[:3, :3] = mat_from_quat(m.geom_quat[gi])
            T[:3, 3] = m.geom_pos[gi]
            tm.apply_transform(T)
            groups.setdefault((body, mat), []).append(tm)
        bodies = {}
        for (body, mat) in groups:
            bodies.setdefault(body, []).append(mat)
        parts = []
        for (body, mat), tms in sorted(groups.items()):
            mesh = trimesh.util.concatenate(tms)
            fn = os.path.join(out_dir, f"yamcart_{variant}__{body}__{mat}.obj")
            mesh.export(fn)
            parts.append(dict(link=body, material=mat, mesh=fn, multi=len(bodies[body]) > 1))
        json.dump(parts, open(index, "w"), indent=1)
    return [(f"{name}/{p['link']}" + (f"/{p['material']}" if p["multi"] else ""), p["link"], p["mesh"],
             dict(MATERIALS[p["material"]])) for p in parts]
