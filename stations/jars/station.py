"""Jars station: three glass jars and three loose screw lids on the table; screw each lid onto the jar it fits.

Physics: every jar, lid and the table top is a libuipc affine body through Genesis's IPC coupler (exact mesh
contact, which is what lets real helical threads engage). Jars and lids are FREE bodies (`coup_type="ipc_only"`),
the table top is a fixed body at exactly z = layout.TABLE_TOP_Z with the layout extents.

    import genesis as gs
    gs.init(backend=gs.gpu)
    env = build(gs, kind="puzzle")          # or kind="twist" (each lid hovering over its own jar's mouth)
    for _ in range(n):
        env.drive.set_twist("jam", omega)    # optional disclosed feasibility drive (torque about the jar axis)
        env.step()                           # scene.step() + drive controller + engagement tracking
    env.success()

The twist drive (feasibility test only, disclosed in NOTES.md): a torque about the jar axis applied to a lid through
libuipc's AffineBodyExternalBodyForce, from a torque-limited angle servo (a 'hand with a slipping clutch'). It never
sets a position or velocity; the threads alone decide how far the lid descends per turn.
"""
import json
import math
import os
import sys
from dataclasses import dataclass, field
from functools import partial

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import layout  # noqa: E402

ASSETS = os.path.join(HERE, "assets")
PARAMS = json.load(open(os.path.join(ASSETS, "params.json")))
NAMES = ("spice", "jam", "mason")
TABLE_Z = layout.TABLE_TOP_Z

# ---------------------------------------------------------------- physical parameters (justified in NOTES.md)
DT = 1.0 / 300.0  # s; 10 IPC steps per 30 fps frame
D_HAT = 2.0e-4  # m; < every designed thread gap (0.346 mm flank, 0.5 mm radial)
EPS_VELOCITY = 1.0e-4  # m/s; friction smoothing, sets static-friction creep (default 1e-2 lets lids creep down)
NEWTON_TOL = 2.0e-4  # m/s; libuipc default 5e-2 (0.17 mm/step slop) and even 5e-3 let a cocked lid creep 95 um/s
FRICTION = dict(glass=0.45, metal=0.45, plastic=0.30, table=0.50)  # per-entity; pairs = geometric mean
LIFT = 6.0e-4  # m; initial clearance of every resting part above what it rests on (> d_hat)
GLASS_RGBA = (0.78, 0.88, 0.90, 0.28)

# puzzle layout, station frame. Jars in a row 6 cm behind the table centre line; lids loose in front of them (closer to
# the robot), shuffled so none lies next to / in front of its own jar. Why this way round: robots/README.md + our reach maps
# (scratch/reach*.py): the YAM Pro reaches top-down at lid/twist heights only for y <~ -0.05 next to the cart centre,
# and a picking arm must not sweep over the jars. The wheeled base drives along x to centre each jar.
JAR_XY = {"jam": (-0.17, -0.06), "spice": (0.0, -0.06), "mason": (0.17, -0.06)}
# lids sit between the jars' x positions so the lane straight in front of each jar stays free for the holding hand
LID_XY = {"spice": ((-0.255, -0.21), "up"), "mason": ((-0.085, -0.21), "up"), "jam": ((0.085, -0.21), "up")}


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    """3x3 rotation -> (w, x, y, z), w >= 0."""
    t = np.trace(R)
    if t > 0:
        k = 0.5 / math.sqrt(t + 1.0)
        q = np.array([0.25 / k, (R[2, 1] - R[1, 2]) * k, (R[0, 2] - R[2, 0]) * k, (R[1, 0] - R[0, 1]) * k])
    else:
        i = int(np.argmax(np.diag(R)))
        j, l = (i + 1) % 3, (i + 2) % 3
        k = 2.0 * math.sqrt(1.0 + R[i, i] - R[j, j] - R[l, l])
        q = np.zeros(4)
        q[0] = (R[l, j] - R[j, l]) / k
        q[1 + i] = 0.25 * k
        q[1 + j] = (R[j, i] + R[i, j]) / k
        q[1 + l] = (R[l, i] + R[i, l]) / k
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def yaw_quat(psi, flip=False):
    """Rotation about +z by psi, optionally preceded by a half turn about x (upside down)."""
    c, s = math.cos(psi / 2), math.sin(psi / 2)
    q = np.array([c, 0.0, 0.0, s])
    if flip:
        qx = np.array([0.0, 1.0, 0.0, 0.0])
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = qx
        q = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
    return q


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def part(name):
    return PARAMS["parts"][name]


def lid_bottom_offset(name, upside_down):
    """Height of the lid-frame origin above the surface the lid rests on."""
    sp = part(name)["spec"]
    return sp["t_top"] * 1e-3 if upside_down else sp["l_skirt"] * 1e-3


# ---------------------------------------------------------------- disclosed twist drive
class LidTorqueDrive:
    """Torque about each lid's jar axis, applied in libuipc as a 12-D affine generalized force.

    For an affine body x = p + A xbar, a world torque tau is the generalized force F_A = 1/2 [tau]x A on the A block
    (row-major, as libuipc's AffineBodyExternalBodyForce expects): it does work tau.omega on rotations and none on
    the 9 - 3 strain modes, and no net force. libuipc turns it into an acceleration M^-1 F like gravity.

    The twist component (about the jar axis) comes from a torque-limited angle servo on the lid's rotation RELATIVE
    to its jar:  tau_z = clip(kp (psi_ref - psi) + kd (omega_ref - omega), -tau_max, tau_max).
    psi_ref advances at the commanded rate and is held within tau_max / kp of psi (a slipping clutch), so a seated lid
    is pressed with at most tau_max and nothing winds up.
    The two other components are a levelling couple, as the fingers holding a lid supply: a capped PD on the angle
    between the lid axis and the jar axis. It is a pure torque (no force), so it can neither lift nor push the lid:
    gravity and the thread flanks alone decide how far it descends per turn.
    """

    def __init__(self, env, level=False):
        self.env = env
        self.level = level
        self.tau = {n: np.zeros(3) for n in env.lids}
        self.omega_cmd = {n: 0.0 for n in env.lids}
        self.active = {n: False for n in env.lids}
        self.psi_ref = {n: None for n in env.lids}
        self.tau_max = {}
        self.kp, self.kd = {}, {}
        self.kp_lev, self.kd_lev, self.lev_max = {}, {}, {}
        for n in env.lids:
            sp = part(n)["spec"]
            m = part(n)["lid_mass_kg"]
            r = (part(n)["lid_r_outer_mm"]) * 1e-3
            inertia = 0.75 * m * r * r  # skirt ~ m r^2, top panel ~ m r^2 / 2
            wn = 2 * np.pi * 4.0
            self.kp[n] = inertia * wn * wn
            self.kd[n] = 2.0 * inertia * wn
            self.tau_max[n] = dict(spice=0.004, jam=0.010, mason=0.020)[n]
            # levelling: same bandwidth on the tilt inertia (~0.4 of the spin inertia), capped at 3 m g r
            self.kp_lev[n] = 0.4 * inertia * wn * wn
            self.kd_lev[n] = 2.0 * 0.4 * inertia * wn
            self.lev_max[n] = 3.0 * m * 9.81 * r
        self.jar_of = {n: n for n in env.lids}  # which jar's axis each lid is driven about
        self.R_prev = {n: None for n in env.lids}

    def set_twist(self, lid, omega, jar=None, tau_max=None):
        """omega in rad/s about the jar's +z (negative = clockwise from above = tighten a right-hand thread);
        omega=None releases the lid (zero torque)."""
        if jar is not None:
            self.jar_of[lid] = jar
        if tau_max is not None:
            self.tau_max[lid] = tau_max
        if omega is None:
            self.active[lid] = False
            self.psi_ref[lid] = None
            self.tau[lid][:] = 0.0
            return
        self.active[lid] = True
        self.omega_cmd[lid] = omega

    def control(self):
        """Called once per step before scene.step(): compute every lid's torque from the current state."""
        dt = self.env.dt
        poses = self.env.poses()
        for n in self.env.lids:
            Rl = poses[f"lid_{n}"][1]
            Rp, self.R_prev[n] = self.R_prev[n], Rl
            if not self.active[n]:
                self.tau[n][:] = 0.0
                continue
            j = self.jar_of[n]
            psi, omega = self.env.rel_yaw[(n, j)], self.env.rel_omega[(n, j)]
            if self.psi_ref[n] is None:
                self.psi_ref[n] = psi
            lim = self.tau_max[n] / self.kp[n]
            self.psi_ref[n] = float(np.clip(self.psi_ref[n] + self.omega_cmd[n] * dt, psi - lim, psi + lim))
            tz = self.kp[n] * (self.psi_ref[n] - psi) + self.kd[n] * (self.omega_cmd[n] - omega)
            tz = float(np.clip(tz, -self.tau_max[n], self.tau_max[n]))
            axis = poses[f"jar_{j}"][1][:, 2]
            # levelling couple about the horizontal axes
            w = np.zeros(3)
            if Rp is not None:
                W = (Rl - Rp) @ Rl.T / dt
                w = 0.5 * np.array([W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1]])
            w_perp = w - axis * (w @ axis)
            lev = self.kp_lev[n] * np.cross(Rl[:, 2], axis) - self.kd_lev[n] * w_perp
            if not self.level:
                lev[:] = 0.0
            nl = np.linalg.norm(lev)
            if nl > self.lev_max[n]:
                lev *= self.lev_max[n] / nl
            self.tau[n][:] = tz * axis + lev

    # -- libuipc side
    def install(self, coupler):
        """Runs inside IPCCoupler._finalize_ipc, before world.init: one external-force constitution per lid body
        plus an animator that writes the current generalized force every step."""
        import uipc
        from uipc.constitution import AffineBodyExternalBodyForce

        ext = AffineBodyExternalBodyForce()
        coupler._ipc_constitution_tabular.insert(ext)
        for n, lid in self.env.lids.items():
            link = lid.base_link
            slot = coupler._abd_slots_by_link[link][0]
            ext.apply_to(slot.geometry(), np.zeros(12))
            (obj,) = coupler._ipc_objects.find(f"rigid_link_{link.idx}_0")
            coupler._ipc_animator.insert(obj, partial(self._animate, n, uipc))

    def _animate(self, n, uipc, info):
        geo = info.geo_slots()[0].geometry()
        uipc.view(geo.instances().find(uipc.builtin.is_constrained))[0] = 1
        A = np.asarray(geo.transforms().view()[0])[:3, :3]
        F = np.zeros(12)
        F[3:] = (0.5 * skew(self.tau[n]) @ A).reshape(-1)
        uipc.view(geo.instances().find("external_force"))[:] = F.reshape(-1, 1)


_FINALIZE_HOOKS = {}
_CENTROID_PATCHED = []


def patch_coupling_at_geometry_centroid():
    """Two-way soft-constraint reaction measured and applied at the coupled link's IPC geometry centroid.

    Genesis 1.3.1 computes the reaction on a coupled robot link as k (p_ipc - p_aim) of the link ORIGIN and applies it
    there. libuipc's SoftTransformConstraint is mass-weighted over the link's IPC geometry, i.e. it pulls at the
    geometry's centroid. For the YAM fingertips the origin is 70-90 mm away from the 8 x 30 x 40 mm pad, so when a short
    object (a 12-14 mm lid) loads only the bottom of the pad, the pad copy tilts a fraction of a degree, the origin moves
    by millimetres and the explicit reaction pulls the arm down; friction against the lid feeds it back and it diverged
    in 10 steps (fingers crossed, hand 6 cm through the table; scratch/gmin_*.log). Measuring the spring at the centroid
    makes the two sides consistent; the rotational spring is Genesis's own. Non-robot (ipc_only) links get no reaction,
    as in robots/yam_cart.patch_ipc_only_coupling_forces. Must run after add_cart(ipc=True) (it replaces that patch)."""
    from genesis.engine.couplers.ipc_coupler import coupler as C

    if _CENTROID_PATCHED:
        C.IPCCoupler._apply_abd_coupling_forces = _CENTROID_PATCHED[0]
        return
    cache = {}

    def _apply(self):
        if (not self.options.two_way_coupling
                or C.COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT not in self._entities_by_coup_type
                or not self._abd_data_by_link):
            return
        d = self._coupling_data
        if id(self) not in cache:
            soft = set(self._entities_by_coup_type[C.COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT])
            mask = np.array([link.entity in soft for link in d.links])
            cents = np.zeros((len(d.links), 3))
            for i, link in enumerate(d.links):
                if mask[i]:
                    geo = self._abd_slots_by_link[link][0].geometry()
                    cents[i] = np.asarray(geo.positions().view()).reshape(-1, 3).mean(0)
            cache[id(self)] = (mask, cents)
        mask, cents = cache[id(self)]
        C.update_coupling_forces(d.ipc_transforms, d.aim_transforms, d.links_mass, d.links_inertia_i,
                                 self._constraint_strength_translation_scaled,
                                 self._constraint_strength_rotation_scaled, d.out_forces, d.out_torques)
        Ti, Ta = d.ipc_transforms, d.aim_transforms
        c_ipc = np.einsum("blij,lj->bli", Ti[..., :3, :3], cents) + Ti[..., :3, 3]
        c_aim = np.einsum("blij,lj->bli", Ta[..., :3, :3], cents) + Ta[..., :3, 3]
        F = self._constraint_strength_translation_scaled * d.links_mass[None, :, None] * (c_ipc - c_aim)
        d.out_torques[:] = d.out_torques + np.cross(c_aim - Ta[..., :3, 3], F)  # F acts at the centroid
        d.out_forces[:] = F
        d.out_forces[:, ~mask] = 0.0
        d.out_torques[:, ~mask] = 0.0
        if np.isnan(d.out_forces).any() or np.isnan(d.out_torques).any():
            raise RuntimeError("NaN IPC coupling forces")
        f, tq = (d.out_forces, d.out_torques) if self.sim.n_envs > 0 else (d.out_forces[0], d.out_torques[0])
        self.rigid_solver.apply_links_external_force(f, links_idx=d.links_idx, local=False)
        self.rigid_solver.apply_links_external_torque(tq, links_idx=d.links_idx, local=False)

    _CENTROID_PATCHED.append(_apply)
    C.IPCCoupler._apply_abd_coupling_forces = _apply


def _patch_coupler():
    from genesis.engine.couplers.ipc_coupler.coupler import IPCCoupler

    if IPCCoupler.__dict__.get("_jars_patched", False):
        return
    orig = IPCCoupler._finalize_ipc

    def _finalize_ipc(self):
        hook = _FINALIZE_HOOKS.pop(id(self), None)
        if hook is not None:
            hook(self)
        orig(self)

    IPCCoupler._finalize_ipc = _finalize_ipc
    IPCCoupler._jars_patched = True


# ---------------------------------------------------------------- environment
@dataclass
class JarsEnv:
    gs: object
    scene: object
    table: object
    jars: dict
    lids: dict
    robot: object = None
    cams: dict = field(default_factory=dict)
    dt: float = DT
    t: float = 0.0
    rel_yaw: dict = field(default_factory=dict)  # (lid, jar) -> unwrapped lid yaw relative to the jar, rad
    rel_omega: dict = field(default_factory=dict)
    engaged_turn: dict = field(default_factory=dict)  # (lid, jar) -> clockwise turns made while on that jar's mouth
    drive: LidTorqueDrive = None
    upside_down: dict = field(default_factory=dict)

    # -- state
    def pose(self, ent):
        """Pose (p, R) of the entity's MESH-FILE frame, taken from libuipc's own body transform.

        Genesis's get_pos/get_quat of an ipc_only body lag the libuipc state by exactly g dt^2 straight down (Genesis
        integrates gravity once more after the coupler writes the pose: the 'double time integration' FIXME in
        IPCCoupler._post_advance_ipc_only; 0.109 mm at dt 1/300, 0.436 mm at 1/150, measured in scratch/t_frames.py).
        libuipc's transform maps the link frame; Genesis put that frame at the centre of mass / principal axes, so the
        mesh-file frame is T_ipc * (geom.init_pos, geom.init_quat)."""
        co = self.scene.sim.coupler
        data = co._abd_data_by_link.get(ent.base_link) if co._abd_data_by_link else None
        if data is None:  # not simulated by libuipc (e.g. before build)
            p, q = ent.get_pos(), ent.get_quat()
            p = p.cpu().numpy() if hasattr(p, "cpu") else np.asarray(p)
            q = q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)
            return p.reshape(3).astype(float), quat_to_R(q.reshape(4).astype(float))
        T = np.asarray(data[0].transform, float)
        U, _, Vt = np.linalg.svd(T[:3, :3])  # nearest rotation to the (rigid-in-practice) affine map
        R_ipc = U @ Vt
        g = ent.geoms[0]
        Rg = quat_to_R(np.asarray(g.init_quat, float))
        return R_ipc @ np.asarray(g.init_pos, float) + T[:3, 3], R_ipc @ Rg

    def poses(self):
        return {n: self.pose(e) for n, e in {**{f"jar_{k}": v for k, v in self.jars.items()},
                                              **{f"lid_{k}": v for k, v in self.lids.items()}}.items()}

    def lid_on_jar(self, lid, jar, poses=None):
        """Lid seat-plane origin in the jar frame and the relative orientation."""
        poses = poses or self.poses()
        pj, Rj = poses[f"jar_{jar}"]
        pl, Rl = poses[f"lid_{lid}"]
        rel = Rj.T @ (pl - pj)
        Rrel = Rj.T @ Rl
        tilt = math.degrees(math.acos(np.clip(Rrel[2, 2], -1, 1)))
        return dict(offset_xy=float(np.hypot(rel[0], rel[1])), height_above_rim=float(rel[2] - part(jar)["rim_z"]),
                    tilt_deg=tilt, yaw=math.atan2(Rrel[1, 0], Rrel[0, 0]), z=float(rel[2]))

    def _track(self, first=False):
        poses = self.poses()
        for ln in self.lids:
            for jn in self.jars:
                s = self.lid_on_jar(ln, jn, poses)
                key = (ln, jn)
                if first or key not in self.rel_yaw:
                    self.rel_yaw[key], self.rel_omega[key], self.engaged_turn[key] = s["yaw"], 0.0, 0.0
                    continue
                prev = self.rel_yaw[key]
                d = (s["yaw"] - prev + np.pi) % (2 * np.pi) - np.pi
                self.rel_yaw[key] = prev + d
                self.rel_omega[key] = d / self.dt
                on_mouth = s["offset_xy"] < 3e-3 and s["tilt_deg"] < 10 and -1e-3 < s["height_above_rim"] < 2.5 * part(jn)["pitch"]
                if on_mouth:
                    self.engaged_turn[key] += -d / (2 * np.pi)  # clockwise from above counts positive

    def step(self, n=1, cart_update=True):
        for _ in range(n):
            if self.drive is not None:
                self.drive.control()
            if self.robot is not None and cart_update:
                self.robot.update()
            self.scene.step()
            self.t += self.dt
            self._track()

    # -- task
    def success(self, tol_xy=1.5e-3, tol_tilt=3.0, seat_band=(-0.2e-3, 0.6e-3), min_turns=0.9):
        poses = self.poses()
        out = {}
        for n in NAMES:
            s = self.lid_on_jar(n, n, poses)
            turns = self.engaged_turn[(n, n)]
            ok = (s["offset_xy"] < tol_xy and s["tilt_deg"] < tol_tilt
                  and seat_band[0] <= s["height_above_rim"] <= seat_band[1] and turns >= min_turns)
            out[n] = dict(ok=bool(ok), offset_xy_mm=s["offset_xy"] * 1e3, tilt_deg=s["tilt_deg"],
                          height_above_rim_mm=s["height_above_rim"] * 1e3, engaged_turns=turns)
        out["all"] = all(v["ok"] for v in out.values())
        return out


def build(gs, kind="puzzle", robot=False, show_viewer=False, cameras=True, dt=DT, jar_xy=None, lid_start=None,
          drive=True, cam_res=(1280, 720), coupler_kw=None):
    """kind='puzzle': jars upright in a row, lids loose on the table (one upside down), per JAR_XY / LID_XY.
    kind='twist': each lid coaxially above its own jar, rotated so its thread is n_up turns out from seated.
    lid_start: {name: (pos, quat)} overrides lid poses; jar_xy overrides jar positions."""
    _patch_coupler()
    jar_xy = dict(JAR_XY if jar_xy is None else jar_xy)
    coupler_opts = dict(contact_d_hat=D_HAT, contact_eps_velocity=EPS_VELOCITY, newton_tolerance=NEWTON_TOL,
                        enable_rigid_rigid_contact=True, enable_rigid_ground_contact=False, two_way_coupling=True)
    coupler_opts.update(coupler_kw or {})
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, -9.81)),
        coupler_options=gs.options.IPCCouplerOptions(**coupler_opts),
        vis_options=gs.options.VisOptions(background_color=layout.BACKGROUND_RGB, ambient_light=(0.45, 0.45, 0.45),
                                          lights=[dict(type="directional", dir=(-0.3, 0.5, -1.0), color=(1, 1, 1),
                                                       intensity=3.0)]),
        viewer_options=gs.options.ViewerOptions(camera_pos=(0.0, -0.9, 1.25), camera_lookat=(0.0, 0.0, 0.8)),
        show_viewer=show_viewer,
    )
    T = layout.TABLE
    table = scene.add_entity(
        gs.morphs.Box(pos=(0.0, 0.0, TABLE_Z - T["top_thickness"] / 2), size=(T["size_x"], T["size_y"], T["top_thickness"]),
                      fixed=True),
        material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=FRICTION["table"]),
        surface=gs.surfaces.Default(color=layout.FLOOR_RGB),
    )
    jars, lids, upside = {}, {}, {}
    for n in NAMES:
        x, y = jar_xy[n]
        jars[n] = scene.add_entity(
            gs.morphs.Mesh(file=os.path.join(ASSETS, part(n)["jar_col"]["file"]), pos=(x, y, TABLE_Z + LIFT),
                           convexify=False, decimate=False),
            material=gs.materials.Rigid(rho=PARAMS["glass_rho"], coup_type="ipc_only", coup_friction=FRICTION["glass"]),
            surface=gs.surfaces.Glass(color=GLASS_RGBA[:3]) if False else gs.surfaces.Plastic(color=GLASS_RGBA),
        )
    for n in NAMES:
        sp = part(n)["spec"]
        if lid_start is not None and n in lid_start:
            pos, quat = lid_start[n]
            upside[n] = quat_to_R(np.asarray(quat, float))[2, 2] < 0
        elif kind == "twist":
            x, y = jar_xy[n]
            pos, quat = twist_start_pose(n, (x, y, TABLE_Z + LIFT))
            upside[n] = False
        else:
            (x, y), side = LID_XY[n]
            upside[n] = side == "down"
            pos = (x, y, TABLE_Z + LIFT + lid_bottom_offset(n, upside[n]))
            quat = yaw_quat(0.0, flip=upside[n])
        kind_mat = sp["lid_kind"]
        lids[n] = scene.add_entity(
            gs.morphs.Mesh(file=os.path.join(ASSETS, part(n)["lid_col"]["file"]), pos=tuple(pos), quat=tuple(quat),
                           convexify=False, decimate=False),
            material=gs.materials.Rigid(rho=part(n)["lid_rho_eff"], coup_type="ipc_only",
                                        coup_friction=FRICTION[kind_mat]),
            surface=gs.surfaces.Metal(color=tuple(sp["lid_color"])) if kind_mat == "metal"
            else gs.surfaces.Plastic(color=tuple(sp["lid_color"])),
        )
    cart = None
    if robot:
        sys.path.insert(0, os.path.join(ROOT, "robots"))
        from yam_cart import add_cart

        cart = add_cart(scene, gs, layout.STATIONS["jars"]["arm"], gripper=layout.STATIONS["jars"]["gripper"], ipc=True)
        patch_coupling_at_geometry_centroid()
    env = JarsEnv(gs=gs, scene=scene, table=table, jars=jars, lids=lids, robot=cart, dt=dt, upside_down=upside)
    cams = {}
    if cameras:
        # from the far side of the table, looking back towards the robot (the cart would hide the jars from -y)
        cams["wide"] = scene.add_camera(res=cam_res, pos=(0.62, 0.78, 1.30), lookat=(0.0, -0.14, 0.82), fov=40)
        cams["close"] = scene.add_camera(res=cam_res, pos=(0.22, 0.52, 0.97), lookat=(0.0, -0.08, 0.81), fov=30)
    env.cams = cams
    if drive:
        env.drive = LidTorqueDrive(env)
        _FINALIZE_HOOKS[id(scene.sim.coupler)] = env.drive.install
    scene.build()
    if cart is not None:
        cart.after_build()
    env._track(first=True)
    return env


def twist_start_pose(name, jar_base_pos, n_up=1.0, drop=1.5e-3):
    """Lid pose on its own jar, n_up turns unscrewed from seated (lid riding on its lower flank) and lifted `drop`
    above that, so it falls a little onto the thread and rests there."""
    p = part(name)
    psi = 2 * np.pi * n_up
    z = jar_base_pos[2] + p["rim_z"] + n_up * p["pitch"] + drop
    return (jar_base_pos[0], jar_base_pos[1], z), yaw_quat(psi)
