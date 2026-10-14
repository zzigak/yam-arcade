"""Mahjong station: 14 riichi tiles scattered on the table (some face down, rotated, two little stacks) and a slotted
wooden tile rack. Task: stand every tile in the rack facing the robot, sorted into the hand order of spec.HAND
(suits grouped m < p < s < honours, numbers ascending), one tile per slot.

    env = build()                 # scattered start, robot attached if robots/READY exists (robot="auto")
    env.scene.step()
    ok, detail = env.success()    # per-slot / per-tile verdict

Everything here is a free rigid body under gravity on a fixed table top (z = 0.75, layout extents). Tiles use box
collision (exact 26 x 35 x 21 mm), the rack is a union of 20 boxes (the same boxes make its visual mesh).
"""
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

import genesis as gs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import layout  # noqa: E402
import spec  # noqa: E402

ASSETS = os.path.join(HERE, "assets")
TABLE_Z = layout.TABLE_TOP_Z

# ---------------------------------------------------------------------------------------------- physical parameters
DT = 0.002                    # s; 500 Hz rigid step (see NOTES.md: contact time constant must be >= 2 dt)
CONSTRAINT_TIMECONST = 0.005  # s; contact stiffness time constant -> ~0.05 mm resting penetration
FRICTION_CONE = "elliptic"    # exact friction cone; with a high impratio it stops soft-friction creep (see NOTES.md)
IMPRATIO = 1000.0             # friction/normal impedance ratio: tile creep in the grip 0.65 mm/s -> 0, in the rack ~0.2 um/s
NOSLIP_ITERATIONS = 0         # no-slip post-solve (incompatible with the elliptic cone; it also made resting tiles buzz)
MU_TILE = 0.35                # melamine on melamine / on lacquered wood
MU_TABLE = 0.35               # lacquered / laminated table top
MU_RACK = 0.40                # oiled hardwood
# preview cameras (Genesis rasterizer): name -> (pos, lookat, vertical fov[, (w, h)])
CAMERAS = {
    # from the far right corner of the table, looking back at the robot (the robot-side views are blocked by the arms)
    "overview": ((0.80, 0.66, 1.30), (-0.03, -0.10, 0.80), 46),
    # straight down over the table (the arms at the bottom edge)
    "top": ((0.0, -0.06, 1.72), (0.0, -0.05, 0.75), 46),
    # close-ups from in front of the cart's camera mast: the scattered tiles, and the rack (tile faces face this way)
    "tiles": ((-0.02, -0.36, 1.06), (-0.02, -0.10, 0.755), 44),
    "rack": ((-0.10, -0.25, 0.94), (-0.10, 0.07, 0.785), 36),
    "rack_full": ((0.0, -0.40, 1.04), (0.0, 0.07, 0.785), 36),
}
TILE_COLOR = (0.955, 0.937, 0.880)
TILE_BACK_COLOR = (0.075, 0.430, 0.330)
RACK_COLOR = (0.60, 0.40, 0.24)


# ---------------------------------------------------------------------------------------------------- small helpers
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qaxis(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    h = math.radians(deg) / 2
    return np.concatenate([[math.cos(h)], math.sin(h) * a])


def qmat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def to_np(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


# --------------------------------------------------------------------------------------------------------- layouts
@dataclass
class TilePlacement:
    kind: str
    pos: tuple
    quat: tuple
    note: str = ""


def scattered_layout():
    """The start state: 14 tiles on the table between the robot and the rack. 10 single tiles and two stacks of two;
    4 tiles face down; yaw spread over +-180 deg. Deterministic (spec.SCATTER_SEED)."""
    rng = np.random.default_rng(spec.SCATTER_SEED)
    kinds = list(spec.HAND)
    rng.shuffle(kinds)
    # 12 footprints (two of them carry a second tile): rejection-sample centres with a circumscribed-disc clearance
    r_foot = 0.5 * math.hypot(spec.TILE_W, spec.TILE_H)
    centres = []
    while len(centres) < 12:
        c = np.array([rng.uniform(-0.40, 0.40), rng.uniform(-0.215, -0.02)])
        if all(np.linalg.norm(c - o) > 2 * r_foot + 0.012 for o in centres):
            centres.append(c)
    centres.sort(key=lambda c: (c[0]))
    face_down = {1, 4, 8, 10}          # footprint indices whose (bottom) tile lies face down
    stacks = {3: True, 9: False}       # footprint index -> top tile face down?
    gap = 0.0001                       # spawn gap above the supporting surface; tiles drop 0.1 mm onto it
    out, ki = [], 0
    for i, c in enumerate(centres):
        yaw = rng.uniform(-180, 180)
        q = qaxis((0, 0, 1), yaw)
        if i in face_down:
            q = qmul(q, qaxis((0, 1, 0), 180))
        z = TABLE_Z + 0.5 * spec.TILE_T + gap
        out.append(TilePlacement(kinds[ki], (c[0], c[1], z), tuple(q), "face down" if i in face_down else "face up"))
        ki += 1
        if i in stacks:
            q2 = qaxis((0, 0, 1), yaw + rng.uniform(15, 35) * rng.choice([-1, 1]))
            if stacks[i]:
                q2 = qmul(q2, qaxis((0, 1, 0), 180))
            off = rng.uniform(-0.003, 0.003, 2)
            out.append(TilePlacement(kinds[ki], (c[0] + off[0], c[1] + off[1], z + spec.TILE_T + gap), tuple(q2),
                                     "stacked, " + ("face down" if stacks[i] else "face up")))
            ki += 1
    assert ki == len(kinds)
    return out


def rack_pose():
    return (spec.RACK_POS[0], spec.RACK_POS[1], TABLE_Z), (1.0, 0.0, 0.0, 0.0)


def solved_layout(gap=0.0001):
    """Every tile seated in its target slot (used to validate success() and the rack's holding behaviour)."""
    (rx, ry, rz), _ = rack_pose()
    out = []
    for k, kind in enumerate(spec.HAND_SORTED):
        (x, y, z), q = spec.seated_tile_pose_rack(k)
        # back off by `gap` along the shelf normal and the backboard normal so nothing starts in contact
        u = np.array([0.0, spec.SIN_A, spec.COS_A])
        f = np.array([0.0, -spec.COS_A, spec.SIN_A])
        p = np.array([rx + x, ry + y, rz + z]) + gap * (u + f)
        out.append(TilePlacement(kind, tuple(p), tuple(q), f"slot {k}"))
    return out


# ------------------------------------------------------------------------------------------------------------- env
@dataclass
class Tile:
    name: str
    kind: str
    entity: object
    start: TilePlacement
    mesh: str


@dataclass
class MahjongEnv:
    scene: object
    table: object
    rack: object
    tiles: list
    robot: object = None
    cameras: dict = field(default_factory=dict)
    dt: float = DT

    # ---------------------------------------------------------------------------------------------------- state
    def tile_poses(self):
        P = np.stack([to_np(t.entity.get_pos()) for t in self.tiles])
        Q = np.stack([to_np(t.entity.get_quat()) for t in self.tiles])
        return P, Q

    def tile_speeds(self):
        V = np.stack([to_np(t.entity.get_dofs_velocity()) for t in self.tiles])
        return np.linalg.norm(V[:, :3], axis=1), np.linalg.norm(V[:, 3:], axis=1)

    def rack_pose(self):
        return to_np(self.rack.get_pos()), to_np(self.rack.get_quat())

    def contacts(self):
        """All current contacts (numpy): link_a, link_b, penetration, position, normal, force."""
        c = self.scene.sim.rigid_solver.collider.get_contacts(as_tensor=True, to_torch=False)
        return {k: np.asarray(v) for k, v in c.items()}

    def link_names(self):
        names = {}
        for t in self.tiles:
            for l in t.entity.links:
                names[l.idx] = t.name
        for l in self.rack.links:
            names[l.idx] = "rack"
        for l in self.table.links:
            names[l.idx] = "table"
        if self.robot is not None:
            for l in self.robot_entity().links:
                names[l.idx] = f"robot/{l.name}"
        return names

    def robot_entity(self):
        return self.robot.entity

    # ---------------------------------------------------------------------------------------------------- task
    def slot_report(self, pos_tol_x=0.005, pos_tol_bu=0.004, ang_tol_deg=10.0, rest_speed=0.005):
        """For every tile: which slot (if any) it is seated in, and whether it is upright, facing the robot, at rest.

        Seated in slot k = tile centre, expressed in the rack's current frame, within pos_tol_x of the slot centre
        along the rack and within pos_tol_bu of the seated pose in the profile plane (back face on the backboard,
        bottom face on the shelf). Upright and facing = tile +y within ang_tol of the rack's 'up the backboard' axis
        and tile +z (engraved face) within ang_tol of the outward normal towards the robot."""
        rp, rq = self.rack_pose()
        Rr = qmat(rq)
        P, Q = self.tile_poses()
        lin, _ = self.tile_speeds()
        u = np.array([0.0, spec.SIN_A, spec.COS_A])
        f = np.array([0.0, -spec.COS_A, spec.SIN_A])
        cos_tol = math.cos(math.radians(ang_tol_deg))
        rows = []
        for i, t in enumerate(self.tiles):
            pl = Rr.T @ (P[i] - rp)
            Rl = Rr.T @ qmat(Q[i])
            k = int(round(pl[0] / spec.SLOT_PITCH + 0.5 * (spec.N_SLOTS - 1)))
            row = dict(tile=t.name, kind=t.kind, slot=None, seated=False, upright=False, facing=False, at_rest=False,
                       correct=False)
            if 0 <= k < spec.N_SLOTS:
                (sx, sy, sz), _ = spec.seated_tile_pose_rack(k)
                dx = pl[0] - sx
                dbu = math.hypot(pl[1] - sy, pl[2] - sz)
                up_cos = float(Rl[:, 1] @ u)
                face_cos = float(Rl[:, 2] @ f)
                row.update(slot=k, dx_mm=1000 * dx, dbu_mm=1000 * dbu,
                           up_err_deg=math.degrees(math.acos(min(1, max(-1, up_cos)))),
                           face_err_deg=math.degrees(math.acos(min(1, max(-1, face_cos)))))
                row["seated"] = abs(dx) <= pos_tol_x and dbu <= pos_tol_bu
                row["upright"] = up_cos >= cos_tol
                row["facing"] = face_cos >= cos_tol
            row["at_rest"] = float(lin[i]) <= rest_speed
            row["speed_mm_s"] = 1000 * float(lin[i])
            ok = row["seated"] and row["upright"] and row["facing"] and row["at_rest"]
            row["correct"] = bool(ok and spec.HAND_SORTED[row["slot"]] == t.kind)
            if not ok:
                row["slot"] = None if not row["seated"] else row["slot"]
            rows.append(row)
        return rows

    def success(self):
        """True iff all 14 slots hold exactly one tile, each seated, upright, facing the robot, at rest, and the
        kinds read spec.HAND_SORTED left to right (the robot's left = -x), and the rack stands upright on the table.
        Returns (ok, detail)."""
        rows = self.slot_report()
        rp, rq = self.rack_pose()
        rack_up = float(qmat(rq)[:, 2] @ np.array([0, 0, 1.0]))
        rack_ok = rack_up >= math.cos(math.radians(3)) and abs(rp[2] - TABLE_Z) < 0.002
        filled = {}
        for r in rows:
            if r["slot"] is not None and r["seated"]:
                filled.setdefault(r["slot"], []).append(r)
        slots = []
        for k in range(spec.N_SLOTS):
            occ = filled.get(k, [])
            good = len(occ) == 1 and occ[0]["correct"]
            slots.append(dict(slot=k, want=spec.HAND_SORTED[k], have=[o["kind"] for o in occ], ok=good))
        n_correct = sum(s["ok"] for s in slots)
        ok = rack_ok and n_correct == spec.N_SLOTS
        return ok, dict(n_correct=n_correct, n_slots=spec.N_SLOTS, rack_upright_cos=rack_up, rack_ok=rack_ok,
                        slots=slots, tiles=rows)


# ----------------------------------------------------------------------------------------------------------- build
def robot_ready():
    return os.path.exists(os.path.join(ROOT, "robots", "READY"))


def build(layout_name="scattered", robot="auto", show_viewer=False, cameras=True, res=(1280, 720),
          friction_cone=FRICTION_CONE, timeconst=CONSTRAINT_TIMECONST, dt=DT, init=True, backend=None,
          precision="64", noslip=NOSLIP_ITERATIONS, impratio=IMPRATIO):
    """Build the station scene. layout_name: 'scattered' (task start) or 'solved' (every tile in its slot).
    robot: 'auto' (attach if robots/READY), True, or False. Returns a MahjongEnv (scene already built)."""
    if init:
        gs.init(backend=backend if backend is not None else gs.cpu, precision=precision, logging_level="warning")
    cone = dict(elliptic=gs.friction_cone.elliptic, pyramidal=gs.friction_cone.pyramidal)[friction_cone]
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=1, gravity=(0, 0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            box_box_detection=True,
            constraint_timeconst=timeconst,
            friction_cone=cone,
            enable_torsional_friction=True,
            iterations=100,
            ls_iterations=50,
            noslip_iterations=noslip,
            impratio=impratio,
            max_collision_pairs=400,
        ),
        vis_options=gs.options.VisOptions(
            background_color=layout.BACKGROUND_RGB,
            ambient_light=(0.42, 0.42, 0.44),
            shadow=True,
            lights=[dict(type="directional", dir=(-0.35, 0.55, -1.0), color=(1.0, 0.98, 0.95), intensity=2.6),
                    dict(type="directional", dir=(0.6, -0.4, -0.7), color=(0.95, 0.97, 1.0), intensity=1.2)],
        ),
        viewer_options=gs.options.ViewerOptions(camera_pos=(0, -1.0, 1.35), camera_lookat=(0, 0, 0.75),
                                                camera_fov=40),
        show_viewer=show_viewer,
    )
    # floor (preview only; nothing reaches it) and the station's table top as one fixed box
    scene.add_entity(gs.morphs.Plane(pos=(0, 0, 0)), surface=gs.surfaces.Default(color=layout.FLOOR_RGB))
    t = layout.TABLE
    table = scene.add_entity(
        gs.morphs.Box(size=(t["size_x"], t["size_y"], t["top_thickness"]),
                      pos=(0, 0, TABLE_Z - 0.5 * t["top_thickness"]), fixed=True),
        material=gs.materials.Rigid(friction=MU_TABLE),
        surface=gs.surfaces.Default(color=(0.93, 0.92, 0.90)),
    )
    rpos, rquat = rack_pose()
    rack = scene.add_entity(
        gs.morphs.URDF(file=os.path.join(ASSETS, "rack", "rack.urdf"), pos=rpos, quat=rquat, fixed=False,
                       prioritize_urdf_material=False),
        material=gs.materials.Rigid(friction=MU_RACK),
    )
    placements = scattered_layout() if layout_name == "scattered" else solved_layout()
    tiles = []
    for i, p in enumerate(placements):
        ent = scene.add_entity(
            gs.morphs.URDF(file=os.path.join(ASSETS, "tiles", f"tile_{p.kind}.urdf"), pos=p.pos, quat=p.quat,
                           fixed=False),
            material=gs.materials.Rigid(friction=MU_TILE),
        )
        tiles.append(Tile(f"tile_{i:02d}_{p.kind}", p.kind, ent, p, os.path.join(ASSETS, "tiles", f"tile_{p.kind}.obj")))

    robot_obj = None
    if robot is True or (robot == "auto" and robot_ready()):
        sys.path.insert(0, os.path.join(ROOT, "robots"))
        from yam_cart import add_cart  # noqa: E402  (robot agent's module; API in robots/README.md)
        robot_obj = add_cart(scene, gs, layout.STATIONS["mahjong"]["arm"], gripper=layout.STATIONS["mahjong"]["gripper"])

    cams = {}
    if cameras:
        for name, spec_c in (CAMERAS if cameras is True else cameras).items():
            pos, lookat, fov = spec_c[:3]
            cams[name] = scene.add_camera(res=spec_c[3] if len(spec_c) > 3 else res, pos=pos, lookat=lookat, fov=fov)
    scene.build()
    if robot_obj is not None:
        robot_obj.after_build()
    env = MahjongEnv(scene, table, rack, tiles, robot_obj, cams, dt)
    return env
