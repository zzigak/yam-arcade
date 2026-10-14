"""Cables station: eight cables of two colours, different stiffness and curved rest shapes, dropped from the air into
a tangled pile on the table; two open-top bins (one per colour) to sort them into.

    env = build(robot=False)          # Genesis scene with table, bins, cables (+ the YAM Ultra cart if robot=True)
    env.step()                        # one sim step (cart.update() first when there is a robot)
    env.cable_points()                # {name: (n, 3) centreline}
    env.success()                     # per-cable bin fractions, overall sorted fraction
    env.tangle_metric()               # crossings (top view), contacts, interlocked pairs

Cable model: cable_model.Framed (libuipc rod + ghost material frames + SoftVertexStitch network) so that each cable
keeps its rest shape and has isotropic bending EI and torsion GJ. See NOTES.md.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "robots"))
import cable_model as CM  # noqa: E402
import layout  # noqa: E402
import plugs as PL  # noqa: E402

ZT = layout.TABLE_TOP_Z
DT = float(os.environ.get("CABLES_DT", 1.0 / 90.0))  # 1/90: 3 steps per 30 fps frame; 1/60: 2
SEG_SCALE = float(os.environ.get("CABLES_SEG_SCALE", 1.0))  # profiling: coarser rod vertex spacing
ASSETS = os.path.join(HERE, "assets")

# Two colours that read on the light-grey studio and stay distinct for colour-blind viewers (orange vs blue).
COLORS = dict(orange=(0.93, 0.45, 0.08), blue=(0.11, 0.38, 0.80))
BIN_TINT = dict(orange=(0.98, 0.83, 0.66), blue=(0.72, 0.83, 0.96))  # pale bins, saturated labels: cables stay visible

# ------------------------------------------------------------------------------------------------------------ cables
# EI: N m^2. rho_line: kg/m. Shapes in cable_model.rest_curve. seg: vertex spacing (seg - 2 r > d_hat, libuipc rod rule).
D = np.radians


def _p(pieces):
    """Shape spec from explicit pieces; every cable is curved along its whole length (no straight runs)."""
    return dict(shape="pieces", pieces=pieces, length=CM.pieces_length(pieces))


CABLES = [
    dict(name="power_coil", plug="iec_c13", color="orange", what="IEC mains cord, 3 x 1.0 mm2, OD 9 mm, still coiled from its packaging",
         shape=_p([("arc", 0.12, D(40)), ("helix", 0.055, D(720), 0.028), ("arc", 0.15, D(-32))]),
         radius=0.0045, seg=0.014, EI=4.0e-3, rho_line=0.085, friction=0.5),
    dict(name="hdmi_ubend", plug="hdmi", color="blue", what="HDMI lead, OD 8 mm, stored folded double; the legs keep a bow",
         shape=_p([("arc", 0.50, D(26)), ("arc", 0.045, D(180)), ("arc", 0.60, D(-22))]),
         radius=0.004, seg=0.013, EI=2.5e-3, rho_line=0.070, friction=0.5),
    dict(name="ext_coil", plug="euro", color="blue", what="light extension lead 2 x 0.75 mm2, OD 7 mm, loosely coiled",
         shape=_p([("arc", 0.15, D(-45)), ("helix", 0.07, D(540), 0.0165), ("arc", 0.15, D(45))]),
         radius=0.0035, seg=0.012, EI=1.2e-3, rho_line=0.050, friction=0.5),
    dict(name="cat6_scurve", plug="rj45", color="orange", what="Cat6 patch cord, OD 6 mm, S-bends from cable ties",
         shape=_p([("arc", 0.10, D(40)), ("arc", 0.08, D(-140)), ("arc", 0.08, D(140)), ("arc", 0.09, D(-120)),
                   ("arc", 0.12, D(40))]),
         radius=0.003, seg=0.010, EI=5.0e-4, rho_line=0.040, friction=0.5),
    dict(name="barrel_arc", plug="barrel", color="blue", what="DC barrel-plug lead, OD 5 mm, curled tighter towards one end",
         shape=_p([("arc", 0.13, D(90)), ("arc", 0.095, D(90)), ("arc", 0.065, D(90))]),
         radius=0.0025, seg=0.009, EI=1.0e-4, rho_line=0.022, friction=0.5),
    dict(name="usbc_wave", plug="usbc", color="blue", what="USB-C charging lead, OD 4 mm, kinked into waves",
         shape=_p([("arc", 0.032, D(75))] + [("arc", 0.032, D(-150 if i % 2 == 0 else 150)) for i in range(5)]
                  + [("arc", 0.032, D(-75))]),
         radius=0.002, seg=0.009, EI=4.0e-5, rho_line=0.018, friction=0.5),
    dict(name="micro_scurve", plug="micro_usb", color="orange", what="short micro-USB lead, OD 4 mm, S-shaped",
         shape=_p([("arc", 0.0625, D(160)), ("arc", 0.0625, D(-160))]),
         radius=0.002, seg=0.009, EI=3.0e-5, rho_line=0.016, friction=0.5),
    dict(name="aux_loop", plug="jack35", color="orange", what="3.5 mm audio lead, OD 4 mm, with a loop kinked into it",
         shape=_p([("arc", 0.20, D(48)), ("helix", 0.04, D(360), 0.009), ("arc", 0.25, D(-40))]),
         radius=0.002, seg=0.009, EI=2.0e-5, rho_line=0.015, friction=0.5),
]
CABLE_BY_NAME = {c["name"]: c for c in CABLES}

# Initial placement in the air (station frame): xy box centre, yaw (deg), z of the lowest point above the table top,
# optional tilt (axis, deg) applied before the yaw, optional flip. Iterated in tests/plan_pile.py / run.py so that
# nothing touches at start and the drop makes a tangled pile in the middle of the table.
PLACEMENT = {
    "cat6_scurve": dict(xy=(0.00, -0.05), yaw=55.0, dz=0.015),
    "usbc_wave": dict(xy=(0.02, -0.07), yaw=95.0, dz=0.035),
    "power_coil": dict(xy=(-0.08, -0.04), yaw=0.0, dz=0.055),
    "hdmi_ubend": dict(xy=(0.06, -0.08), yaw=-40.0, dz=0.100),
    "aux_loop": dict(xy=(0.00, -0.02), yaw=70.0, dz=0.125, tilt=((1, 0, 0), 90.0)),
    "micro_scurve": dict(xy=(0.00, -0.02), yaw=-20.0, dz=0.160),
    "barrel_arc": dict(xy=(0.08, 0.00), yaw=120.0, dz=0.200, tilt=((0, 1, 0), 35.0)),
    "ext_coil": dict(xy=(-0.04, -0.10), yaw=30.0, dz=0.250, tilt=((1, 0, 0), 20.0)),
}

# ------------------------------------------------------------------------------------------------------------- bins
# Open-top polypropylene storage bins, 2 mm-wall class but modelled 5 mm thick (IPC contact needs a few mm of wall to
# keep fast cable vertices from tunnelling; a real bin has ribs/rims that thick). Free rigid bodies resting on the table.
# 7 cm tall sorting trays: the YAM Ultra reaches only ~0.2 m (top-down) to ~0.27 m (forward-tilted) above the table
# over them, and a cable hanging from the gripper has to clear the rim. Placed in front of each arm (robots/README).
BIN = dict(size=(0.22, 0.28, 0.07), wall=0.005, floor=0.005, mass=0.30, friction=0.5)
BINS = {
    "orange": dict(xy=(0.39, -0.10)),
    "blue": dict(xy=(-0.39, -0.10)),
}


def arm_for_color(color):
    """The arm in front of that colour's bin (left arm at -x)."""
    return "left" if BINS[color]["xy"][0] < 0 else "right"
BIN_LIFT = 0.002  # bins start 2 mm above the table (libuipc rejects zero-distance contact at init)


def make_bin_obj(path):
    """Watertight open-top box (outer bottom at z = 0, frame at the outer bottom centre) as one solid."""
    import trimesh
    a, b = 0.5 * BIN["size"][0], 0.5 * BIN["size"][1]
    H, t, tb = BIN["size"][2], BIN["wall"], BIN["floor"]
    ai, bi = a - t, b - t
    O = [(-a, -b), (a, -b), (a, b), (-a, b)]
    Ii = [(-ai, -bi), (ai, -bi), (ai, bi), (-ai, bi)]
    V = [(x, y, 0.0) for x, y in O] + [(x, y, H) for x, y in O] + [(x, y, tb) for x, y in Ii] + [(x, y, H) for x, y in Ii]
    o, ot, ib, it = 0, 4, 8, 12
    F = [(o + 0, o + 3, o + 2), (o + 0, o + 2, o + 1)]
    for k in range(4):
        k1 = (k + 1) % 4
        F += [(o + k, o + k1, ot + k1), (o + k, ot + k1, ot + k)]  # outer wall
        F += [(ot + k, ot + k1, it + k1), (ot + k, it + k1, it + k)]  # rim
        F += [(ib + k, it + k, it + k1), (ib + k, it + k1, ib + k1)]  # inner wall
    F += [(ib + 0, ib + 1, ib + 2), (ib + 0, ib + 2, ib + 3)]
    m = trimesh.Trimesh(np.array(V), np.array(F), process=False)
    assert m.is_watertight and m.is_winding_consistent and m.volume > 0, "bin mesh"
    m.export(path)
    return m


def make_bin_urdf(collision_obj, visual_obj, urdf_path):
    """One-link URDF: flat-shaded visual (every face its own vertices, so the rasterizer does not smooth normals over
    the box corners) and the watertight solid as collision; link frame = mesh frame."""
    import trimesh
    m = trimesh.load(collision_obj, process=False)
    v = trimesh.Trimesh(m.vertices[m.faces].reshape(-1, 3), np.arange(3 * len(m.faces)).reshape(-1, 3), process=False)
    v.export(visual_obj)
    open(urdf_path, "w").write(f"""<?xml version="1.0"?>
<robot name="bin">
  <link name="bin">
    <visual><geometry><mesh filename="{os.path.basename(visual_obj)}"/></geometry></visual>
    <collision><geometry><mesh filename="{os.path.basename(collision_obj)}"/></geometry></collision>
  </link>
</robot>
""")


def make_label_obj(path):
    """Colour label sticker on the bin's front (-y) face, 0.5 mm proud, in the bin frame."""
    import trimesh
    b, H = 0.5 * BIN["size"][1], BIN["size"][2]
    m = trimesh.creation.box(extents=(0.12, 0.0005, 0.045))
    m.apply_translation((0.0, -b - 0.00025, 0.55 * H))
    m.export(path)
    return m


def bin_inner_box():
    """Inner cavity of a bin in its frame: (lo, hi) corners (up to the rim)."""
    a, b = 0.5 * BIN["size"][0] - BIN["wall"], 0.5 * BIN["size"][1] - BIN["wall"]
    return np.array([-a, -b, BIN["floor"]]), np.array([a, b, BIN["size"][2]])


# --------------------------------------------------------------------------------------------------------- geometry
PLACEMENT_JSON = os.path.join(HERE, "placement.json")  # written by tests/plan_pile.py --optimize
if os.path.exists(PLACEMENT_JSON):
    import json as _json
    PLACEMENT = _json.load(open(PLACEMENT_JSON))


_PLUG_INFO = {}


def plug_info(kind):
    if kind not in _PLUG_INFO:
        _PLUG_INFO[kind] = PL.make_plug(kind)
        import trimesh
        _PLUG_INFO[kind]["hull_pts"] = np.asarray(trimesh.load(_PLUG_INFO[kind]["hull"], process=False).vertices)
    return _PLUG_INFO[kind]


def rest_local(c):
    """Local rest layout of a cable with its plug: the cable runs straight through the plug from the tip vertex to
    the boot vertex in n_inner segments, then follows its rest shape. Returns dict(P, n_inner, R, t) where (R, t) is
    the plug pose (plug frame -> local), xy-centred, lowest point (cable or plug) at z = 0."""
    outer, _ = CM.resample(CM.rest_curve(c["shape"], center=False), c["seg"] * SEG_SCALE)
    if c.get("plug") is None:
        P = outer
        R, t, ni = None, None, 0
    else:
        info = plug_info(c["plug"])
        ni = PL.n_inner(info["length"], c["radius"], CM.D_HAT)
        R = CM.rot([0, 0, 1], np.pi)  # plug +x (toward the tip) = local -x; the cable leaves the boot along +x
        t = np.zeros(3)
        tip_local = R @ np.array(info["tip"])
        inner = np.array([tip_local * (1 - k / ni) for k in range(ni)])  # tip -> just before the boot (origin)
        P = np.vstack([inner, outer])
    allp = P if R is None else np.vstack([P, plug_info(c["plug"])["hull_pts"] @ R.T + t])
    shift = np.zeros(3)
    shift[:2] = -0.5 * (allp[:, :2].min(0) + allp[:, :2].max(0))
    shift[2] = -allp[:, 2].min()
    return dict(P=P + shift, n_inner=ni, R=R, t=None if t is None else t + shift)


def ring_of(P, spec, n_inner=0):
    """Centre and unit normal of the helical loop of a 'pieces' cable (the helix piece's vertices)."""
    P = P[n_inner:]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    a = 0.0
    for pc in spec["shape"]["pieces"]:
        L = CM.pieces_length([pc])
        if pc[0] == "helix":
            m = (s >= a) & (s <= a + L)
            Q = P[m]
            c = Q.mean(0)
            n = np.linalg.svd(Q - c)[2][2]
            return c, n
        a += L
    raise ValueError("no loop")


def place_one(c, pl, placed=None):
    """World layout of one cable: dict(P, n_inner, plug_R, plug_t)."""
    loc = rest_local(c)
    P0, ni = loc["P"], loc["n_inner"]
    if "thread" in pl:  # pass this cable's middle through the loop of another cable, perpendicular to the ring
        host = placed[pl["thread"]]
        ctr, nrm = ring_of(host["P"], CABLE_BY_NAME[pl["thread"]], host["n_inner"])
        i = ni + (len(P0) - ni) // 2
        tg = P0[i + 1] - P0[i - 1]
        nh = np.array([nrm[0], nrm[1], 0.0])
        nh /= np.linalg.norm(nh)
        yaw = np.arctan2(nh[1], nh[0]) - np.arctan2(tg[1], tg[0]) + np.radians(pl.get("yaw_extra", 0.0))
        R = CM.rot([0, 0, 1], yaw)
        t = ctr + np.array([0, 0, pl.get("dz_ring", 0.0)]) - R @ P0[i]
        P = P0 @ R.T + t
    else:
        extra = None if loc["R"] is None else plug_info(c["plug"])["hull_pts"] @ loc["R"].T + loc["t"]
        P, R, t = CM.place(P0, pl["xy"], pl["yaw"], ZT + c["radius"] + pl["dz"], tilt=pl.get("tilt"),
                           flip=pl.get("flip", False), extra=extra, return_transform=True)
    out = dict(P=P, n_inner=ni, plug_R=None, plug_t=None)
    if loc["R"] is not None:
        out["plug_R"] = R @ loc["R"]
        out["plug_t"] = R @ loc["t"] + t
    return out


def initial_layout(placement=None):
    """{name: dict(P, n_inner, plug_R, plug_t)} initial (= rest, rigidly placed) cables and plug poses."""
    placement = PLACEMENT if placement is None else placement
    out = {}
    order = sorted(CABLES, key=lambda c: "thread" in placement[c["name"]])  # threaded cables after their hosts
    for c in order:
        out[c["name"]] = place_one(c, placement[c["name"]], out)
    return {c["name"]: out[c["name"]] for c in CABLES}


def initial_polylines(placement=None):
    return {n: v["P"] for n, v in initial_layout(placement).items()}


def plug_points(name, lay):
    """World hull vertices of a cable's plug in layout entry lay."""
    c = CABLE_BY_NAME[name]
    return plug_info(c["plug"])["hull_pts"] @ lay["plug_R"].T + lay["plug_t"]


def point_seg_dist(X, P):
    """Distance from each point X (k, 3) to polyline P: (k,) array."""
    a, b = P[:-1][None], P[1:][None]
    ab = b - a
    t = np.clip(((X[:, None] - a) * ab).sum(-1) / (ab * ab).sum(-1), 0, 1)
    return np.linalg.norm(X[:, None] - (a + ab * t[..., None]), axis=-1).min(1)


def seg_seg_dist(P, Q):
    """Minimum distance between every segment of polyline P and every segment of polyline Q: (m-1, k-1) array."""
    p0, p1 = P[:-1, None, :], P[1:, None, :]
    q0, q1 = Q[None, :-1, :], Q[None, 1:, :]
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a = (d1 * d1).sum(-1)
    e = (d2 * d2).sum(-1)
    f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1)
    b = (d1 * d2).sum(-1)
    den = a * e - b * b
    s = np.where(den > 1e-14, np.clip((b * f - c * e) / np.where(den > 1e-14, den, 1), 0, 1), 0.0)
    t = (b * s + f) / e
    t = np.clip(t, 0, 1)
    s = np.clip((b * t - c) / a, 0, 1)
    t = np.clip((b * s + f) / e, 0, 1)
    diff = (p0 + d1 * s[..., None]) - (q0 + d2 * t[..., None])
    return np.linalg.norm(diff, axis=-1)


def init_clearances(polys, layout_=None):
    """Smallest surface gap between any two cables (and plugs), and between each cable/plug and the table, for an
    init check. Plugs are tested by their hull vertices (point-to-polyline / point-to-point)."""
    names = list(polys)
    gaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = seg_seg_dist(polys[a], polys[b]).min() - CABLE_BY_NAME[a]["radius"] - CABLE_BY_NAME[b]["radius"]
            gaps[f"{a}|{b}"] = float(d)
    table = {n: float(P[:, 2].min() - ZT - CABLE_BY_NAME[n]["radius"]) for n, P in polys.items()}
    if layout_ is not None:
        pp = {n: plug_points(n, lay) for n, lay in layout_.items() if lay["plug_R"] is not None}
        for n, X in pp.items():
            table[f"plug:{n}"] = float(X[:, 2].min() - ZT)
            for m, P in polys.items():
                if m == n:
                    continue
                gaps[f"plug:{n}|{m}"] = float(point_seg_dist(X, P).min() - CABLE_BY_NAME[m]["radius"])
            for m, Y in pp.items():
                if m > n:
                    gaps[f"plug:{n}|plug:{m}"] = float(np.linalg.norm(X[:, None] - Y[None], axis=-1).min())
    return gaps, table


def _seg_intersections_2d(P, Q):
    """Proper crossings of the xy projections of polylines P and Q: list of (i, j, s, t) segment indices and
    parameters."""
    a0, a1 = P[:-1, None, :2], P[1:, None, :2]
    b0, b1 = Q[None, :-1, :2], Q[None, 1:, :2]
    da, db = a1 - a0, b1 - b0
    den = da[..., 0] * db[..., 1] - da[..., 1] * db[..., 0]
    w = b0 - a0
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (w[..., 0] * db[..., 1] - w[..., 1] * db[..., 0]) / den
        t = (w[..., 0] * da[..., 1] - w[..., 1] * da[..., 0]) / den
    m = (np.abs(den) > 1e-12) & (s > 0) & (s < 1) & (t > 0) & (t < 1)
    i, j = np.nonzero(m)
    return [(int(ii), int(jj), float(s[ii, jj]), float(t[ii, jj])) for ii, jj in zip(i, j)]


def tangle_metric(pts, contact_tol=0.002):
    """Tangle measures on a dict {name: (n, 3)}:
    crossings      - top-view crossings between different cables (each records which is on top)
    self_crossings - top-view self-crossings (coil turns, loops)
    contacts       - pairs of segments of different cables whose surfaces are within contact_tol
    touching_pairs - cable pairs in contact
    interlocked    - cable pairs that cross each other both over and under (A over B somewhere, under elsewhere)
    middle         - cables lying over one cable and under another (woven, not just stacked)"""
    names = list(pts)
    crossings, contacts, touching, inter = 0, 0, [], []
    over = {n: set() for n in names}
    under = {n: set() for n in names}
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            P, Q = pts[a], pts[b]
            xs = _seg_intersections_2d(P, Q)
            sgn = []
            for ii, jj, s, t in xs:
                za = P[ii, 2] + s * (P[ii + 1, 2] - P[ii, 2])
                zb = Q[jj, 2] + t * (Q[jj + 1, 2] - Q[jj, 2])
                sgn.append(np.sign(za - zb))
                (over[a].add(b), under[b].add(a)) if za > zb else (over[b].add(a), under[a].add(b))
            crossings += len(xs)
            if len(set(sgn)) == 2:
                inter.append(f"{a}|{b}")
            d = seg_seg_dist(P, Q) - CABLE_BY_NAME[a]["radius"] - CABLE_BY_NAME[b]["radius"]
            nc = int((d < contact_tol).sum())
            contacts += nc
            if nc:
                touching.append(f"{a}|{b}")
            pairs[f"{a}|{b}"] = dict(crossings=len(xs), contacts=nc)
    selfx = 0
    for n in names:
        P = pts[n]
        for ii, jj, s, t in _seg_intersections_2d(P, P):
            if jj > ii + 1:
                selfx += 1
    middle = [n for n in names if over[n] and under[n]]
    return dict(crossings=crossings, self_crossings=selfx, contacts=contacts, touching_pairs=len(touching),
                interlocked_pairs=inter, middle=middle, pairs=pairs,
                over={n: sorted(v) for n, v in over.items()}, under={n: sorted(v) for n, v in under.items()})


# ------------------------------------------------------------------------------------------------------------- env
class Env:
    def __init__(self, scene, cables, bins, labels, robot, gs):
        self.scene, self.cables, self.bins, self.labels, self.robot, self.gs = scene, cables, bins, labels, robot, gs
        self.t = 0.0
        self._bin_d = {}

    def step(self):
        if self.robot is not None:
            self.robot.update()
        self.scene.step()
        self.t += DT

    def cable_points(self):
        from genesis.utils.misc import tensor_to_array as A
        return {n: A(c["entity"].get_state().pos).reshape(-1, 3).astype(np.float64) for n, c in self.cables.items()}

    def bin_poses(self, genesis_copy=False):
        """{colour: (pos, quat_wxyz)} of each bin's frame, from libuipc's affine body (the physical state; the drawn
        Genesis link is re-set from it each step and then integrated once more, so it trails slightly)."""
        from genesis.utils.misc import tensor_to_array as A
        if genesis_copy:
            return {k: (A(e.base_link.get_pos()).astype(np.float64), A(e.base_link.get_quat()).astype(np.float64))
                    for k, e in self.bins.items()}
        abd = self.scene.sim.coupler._abd_data_by_link
        out = {}
        for k, e in self.bins.items():
            Tm = np.asarray(abd[e.base_link][0].transform, dtype=np.float64)
            if np.allclose(Tm, np.eye(4)):  # not retrieved yet (before the first step): the link is still at its init pose
                out[k] = (A(e.base_link.get_pos()).astype(np.float64), A(e.base_link.get_quat()).astype(np.float64))
                continue
            U, _, Vt = np.linalg.svd(Tm[:3, :3])  # polar part: drop the affine body's tiny strain
            R = U @ Vt
            # libuipc's body frame is the centre of mass; its local vertices are the mesh's shifted by d
            out[k] = (Tm[:3, 3] + R @ self._bin_offset(k), mat_to_quat(R))
        return out

    def _bin_offset(self, k):
        return self._rigid_offset(("bin", k), self.bins[k], self.bin_mesh_path)

    def _rigid_offset(self, key, entity, mesh_path):
        """Offset d with libuipc local vertex = mesh vertex + d (Genesis may put the geom at -COM in the link),
        measured on the libuipc geometry itself."""
        if key not in self._bin_d:
            import trimesh
            M = np.asarray(trimesh.load(mesh_path, process=False).vertices)
            L = _find_slot(self.scene.sim.coupler, "rigid", entity.base_link.idx).geometry().positions().view().squeeze(-1)
            d = L.min(0) - M.min(0)
            assert np.abs(L.max(0) - M.max(0) - d).max() < 1e-6, "geometry in libuipc is not the mesh shifted"
            self._bin_d[key] = d
        return self._bin_d[key]

    def rigid_pose(self, key, entity, mesh_path):
        """(pos, quat_wxyz) of the mesh frame of a free ipc_only rigid, from libuipc's affine body."""
        from genesis.utils.misc import tensor_to_array as A
        Tm = np.asarray(self.scene.sim.coupler._abd_data_by_link[entity.base_link][0].transform, dtype=np.float64)
        if np.allclose(Tm, np.eye(4)):
            return A(entity.base_link.get_pos()).astype(np.float64), A(entity.base_link.get_quat()).astype(np.float64)
        U, _, Vt = np.linalg.svd(Tm[:3, :3])
        R = U @ Vt
        return Tm[:3, 3] + R @ self._rigid_offset(key, entity, mesh_path), mat_to_quat(R)

    def plug_poses(self):
        return {n: self.rigid_pose(("plug", n), e, plug_info(CABLE_BY_NAME[n]["plug"])["hull"])
                for n, e in self.plugs.items()}

    def success(self, pts=None):
        """Per cable: fraction of its vertices inside each bin's cavity; the bin it is in (> 0.5 of its vertices) and
        whether that is its colour. overall = fraction of cables sorted into the right bin."""
        pts = self.cable_points() if pts is None else pts
        lo, hi = bin_inner_box()
        per = {}
        for n, P in pts.items():
            fr = {}
            for col, (p, q) in self.bin_poses().items():
                R = quat_to_mat(q)
                L = (P - p) @ R
                fr[col] = float(np.all((L > lo) & (L < hi), axis=1).mean())
            best = max(fr, key=fr.get)
            in_bin = best if fr[best] > 0.5 else None
            per[n] = dict(fractions=fr, bin=in_bin, correct=in_bin == CABLE_BY_NAME[n]["color"])
        return dict(per_cable=per, sorted_fraction=float(np.mean([v["correct"] for v in per.values()])))

    def tangle_metric(self, pts=None):
        return tangle_metric(self.cable_points() if pts is None else pts)


def mat_to_quat(R):
    """Rotation matrix -> (w, x, y, z)."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _find_slot(coupler, kind, idx):
    from uipc.backend import SceneVisitor
    from uipc.geometry import SimplicialComplexSlot
    for slot in SceneVisitor(coupler._ipc_scene).geometries():
        if not isinstance(slot, SimplicialComplexSlot):
            continue
        meta = slot.geometry().meta()
        st_ = meta.find("solver_type")
        if st_ is None or st_.view()[0] != kind:
            continue
        key = "entity_idx" if kind == "rod" else "link_idx"
        if int(meta.find(key).view()[0]) == idx:
            return slot
    raise RuntimeError(f"{kind} {idx} not in the IPC scene")


def attach_plug(cable, plug, P, n_inner, info):
    """Hook (before libuipc world.init): SoftVertexStitch cable vertex 0 <-> the plug hull's tip vertex and cable
    vertex n_inner <-> its boot vertex (zero rest length, kappa PL.STITCH_KAPPA), and no contact between the cable
    and its own plug (genesis_cable.py's construction). Returns a dict filled at build."""
    rec = {}

    def hook(coupler):
        from uipc.constitution import SoftVertexStitch
        rod_slot = CM.find_rod_slot(coupler, cable)
        plug_slot = _find_slot(coupler, "rigid", plug.base_link.idx)
        g = plug_slot.geometry()
        Tm = g.transforms().view()[0]
        pv = g.positions().view().squeeze(-1) @ Tm[:3, :3].T + Tm[:3, 3]
        pairs = []
        for i in (0, n_inner):
            j = int(np.argmin(np.linalg.norm(pv - P[i], axis=1)))
            err = float(np.linalg.norm(pv[j] - P[i]))
            assert err < 1e-5, f"no plug vertex at cable vertex {i} ({err:.2e} m)"
            pairs.append([i, j])
        svs = SoftVertexStitch()
        coupler._ipc_constitution_tabular.insert(svs)
        geo = svs.create_geometry((rod_slot, plug_slot), np.array(pairs, dtype=np.int32), kappa=PL.STITCH_KAPPA)
        coupler._ipc_objects.create(f"plug_stitch_{cable.idx}").geometries().create(geo)
        coupler._ipc_contact_tabular.insert(coupler._ipc_rod_contacts[cable], coupler._ipc_abd_contacts[plug],
                                            0.0, 0.0, False)
        rec.update(pairs=pairs, kappa=PL.STITCH_KAPPA)

    CM.add_hook(hook)
    return rec


def build(robot=False, placement=None, show_viewer=False, camera=None, coupling=10.0, krb_only=False, only=None,
          coupler_extra=None, coupling_rot=None):
    """Genesis scene for the station. robot=True adds the wheeled bimanual YAM Ultra cart with IPC-coupled pads.
    camera: dict(res, pos, lookat, fov) for a debug camera (rods are drawn as tubes by the caller).
    Returns Env (env.cam is the camera or None)."""
    import genesis as gs
    os.makedirs(ASSETS, exist_ok=True)
    bin_obj = os.path.join(ASSETS, "bin.obj")
    bin_vis = os.path.join(ASSETS, "bin_visual.obj")
    bin_urdf = os.path.join(ASSETS, "bin.urdf")
    label_obj = os.path.join(ASSETS, "bin_label.obj")
    bin_mesh = make_bin_obj(bin_obj)
    make_bin_urdf(bin_obj, bin_vis, bin_urdf)
    make_label_obj(label_obj)
    kw = {}
    if robot:
        kw["coupler_kw"] = dict(constraint_strength_translation=coupling,
                                constraint_strength_rotation=coupling if coupling_rot is None else coupling_rot)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, -9.81)),
        # libuipc does all contact; Genesis's own rigid collision would double-count it (franka-deformables lesson)
        rigid_options=gs.options.RigidOptions(enable_collision=False),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=CM.D_HAT, contact_resistance=1e7, newton_semi_implicit_enable=False,
            enable_rigid_rigid_contact=True,  # bins on the table, gripper pads vs bins/table
            enable_rigid_ground_contact=True, **kw.get("coupler_kw", {}), **(coupler_extra or {})),
        vis_options=gs.options.VisOptions(show_world_frame=False, background_color=layout.BACKGROUND_RGB,
                                          ambient_light=(0.42, 0.42, 0.42)),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=0.6),
                     surface=gs.surfaces.Default(color=layout.FLOOR_RGB))
    T = layout.TABLE
    scene.add_entity(gs.morphs.Box(pos=(0.0, 0.0, ZT - 0.5 * T["top_thickness"]),
                                   size=(T["size_x"], T["size_y"], T["top_thickness"]), fixed=True),
                     material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=0.5),
                     surface=gs.surfaces.Default(color=(0.78, 0.66, 0.50)))
    bins, labels = {}, {}
    rho_bin = BIN["mass"] / bin_mesh.volume
    for col, bspec in BINS.items():
        bins[col] = scene.add_entity(
            gs.morphs.URDF(file=bin_urdf, pos=(bspec["xy"][0], bspec["xy"][1], ZT + BIN_LIFT), fixed=False,
                           convexify=False, decimate=False, watertighten=0, align=False),
            material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=BIN["friction"], rho=rho_bin),
            surface=gs.surfaces.Default(color=BIN_TINT[col]))
        labels[col] = label_obj
    cables, plugs = {}, {}
    lay = initial_layout(placement)
    for c in CABLES:
        if only is not None and c["name"] not in only:
            continue
        L = lay[c["name"]]
        ni = L["n_inner"]
        hs = np.ones(len(L["P"]) - 2)
        hs[:max(0, ni - 1)] = PL.INNER_STIFFEN  # hinges 1..n_inner-1: the section moulded into the plug
        fr = CM.Framed(L["P"], c["radius"], c["EI"], hinge_scale=hs if ni else None)
        h = CM.add_framed_cable(scene, gs, fr, c["rho_line"], c["friction"], COLORS[c["color"]], krb_only=krb_only)
        cables[c["name"]] = dict(entity=h["cable"], ghosts=h["ghosts"], framed=fr, info=h["info"], spec=c, n_inner=ni)
        if ni:
            info = plug_info(c["plug"])
            plugs[c["name"]] = scene.add_entity(
                gs.morphs.URDF(file=info["urdf"], pos=tuple(L["plug_t"]), quat=tuple(mat_to_quat(L["plug_R"])),
                               fixed=False, convexify=False, decimate=False, watertighten=0, align=False),
                material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=0.6, rho=info["mass"] / info["volume"]),
                surface=gs.surfaces.Default(color=PL.PLUGS[c["plug"]]["color"]))
            cables[c["name"]]["plug_info"] = attach_plug(h["cable"], plugs[c["name"]], L["P"], ni, info)
    cart = None
    if robot:
        from yam_cart import add_cart
        cart = add_cart(scene, gs, layout.STATIONS["cables"]["arm"], ipc=True)
    cam = None
    if camera is not None:
        cam = scene.add_camera(res=camera["res"], pos=camera["pos"], lookat=camera["lookat"], fov=camera["fov"],
                               GUI=False, debug=True)
    scene.build()
    if cart is not None:
        cart.after_build()
    env = Env(scene, cables, bins, labels, cart, gs)
    env.plugs = plugs
    env.cam = cam
    env.bin_mesh_path, env.bin_visual_path, env.label_mesh_path = bin_obj, bin_vis, label_obj
    return env
