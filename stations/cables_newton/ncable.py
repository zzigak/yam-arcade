"""Newton model of the cables station: cables as VBD rods (capsule chains + cable joints) with a rigid plug on one end,
a fixed table top, two free open-top bins, and helpers that read the state back (centrelines, plug poses, contact
penetrations) and measure it (rest-shape retention, tangle metric).

Specs (cables, rest shapes, bins, drop layout) come from spec_genesis.json, a snapshot of the Genesis station
(stations/cables/station.py, read-only; see snapshot_spec.py), so both versions simulate the same objects.

Rest shape: SolverVBD measures each cable joint's bend/twist as the rotation of the child frame relative to the parent
frame *relative to their rest relative rotation*, and the rest pose is `model.body_q` at solver creation
(rigid_vbd_kernels.compute_kappa: q_align = q_rel * inverse(q_rel_rest)). A rod built along a curved polyline is
therefore stress-free in that curved shape. Body frames come from parallel transport along the rest curve, so the rest
twist is zero along the cable.

    import ncable as NC
    b = newton.ModelBuilder(gravity=-9.81); NC.register(b)
    NC.add_table(b); bins = NC.add_bins(b); cab = NC.add_cable(b, NC.CABLE_BY_NAME["hdmi_ubend"], P, scale)
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import layout  # noqa: E402

os.environ.setdefault("PYGLET_HEADLESS", "true")
sys.path.insert(0, os.environ.get("NEWTON_ROOT", os.path.expanduser("~/sims/newton")))
import warp as wp  # noqa: E402
import newton  # noqa: E402
from newton import ModelBuilder  # noqa: E402
from newton.solvers import SolverVBD  # noqa: E402

SPEC = json.load(open(os.path.join(HERE, "spec_genesis.json")))
CABLES = SPEC["cables"]
CABLE_BY_NAME = {c["name"]: c for c in CABLES}
COLORS = {k: tuple(v) for k, v in SPEC["colors"].items()}
BIN_TINT = {k: tuple(v) for k, v in SPEC["bin_tint"].items()}
BIN = SPEC["bin"]
BINS = SPEC["bins"]
ZT = layout.TABLE_TOP_Z
ASSETS = os.path.join(HERE, "assets")
CALIB_JSON = os.path.join(HERE, "calib_bend.json")

# ------------------------------------------------------------------------------------------------------ materials
TABLE_KE, TABLE_MU = 1.0e5, 0.5  # wood top (same friction as the Genesis table)
BIN_KE, BIN_MU = 1.0e5, 0.5  # polypropylene bin (Genesis BIN friction 0.5)
CABLE_KE = 1.0e4  # PVC jacket; a 1 g segment resting on the table sinks 1 um
PLUG_KE, PLUG_MU = 2.0e4, 0.8  # moulded PVC/TPE overmould: a 8 N squeeze is ~0.4 mm (pad and plug ke averaged)
CONTACT_GAP = 0.004  # m, contact detection margin (speculative contacts: a 2 m/s cable moves 3.3 mm per 600 Hz substep)
STRETCH_KE = 2.0e3  # N/m per joint; stretch slots are hard (augmented Lagrangian): this only sets conditioning
BEND_TAU = 0.02  # s, bend/twist damping = BEND_TAU * bend stiffness (jacket hysteresis; franka calibration value)
BOOT_EI_FACTOR = 3.0  # the plug's strain-relief boot is ~3x stiffer than the cable it holds

# ---------------------------------------------------------------------------------------------------------- plugs
# Plug-local frame: origin on the cable axis at the boot end (where the cable leaves the plug), +x towards the tip,
# z = the plug's thin direction. Parts: ("box", centre, half extents, material) or ("cap", x0, x1, radius, y,
# material) = a capsule along x spanning x0..x1 (caps included). Real-world sizes; masses from typical moulded plugs.
PLUGS = {
    "power_coil": dict(kind="IEC C13 socket", mass=0.030, grip=("box", 0.040), parts=[
        ("cap", 0.0, 0.024, 0.0065, 0.0, "body"),
        ("box", (0.041, 0.0, 0.0), (0.019, 0.0165, 0.0110), "body")]),
    "hdmi_ubend": dict(kind="HDMI type-A plug", mass=0.010, grip=("box", 0.024), parts=[
        ("cap", 0.0, 0.014, 0.0055, 0.0, "body"),
        ("box", (0.024, 0.0, 0.0), (0.014, 0.0105, 0.0055), "body"),
        ("box", (0.0435, 0.0, 0.0), (0.0055, 0.0070, 0.00225), "metal")]),
    "ext_coil": dict(kind="Europlug (CEE 7/16)", mass=0.020, grip=("box", 0.022), parts=[
        ("cap", 0.0, 0.012, 0.0050, 0.0, "body"),
        ("box", (0.022, 0.0, 0.0), (0.018, 0.0175, 0.0075), "body"),
        ("cap", 0.040, 0.059, 0.0020, 0.0095, "metal"),
        ("cap", 0.040, 0.059, 0.0020, -0.0095, "metal")]),
    "cat6_scurve": dict(kind="RJ45 plug with boot", mass=0.006, grip=("box", 0.029), parts=[
        ("cap", 0.0, 0.020, 0.0050, 0.0, "body"),
        ("box", (0.029, 0.0, 0.0), (0.011, 0.0059, 0.0068), "clear")]),
    "barrel_arc": dict(kind="DC barrel plug 5.5 x 2.1", mass=0.004, grip=("cap", 0.012), parts=[
        ("cap", 0.0, 0.025, 0.0045, 0.0, "body"),
        ("cap", 0.023, 0.034, 0.00275, 0.0, "metal")]),
    "usbc_wave": dict(kind="USB-C plug", mass=0.004, grip=("box", 0.014), parts=[
        ("box", (0.013, 0.0, 0.0), (0.012, 0.0060, 0.00325), "body"),
        ("box", (0.0285, 0.0, 0.0), (0.0035, 0.0042, 0.00125), "metal")]),
    "micro_scurve": dict(kind="micro-USB plug", mass=0.003, grip=("box", 0.011), parts=[
        ("box", (0.011, 0.0, 0.0), (0.010, 0.0055, 0.0030), "body"),
        ("box", (0.0245, 0.0, 0.0), (0.0035, 0.0035, 0.0009), "metal")]),
    "aux_loop": dict(kind="3.5 mm TRS jack plug", mass=0.004, grip=("cap", 0.011), parts=[
        ("cap", 0.0, 0.023, 0.0040, 0.0, "body"),
        ("cap", 0.021, 0.036, 0.00175, 0.0, "metal")]),
}
PLUG_INNER = 0.6  # the cable runs this fraction of the overmould length into the plug (body 0's hidden capsule)
METAL_RGB = (0.72, 0.73, 0.75)
CLEAR_RGB = (0.85, 0.87, 0.88)


def plug_len(name):
    """Overmould length (plastic parts' x extent)."""
    xs = []
    for p in PLUGS[name]["parts"]:
        if p[-1] == "metal":
            continue
        xs += [p[1][0] + p[2][0]] if p[0] == "box" else [p[2]]
    return max(xs)


def plug_extent(name):
    """Plug-local AABB (lo, hi) of every part."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for p in PLUGS[name]["parts"]:
        if p[0] == "box":
            c, h = np.array(p[1]), np.array(p[2])
            lo, hi = np.minimum(lo, c - h), np.maximum(hi, c + h)
        else:
            _, x0, x1, r, y, _ = p
            lo = np.minimum(lo, [x0, y - r, -r])
            hi = np.maximum(hi, [x1, y + r, r])
    return lo, hi


def plug_points(name, n_cap=12):
    """Surface sample points of the plug's parts in the plug frame (for clearance and penetration checks)."""
    pts = []
    for p in PLUGS[name]["parts"]:
        if p[0] == "box":
            c, h = np.array(p[1]), np.array(p[2])
            g = np.linspace(-1, 1, 5)
            for a in g:
                for b_ in g:
                    for ax in range(3):
                        for s in (-1, 1):
                            v = np.zeros(3)
                            v[ax] = s
                            o = [i for i in range(3) if i != ax]
                            v[o[0]], v[o[1]] = a, b_
                            pts.append(c + v * h)
        else:
            _, x0, x1, r, y, _ = p
            for x in np.linspace(x0 + r, x1 - r, 4):
                for t in np.linspace(0, 2 * np.pi, n_cap, endpoint=False):
                    pts.append([x, y + r * np.cos(t), r * np.sin(t)])
            for s, xc in ((-1, x0 + r), (1, x1 - r)):
                pts.append([xc + s * r, y, 0.0])
    return np.unique(np.round(np.array(pts), 7), axis=0)


def plug_meshes(name):
    """{material: trimesh} of the plug in its frame (for export and previews)."""
    import trimesh
    out = {}
    for p in PLUGS[name]["parts"]:
        if p[0] == "box":
            m = trimesh.creation.box(extents=2 * np.array(p[2]))
            m.apply_translation(p[1])
            mat = p[3]
        else:
            _, x0, x1, r, y, mat = p
            m = trimesh.creation.capsule(height=(x1 - x0) - 2 * r, radius=r, count=[16, 16])
            m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
            m.apply_translation([(x0 + x1) / 2, y, 0.0])
        out.setdefault(mat, []).append(m)
    return {k: trimesh.util.concatenate(v) for k, v in out.items()}


# ------------------------------------------------------------------------------------------------------- helpers
def quat_to_mat_xyzw(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat_xyzw(R):
    t = np.trace(R)
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        q = [(R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s, 0.25 / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[3] = (R[k, j] - R[j, k]) / s
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def tf_mat(t):
    """4x4 from a Newton transform [px py pz qx qy qz qw]."""
    t = np.asarray(t, float)
    M = np.eye(4)
    M[:3, :3] = quat_to_mat_xyzw(t[3:7])
    M[:3, 3] = t[:3]
    return M


def mat_tf(M):
    q = mat_to_quat_xyzw(M[:3, :3])
    return wp.transform(wp.vec3(*map(float, M[:3, 3])), wp.quat(*map(float, q)))


def xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def capsule_volume(r, L):
    return math.pi * r * r * L + 4.0 / 3.0 * math.pi * r ** 3


def register(b):
    """Builder-level settings every scene uses: VBD custom attributes, elastic (non-hysteretic) cable bending."""
    SolverVBD.register_custom_attributes(b, dahl_defaults_enabled=False)
    b.rigid_gap = CONTACT_GAP


def bend_scales():
    """Per-cable calibrated bend scale (calib_bend.json), keyed by (substeps, iterations)."""
    if not os.path.exists(CALIB_JSON):
        return {}
    return json.load(open(CALIB_JSON))


# --------------------------------------------------------------------------------------------------------- scene
def add_table(b):
    """Fixed table top (z = 0.75 at its surface, layout extents) and the floor."""
    T = layout.TABLE
    cfg = ModelBuilder.ShapeConfig(ke=TABLE_KE, kd=0.0, mu=TABLE_MU, margin=0.0, gap=CONTACT_GAP)
    s_table = b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, ZT - 0.5 * T["top_thickness"]), wp.quat_identity()),
                              hx=0.5 * T["size_x"], hy=0.5 * T["size_y"], hz=0.5 * T["top_thickness"], cfg=cfg,
                              color=(0.78, 0.66, 0.50), label="table_top")
    s_floor = b.add_ground_plane(cfg=ModelBuilder.ShapeConfig(ke=TABLE_KE, kd=0.0, mu=0.6, margin=0.0, gap=CONTACT_GAP))
    return s_table, s_floor


def bin_boxes():
    """The bin as five boxes in its frame (outer bottom centre): [(centre, half extents)]. Same solid as the Genesis
    bin mesh (stations/cables/station.make_bin_obj): floor under the whole footprint, walls on top of it."""
    a, bh = 0.5 * BIN["size"][0], 0.5 * BIN["size"][1]
    H, t, tb = BIN["size"][2], BIN["wall"], BIN["floor"]
    hw = 0.5 * (H - tb)
    zc = tb + hw
    return [((0.0, 0.0, 0.5 * tb), (a, bh, 0.5 * tb)),
            ((0.0, bh - 0.5 * t, zc), (a, 0.5 * t, hw)), ((0.0, -bh + 0.5 * t, zc), (a, 0.5 * t, hw)),
            ((a - 0.5 * t, 0.0, zc), (0.5 * t, bh - t, hw)), ((-a + 0.5 * t, 0.0, zc), (0.5 * t, bh - t, hw))]


def box_sdf(X, c, h):
    """Signed distance from points X to an axis-aligned box (centre c, half extents h)."""
    q = np.abs(X - np.asarray(c)) - np.asarray(h)
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(1), 0.0)


def bin_sdf(X, col, lift=0.0005):
    """Signed distance from world points to the (unrotated, initial) bin of colour col."""
    o = np.array([BINS[col]["xy"][0], BINS[col]["xy"][1], ZT + lift])
    return np.min([box_sdf(X - o, c, h) for c, h in bin_boxes()], axis=0)


def bin_inner_box():
    a, bh = 0.5 * BIN["size"][0] - BIN["wall"], 0.5 * BIN["size"][1] - BIN["wall"]
    return np.array([-a, -bh, BIN["floor"]]), np.array([a, bh, BIN["size"][2]])


def bin_mesh():
    import trimesh
    ms = []
    for c, h in bin_boxes():
        m = trimesh.creation.box(extents=2 * np.array(h))
        m.apply_translation(c)
        ms.append(m)
    return trimesh.util.concatenate(ms)


def label_mesh():
    """Colour label sticker on the bin's front (-y) face, 0.5 mm proud (as in the Genesis station)."""
    import trimesh
    bh, H = 0.5 * BIN["size"][1], BIN["size"][2]
    m = trimesh.creation.box(extents=(0.12, 0.0005, 0.045))
    m.apply_translation((0.0, -bh - 0.00025, 0.55 * H))
    return m


def add_bins(b, lift=0.0005):
    """Two free open-top bins (0.30 kg each) resting on the table, frame at the outer bottom centre."""
    vol = sum(8 * h[0] * h[1] * h[2] for _, h in bin_boxes())
    cfg = ModelBuilder.ShapeConfig(density=BIN["mass"] / vol, ke=BIN_KE, kd=0.0, mu=BIN_MU, margin=0.0, gap=CONTACT_GAP)
    out = {}
    for col, bs in BINS.items():
        body = b.add_body(xform=wp.transform((bs["xy"][0], bs["xy"][1], ZT + lift), wp.quat_identity()),
                          label=f"bin_{col}")
        shapes = [b.add_shape_box(body, xform=wp.transform(c, wp.quat_identity()), hx=h[0], hy=h[1], hz=h[2], cfg=cfg,
                                  color=BIN_TINT[col], label=f"bin_{col}_{k}") for k, (c, h) in enumerate(bin_boxes())]
        out[col] = dict(body=body, shapes=shapes)
    return out


def rod_frames(nodes):
    """Parallel-transport capsule frames (local +Z along each segment) for the rest polyline."""
    return newton.utils.create_parallel_transport_cable_quaternions([wp.vec3(*map(float, p)) for p in nodes])


def add_cable(b, c, P, bend_scale=1.0, plug=True, color=None, ke=CABLE_KE):
    """Add cable spec `c` along world polyline P (its rest shape). With plug=True a rigid plug is fixed on the cable's
    first node: an extra rod body (hidden, non-colliding capsule inside the plug) carrying the plug's shapes, joined to
    the cable by a cable joint with the boot's stiffness. Returns a dict with bodies, joints, shapes and geometry."""
    P = np.asarray(P, float)
    r, EI = c["radius"], c["EI"]
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    nodes = P
    L_in = 0.0
    if plug:
        L_in = PLUG_INNER * plug_len(c["name"])
        t0 = (P[0] - P[1]) / np.linalg.norm(P[0] - P[1])  # from the cable into the plug
        nodes = np.vstack([P[0] + L_in * t0, P])
    quats = rod_frames(nodes)
    lseg = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    rho_seg = c["rho_line"] * seg.mean() / capsule_volume(r, seg.mean())
    ccfg = ModelBuilder.ShapeConfig(density=float(rho_seg), ke=ke, kd=0.0, mu=c["friction"], margin=0.0,
                                    gap=CONTACT_GAP)
    rgb = COLORS[c["color"]] if color is None else color
    k_ref = bend_scale * EI / seg.mean()
    bodies, joints = b.add_rod(positions=[wp.vec3(*map(float, p)) for p in nodes], quaternions=quats, radius=r,
                               cfg=ccfg, stretch_stiffness=STRETCH_KE, stretch_damping=0.0, bend_stiffness=k_ref,
                               bend_damping=BEND_TAU * k_ref, label=c["name"], body_frame_origin="com", color=rgb)
    # per-joint bend stiffness: EI over the joint's Voronoi length (segments are equal apart from the plug's)
    for j_i, j in enumerate(joints):
        lj = 0.5 * (lseg[j_i] + lseg[j_i + 1])
        ei = EI * (BOOT_EI_FACTOR if (plug and j_i == 0) else 1.0)
        k = bend_scale * ei / lj
        dof = b.joint_qd_start[j] + 1
        b.joint_target_ke[dof] = k
        b.joint_target_kd[dof] = BEND_TAU * k
    out = dict(name=c["name"], spec=c, bodies=list(bodies), joints=list(joints), nodes0=nodes, P0=P, seg=seg,
               lseg=lseg, plug=plug, L_in=L_in, quats=quats, cable_shapes=[])
    for bi in bodies:
        out["cable_shapes"] += list(b.body_shapes[bi])
    if plug:
        pb = bodies[0]
        s_in = b.body_shapes[pb][0]
        # the hidden capsule inside the plug: keeps its mass (the cable inside the overmould), no collision, not drawn
        b.shape_flags[s_in] = b.shape_flags[s_in] & ~int(newton.ShapeFlags.COLLIDE_SHAPES) \
            & ~int(newton.ShapeFlags.COLLIDE_PARTICLES) & ~int(newton.ShapeFlags.VISIBLE)
        out["cable_shapes"].remove(s_in)
        m_in = rho_seg * capsule_volume(r, lseg[0])
        # plug frame in the world: x into the plug, z as close to world up as possible
        x = (nodes[0] - nodes[1]) / np.linalg.norm(nodes[0] - nodes[1])
        up = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        z = up - up.dot(x) * x
        z /= np.linalg.norm(z)
        Pw = np.eye(4)
        Pw[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
        Pw[:3, 3] = nodes[1]
        B0 = np.eye(4)
        B0[:3, :3] = quat_to_mat_xyzw([quats[0][0], quats[0][1], quats[0][2], quats[0][3]])
        B0[:3, 3] = 0.5 * (nodes[0] + nodes[1])
        plug_local = np.linalg.inv(B0) @ Pw
        spec = PLUGS[c["name"]]
        vol = 0.0
        for p in spec["parts"]:
            vol += 8 * np.prod(p[2]) if p[0] == "box" else capsule_volume(p[3], (p[2] - p[1]) - 2 * p[3])
        dens = (spec["mass"] - m_in) / vol
        assert dens > 0, f"{c['name']}: plug mass below the inner cable mass"
        pcfg = ModelBuilder.ShapeConfig(density=float(dens), ke=PLUG_KE, kd=0.0, mu=PLUG_MU, margin=0.0,
                                        gap=CONTACT_GAP)
        shapes, mats = [], []
        for k, p in enumerate(spec["parts"]):
            mats.append(p[-1])
            col = METAL_RGB if p[-1] == "metal" else (CLEAR_RGB if p[-1] == "clear" else rgb)
            if p[0] == "box":
                L = np.eye(4)
                L[:3, 3] = p[1]
                shapes.append(b.add_shape_box(pb, xform=mat_tf(plug_local @ L), hx=p[2][0], hy=p[2][1], hz=p[2][2],
                                              cfg=pcfg, color=col, label=f"{c['name']}_plug_{k}"))
            else:
                _, x0, x1, rr, y, _ = p
                L = np.eye(4)
                L[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)  # capsule +Z -> plug +x
                L[:3, 3] = [(x0 + x1) / 2, y, 0.0]
                shapes.append(b.add_shape_capsule(pb, xform=mat_tf(plug_local @ L), radius=rr,
                                                  half_height=0.5 * (x1 - x0) - rr, cfg=pcfg, color=col,
                                                  label=f"{c['name']}_plug_{k}"))
        out.update(plug_body=pb, plug_shapes=shapes, plug_mats=mats, plug_local=plug_local, plug_world0=Pw)
    return out


def set_hard_stretch(solver, joints):
    """Cable stretch slots -> hard (augmented Lagrangian), in one array update (franka calibration: a soft stretch
    penalty much stiffer than the bend springs leaves block Gauss-Seidel unconverged and the rod sags)."""
    c_start = solver.joint_constraint_start.numpy()
    is_hard = solver.joint_is_hard.numpy()
    for j in joints:
        is_hard[c_start[j] + SolverVBD.JointSlot.STRETCH] = 1
    solver.joint_is_hard = wp.array(is_hard, dtype=wp.int32, device=solver.device)


def make_kinematic(b, bodies):
    for i in bodies:
        b.body_mass[i] = 0.0
        b.body_inv_mass[i] = 0.0
        b.body_inertia[i] = wp.mat33(0.0)
        b.body_inv_inertia[i] = wp.mat33(0.0)
        b.body_flags[i] = int(newton.BodyFlags.KINEMATIC)


# --------------------------------------------------------------------------------------------------- read back
def cable_nodes(bq, cab):
    """Centreline nodes (cable only, plug's inner segment excluded) from body transforms: joint midpoints between
    consecutive capsule ends, plus the two free ends. Also returns the joint gaps (m)."""
    T = bq[cab["bodies"]]
    half = 0.5 * cab["lseg"]
    R = np.stack([quat_to_mat_xyzw(t[3:7]) for t in T])
    z = R[:, :, 2]
    a = T[:, :3] - half[:, None] * z
    e = T[:, :3] + half[:, None] * z
    gaps = np.linalg.norm(a[1:] - e[:-1], axis=1)
    nodes = np.vstack([a[:1], 0.5 * (e[:-1] + a[1:]), e[-1:]])
    if cab["plug"]:
        nodes = nodes[1:]
    return nodes, gaps


def plug_pose(bq, cab):
    """4x4 world pose of the plug frame."""
    return tf_mat(bq[cab["plug_body"]]) @ cab["plug_local"]


# ---------------------------------------------------------------------------------------------------- metrics
def kabsch_rmsd(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return float(np.sqrt(((Xc @ R - Yc) ** 2).sum(1).mean()))


def straight_rmsd(P):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return kabsch_rmsd(np.stack([s, 0 * s, 0 * s], 1), P)


def turning_angles(P):
    e = np.diff(P, axis=0)
    a, b_ = e[:-1], e[1:]
    c = np.einsum("ij,ij->i", a, b_) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b_, axis=1))
    return np.arccos(np.clip(c, -1.0, 1.0))


def retention(X, P):
    """Same measures as the Genesis station: 1 - RMSD(X, rest) / RMSD(straight, rest), summed-curvature ratio."""
    rm = kabsch_rmsd(X, P)
    th0, th = turning_angles(P), turning_angles(X)
    return dict(rmsd_mm=1e3 * rm, straight_rmsd_mm=1e3 * straight_rmsd(P), retention=1.0 - rm / straight_rmsd(P),
                curvature_ratio=float(th.sum() / th0.sum()), curvature_corr=float(np.corrcoef(th0, th)[0, 1]))


def seg_seg_dist(P, Q):
    """Minimum distance between every segment of polyline P and every segment of polyline Q: (m-1, k-1) array.
    (Copied from stations/cables/station.py.)"""
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


def _seg_intersections_2d(P, Q):
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
    """Tangle measures on {name: (n, 3)} (same definitions as stations/cables/station.tangle_metric):
    crossings (top-view crossings between different cables), self_crossings, contacts (segment pairs of different
    cables within contact_tol), touching_pairs, interlocked_pairs (A over B somewhere and under it elsewhere),
    middle (cables lying over one cable and under another)."""
    names = list(pts)
    crossings, contacts, touching, inter = 0, 0, [], []
    over = {n: set() for n in names}
    under = {n: set() for n in names}
    for i, a in enumerate(names):
        for bn in names[i + 1:]:
            P, Q = pts[a], pts[bn]
            xs = _seg_intersections_2d(P, Q)
            sgn = []
            for ii, jj, s, t in xs:
                za = P[ii, 2] + s * (P[ii + 1, 2] - P[ii, 2])
                zb = Q[jj, 2] + t * (Q[jj + 1, 2] - Q[jj, 2])
                sgn.append(np.sign(za - zb))
                (over[a].add(bn), under[bn].add(a)) if za > zb else (over[bn].add(a), under[a].add(bn))
            crossings += len(xs)
            if len(set(sgn)) == 2:
                inter.append(f"{a}|{bn}")
            d = seg_seg_dist(P, Q) - CABLE_BY_NAME[a]["radius"] - CABLE_BY_NAME[bn]["radius"]
            nc = int((d < contact_tol).sum())
            contacts += nc
            if nc:
                touching.append(f"{a}|{bn}")
    selfx = 0
    for n in names:
        P = pts[n]
        for ii, jj, s, t in _seg_intersections_2d(P, P):
            if jj > ii + 1:
                selfx += 1
    middle = [n for n in names if over[n] and under[n]]
    return dict(crossings=crossings, self_crossings=selfx, contacts=contacts, touching_pairs=len(touching),
                interlocked_pairs=inter, middle=middle)


def min_cable_gaps(pts):
    """Smallest surface gap between cables (pairs) and within a cable (segments >= 3 apart), m (negative = overlap)."""
    names = list(pts)
    out = {}
    for i, a in enumerate(names):
        for bn in names[i + 1:]:
            out[f"{a}|{bn}"] = float(seg_seg_dist(pts[a], pts[bn]).min() - CABLE_BY_NAME[a]["radius"]
                                     - CABLE_BY_NAME[bn]["radius"])
        D = seg_seg_dist(pts[a], pts[a])
        n = len(D)
        idx = np.arange(n)
        D[np.abs(idx[:, None] - idx[None, :]) < 3] = np.inf
        out[f"{a}|self"] = float(D.min() - 2 * CABLE_BY_NAME[a]["radius"])
    return out
