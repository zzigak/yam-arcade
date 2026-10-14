"""Procedural glass jars and screw lids with real helical threads (jars station).

Every part is a lathe (a closed r-z profile revolved about +z) unioned with a swept thread ridge (a trapezoid
axial profile swept along a helix, its ends tapered down into the wall so they run out like real threads). The
union is done with manifold3d, so each output mesh is a single closed, consistently oriented 2-manifold.

Two meshes per part:
  *_col.obj  collision mesh for libuipc (120 segments per turn, trapezoid thread)
  *_vis.obj  render mesh (240+ segments, chamfered thread crests, knurled/ribbed lid rims). It stays within 0.1 mm of
             the collision surface (finer fillets poke out <= 0.09 mm; ribs/knurls are cut inward).

Frames (metres in the OBJ files, mm inside this script):
  jar: origin on the axis at the plane of the base ring (the jar stands on z = 0), +z up.
  lid: origin on the axis at the SEAT PLANE (the underside of the top panel that lands on the jar rim), +z towards
       the top panel. A lid seated on its jar has lid-frame origin = (0, 0, rim_z) in the jar frame.

Run: ~/projects/yam-arcade/.venvs/jars/bin/python stations/jars/geometry.py
Writes stations/jars/assets/*.obj and stations/jars/assets/params.json (read by station.py).
"""
import json
import math
import os
from dataclasses import asdict, dataclass

import manifold3d as m3d
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

FLANK_DEG = 30.0  # thread flank half-angle (60 deg included, like a GCMI 'M' / ISO profile)
EMBED = 0.5  # mm, thread base sunk into the wall so the union is clean
TAPER_DEG = 45.0  # run-out at each thread end
C_RADIAL = 0.5  # mm, radial clearance crest-to-root on both parts
G_AXIAL = 0.4  # mm, axial clearance per flank with the lid centred in the groove (backlash 0.8 mm)


@dataclass
class JarSpec:
    name: str
    title: str
    # body (mm)
    r_body: float
    rim_z: float
    t_wall: float
    t_base: float
    z_body_top: float  # where the straight body wall ends and the shoulder begins
    z_neck_root: float  # where the shoulder meets the neck
    r_root: float  # neck (thread root) radius
    t_neck: float
    # thread
    pitch: float
    h: float  # thread depth
    w_crest: float  # jar thread crest width
    turns_jar: float
    a_top: float  # jar thread top-end centre below the rim
    # lid
    lid_kind: str  # 'plastic' or 'metal'
    lid_color: tuple
    t_skirt: float
    l_skirt: float  # skirt length below the seat plane
    t_top: float
    turns_lid: float
    lid_mass_g: float  # target lid mass (effective density is derived)
    lid_style: str  # 'ribbed', 'smooth', 'knurled'
    n_ribs: int
    rib_depth: float
    bead_z: float = -1.0  # mason-style bead on the neck (centre z), <0 = none
    bead_r: float = 0.9


SPECS = [
    # Small spice jar, 43 mm finish, red polypropylene cap with ribbed rim.
    JarSpec(name="spice", title="spice jar", r_body=24.0, rim_z=100.0, t_wall=2.5, t_base=5.0, z_body_top=80.0,
            z_neck_root=86.0, r_root=20.5, t_neck=2.4, pitch=3.2, h=1.0, w_crest=0.5, turns_jar=1.75, a_top=2.2,
            lid_kind="plastic", lid_color=(0.78, 0.10, 0.08), t_skirt=1.6, l_skirt=10.0, t_top=1.6, turns_lid=1.5,
            lid_mass_g=6.0, lid_style="ribbed", n_ribs=90, rib_depth=0.45),
    # Jam jar, 63 mm finish, gold-lacquered tinplate lid.
    JarSpec(name="jam", title="jam jar", r_body=35.0, rim_z=98.0, t_wall=3.0, t_base=6.0, z_body_top=78.0,
            z_neck_root=85.0, r_root=30.3, t_neck=2.8, pitch=3.6, h=1.2, w_crest=0.6, turns_jar=1.5, a_top=2.4,
            lid_kind="metal", lid_color=(0.83, 0.66, 0.30), t_skirt=1.2, l_skirt=10.5, t_top=1.2, turns_lid=1.5,
            lid_mass_g=13.0, lid_style="smooth", n_ribs=0, rib_depth=0.0),
    # Regular-mouth pint mason jar, 70 mm finish, 6 TPI, one-piece zinc-plated steel lid with knurled band.
    JarSpec(name="mason", title="mason jar", r_body=42.0, rim_z=128.0, t_wall=3.5, t_base=6.0, z_body_top=98.0,
            z_neck_root=110.0, r_root=33.8, t_neck=3.0, pitch=4.23, h=1.4, w_crest=0.7, turns_jar=1.75, a_top=2.8,
            lid_kind="metal", lid_color=(0.74, 0.75, 0.77), t_skirt=1.3, l_skirt=12.5, t_top=1.3, turns_lid=1.5,
            lid_mass_g=20.0, lid_style="knurled", n_ribs=150, rib_depth=0.3, bead_z=113.0),
]

GLASS_RHO = 2500.0  # kg/m^3 soda-lime glass


# ----------------------------------------------------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------------------------------------------------
def lathe(profile, n_seg, theta0=0.0, ring_mod=None):
    """Revolve a closed r-z polygon about +z. Points with r == 0 become poles.

    ring_mod(i_profile, theta, r, z) -> r' optionally displaces ring vertices (ribs/knurls)."""
    verts, rings = [], []
    th = theta0 + 2 * np.pi * np.arange(n_seg) / n_seg
    for i, (r, z) in enumerate(profile):
        if r <= 1e-12:
            rings.append([len(verts)])
            verts.append((0.0, 0.0, z))
        else:
            rr = np.full(n_seg, r) if ring_mod is None else ring_mod(i, th, r, z)
            idx = list(range(len(verts), len(verts) + n_seg))
            verts.extend(zip(rr * np.cos(th), rr * np.sin(th), np.full(n_seg, z)))
            rings.append(idx)
    faces = []
    n = len(profile)
    for i in range(n):
        a, b = rings[i], rings[(i + 1) % n]
        if len(a) == 1 and len(b) == 1:
            continue  # axis segment
        for k in range(n_seg):
            k1 = (k + 1) % n_seg
            if len(a) == 1:
                faces.append((a[0], b[k], b[k1]))
            elif len(b) == 1:
                faces.append((a[k], b[0], a[k1]))
            else:
                faces.append((a[k], b[k], b[k1]))
                faces.append((a[k], b[k1], a[k1]))
    return np.array(verts, float), np.array(faces, np.int64)


def sweep_thread(r_ref, sign, pitch, z_start, theta_start, turns, prof, spt, taper_deg=TAPER_DEG, embed=EMBED):
    """Sweep an axial thread profile along a right-handed helix.

    r_ref: thread root radius; sign +1 = external thread (ridge points outward), -1 = internal (points inward).
    prof: list of (d, u): d radial height above the root (towards the crest), u axial offset from the helix centre.
    The helix centre is z = z_start + pitch * phi / 2pi at angle theta_start + phi, phi in [0, 2pi turns].
    """
    n = max(8, int(round(turns * spt)))
    phi = np.linspace(0.0, 2 * np.pi * turns, n + 1)
    taper = np.radians(taper_deg)
    hmax = max(d for d, _ in prof)
    s0 = 0.5 * embed / (hmax + embed)  # at the very ends the crest sits embed/2 inside the wall
    ramp = np.clip(np.minimum(phi, phi[-1] - phi) / taper, 0.0, 1.0)
    s = s0 + (1.0 - s0) * (3 * ramp**2 - 2 * ramp**3)
    m = len(prof)
    verts = []
    for k in range(n + 1):
        th = theta_start + phi[k]
        zc = z_start + pitch * phi[k] / (2 * np.pi)
        for d, u in prof:
            dd = -embed + (d + embed) * s[k]
            r = r_ref + sign * dd
            verts.append((r * math.cos(th), r * math.sin(th), zc + u))
    faces = []
    for k in range(n):
        for j in range(m):
            j1 = (j + 1) % m
            a, b, c, d = k * m + j, k * m + j1, (k + 1) * m + j1, (k + 1) * m + j
            faces += [(a, b, c), (a, c, d)]
    # end caps (profile polygon is convex)
    last = n * m
    for j in range(1, m - 1):
        faces.append((0, j + 1, j))
        faces.append((last, last + j, last + j + 1))
    return np.array(verts, float), np.array(faces, np.int64)


def to_manifold(v, f):
    tm = trimesh.Trimesh(v, f, process=False)
    if tm.volume < 0:
        f = f[:, ::-1]
    mesh = m3d.Mesh(vert_properties=np.ascontiguousarray(v, np.float32), tri_verts=np.ascontiguousarray(f, np.uint32))
    man = m3d.Manifold(mesh)
    assert man.status() == m3d.Error.NoError, man.status()
    return man


def from_manifold(man):
    mesh = man.to_mesh()
    return np.asarray(mesh.vert_properties[:, :3], float), np.asarray(mesh.tri_verts, np.int64)


def arc(cx, cz, rad, a0, a1, n):
    a = np.linspace(np.radians(a0), np.radians(a1), n + 1)
    return [(cx + rad * math.cos(t), cz + rad * math.sin(t)) for t in a]


def smoothstep(t):
    return 3 * t**2 - 2 * t**3


# ----------------------------------------------------------------------------------------------------------------
# profiles
# ----------------------------------------------------------------------------------------------------------------
def assert_simple(profile, what):
    """The r-z lathe profile must be a simple polygon (no fold-backs), else the revolved mesh self-intersects."""
    P = np.asarray(profile)
    n = len(P)
    segs = [(P[i], P[(i + 1) % n]) for i in range(n) if not (P[i][0] < 1e-12 and P[(i + 1) % n][0] < 1e-12)]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    for i in range(len(segs)):
        for j in range(i + 2, len(segs)):
            if i == 0 and j == len(segs) - 1:
                continue
            a, b = segs[i]
            c, d = segs[j]
            if (cross(a, b, c) * cross(a, b, d) < 0) and (cross(c, d, a) * cross(c, d, b) < 0):
                raise AssertionError(f"{what}: profile segments {i} and {j} cross")
    if min(P[:, 0]) < -1e-12:
        raise AssertionError(f"{what}: negative radius")


def thread_dims(sp):
    ta = math.tan(math.radians(FLANK_DEG))
    w_root = sp.w_crest + 2 * sp.h * ta  # jar ridge width at its root
    # lid ridge crest width so that the flanks are parallel with G_AXIAL per side when centred in the groove
    w_lid_crest = sp.pitch - w_root + 2 * C_RADIAL * ta - 2 * G_AXIAL
    w_lid_root = w_lid_crest + 2 * sp.h * ta
    return ta, w_root, w_lid_crest, w_lid_root


def thread_profile(h, w_crest, ta, visual):
    """Trapezoid (d, u) polygon, base sunk EMBED into the wall. Visual version has chamfered crest corners."""
    hb = w_crest / 2 + (h + EMBED) * ta
    if not visual:
        return [(-EMBED, -hb), (h, -w_crest / 2), (h, w_crest / 2), (-EMBED, hb)]
    c = min(0.15, 0.3 * h)
    return [(-EMBED, -hb), (h - c, -(w_crest / 2 + c * ta)), (h, -w_crest / 2 + 0.6 * c),
            (h, w_crest / 2 - 0.6 * c), (h - c, w_crest / 2 + c * ta), (-EMBED, hb)]


def jar_profile(sp, visual):
    """Closed r-z polygon of the glass jar wall (real wall thickness, open mouth)."""
    nf = 8 if visual else 3  # points per fillet
    nsh = 16 if visual else 8  # shoulder segments
    R, Ri = sp.r_body, sp.r_body - sp.t_wall
    Rn, Rni = sp.r_root, sp.r_root - sp.t_neck
    fb, rr, ri, fi = 2.0, 0.9, 0.6, 1.5
    P = []
    P.append((0.0, 1.2))  # punt (base recess) centre
    P.append((0.6 * R, 0.0))
    P += arc(R - fb, fb, fb, -90, 0, nf)[0:]  # outer bottom fillet, ends at (R, fb)
    # straight body wall: add a few rings so long triangles stay reasonable
    for z in np.linspace(fb, sp.z_body_top, 5)[1:]:
        P.append((R, z))
    # S-shaped shoulder, vertical tangents at both ends
    for t in np.linspace(0, 1, nsh + 1)[1:]:
        P.append((R - (R - Rn) * smoothstep(t), sp.z_body_top + (sp.z_neck_root - sp.z_body_top) * t))
    # neck (thread root cylinder) with axial rings every ~1 mm, optional bead
    # (no rings inside the thread band: ring vertices near the thread/neck intersection curve make sliver triangles)
    ta = math.tan(math.radians(FLANK_DEG))
    hb = sp.w_crest / 2 + (sp.h + EMBED) * ta + 0.4
    band = (sp.rim_z - sp.a_top - sp.pitch * sp.turns_jar - hb, sp.rim_z - sp.a_top + hb)
    zs = list(np.arange(sp.z_neck_root, sp.rim_z - rr, 1.0)[1:])
    for z in zs:
        if sp.bead_z > 0 and abs(z - sp.bead_z) < sp.bead_r + 0.6:
            continue
        if band[0] < z < band[1]:
            continue
        if sp.bead_z > 0 and z > sp.bead_z and (len(P) and P[-1][1] < sp.bead_z):
            P += [(Rn + sp.bead_r * math.sin(a), sp.bead_z - sp.bead_r * math.cos(a))
                  for a in np.linspace(0, np.pi, 9 if visual else 5)]
        P.append((Rn, z))
    P += arc(Rn - rr, sp.rim_z - rr, rr, 0, 90, nf)  # rim outer round, ends at (Rn - rr, rim_z)
    P += arc(Rni + ri, sp.rim_z - ri, ri, 90, 180, nf)  # rim inner round
    for z in np.arange(sp.rim_z - ri, sp.z_neck_root, -2.0)[1:]:
        P.append((Rni, z))
    P.append((Rni, sp.z_neck_root))
    for t in np.linspace(0, 1, nsh + 1)[1:]:
        P.append((Rni + (Ri - Rni) * smoothstep(t), sp.z_neck_root + (sp.z_body_top - sp.z_neck_root) * t))
    for z in np.linspace(sp.z_body_top, sp.t_base + fi, 5)[1:]:
        P.append((Ri, z))
    P += arc(Ri - fi, sp.t_base + fi, fi, 0, -90, nf)[1:]
    P.append((0.0, sp.t_base))
    # drop exact duplicates
    out = [P[0]]
    for p in P[1:]:
        if abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    return out


def lid_profile(sp, visual):
    nf = 8 if visual else 3
    _, _, _, _ = thread_dims(sp)
    Rr = sp.r_root + sp.h + C_RADIAL  # lid thread root (skirt inner wall)
    Ro = Rr + sp.t_skirt
    rt = 1.2 if sp.lid_kind == "metal" else 1.0
    rb = 0.9 if sp.lid_style == "smooth" else 0.5
    ch, fi = 0.6, 0.4
    rb = min(rb, sp.t_skirt - ch - 0.1)  # bottom fillet + lead-in chamfer must fit in the skirt wall
    L = sp.l_skirt
    P = [(0.0, sp.t_top)]
    if visual and sp.lid_style == "smooth":
        # shallow concentric groove on the top panel (sunk below the collision top, never above it)
        P += [(0.55 * Ro, sp.t_top), (0.58 * Ro, sp.t_top - 0.35), (0.62 * Ro, sp.t_top - 0.35), (0.65 * Ro, sp.t_top)]
    P += arc(Ro - rt, sp.t_top - rt, rt, 90, 0, nf)
    # outer skirt rings (used by the rib modulation)
    for z in np.linspace(sp.t_top - rt, -L + rb, 6)[1:]:
        P.append((Ro, z))
    P += arc(Ro - rb, -L + rb, rb, 0, -90, nf)[1:]
    P.append((Rr + ch, -L))
    P.append((Rr, -L + ch))
    hl = helix_layout(sp)
    hb = hl["w_lid_root"] / 2 + EMBED * hl["ta"] + 0.4
    for z in np.arange(-L + ch, -fi, 1.0)[1:]:
        if hl["lid_z_start"] - hb < z < hl["lid_z_top"] + hb:
            continue  # no rings in the thread band (see jar_profile)
        P.append((Rr, z))
    P += arc(Rr - fi, -fi, fi, 0, 90, nf)
    P.append((0.0, 0.0))
    out = [P[0]]
    for p in P[1:]:
        if abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    return out, Rr, Ro


def helix_layout(sp):
    """Jar and lid thread helices. Returns dict of the numbers station.py needs."""
    ta, w_root, w_lc, w_lr = thread_dims(sp)
    P = sp.pitch
    jar_z_start = sp.rim_z - sp.a_top - P * sp.turns_jar  # jar ridge centre at theta = 0
    # lid ridge centre (lid frame, lid angle theta): zL0 + P theta / 2pi. Seated (lid origin at rim_z, no rotation)
    # with the lid riding on its LOWER flank: lid centre = jar centre + P/2 - G_AXIAL.
    zl0 = (jar_z_start + P / 2 - G_AXIAL - sp.rim_z) % P
    top_clear = 0.7
    z_lid_top = -(w_lr / 2 + EMBED * ta + top_clear)
    th_le = 2 * np.pi * (z_lid_top - zl0) / P
    th_ls = th_le - 2 * np.pi * sp.turns_lid
    z_lid_start = z_lid_top - P * sp.turns_lid
    return dict(ta=ta, w_root=w_root, w_lid_crest=w_lc, w_lid_root=w_lr, jar_z_start=jar_z_start,
                lid_theta_start=th_ls, lid_z_start=z_lid_start, lid_z_top=z_lid_top)


# ----------------------------------------------------------------------------------------------------------------
# parts
# ----------------------------------------------------------------------------------------------------------------
def build_jar(sp, visual):
    n_seg = 240 if visual else 120
    hl = helix_layout(sp)
    # lathe columns sit half a step off the thread samples: coincident axial-plane edges would give the boolean
    # zero-length edges along the intersection curve
    prof = jar_profile(sp, visual)
    assert_simple(prof, f"{sp.name} jar")
    v, f = lathe(prof, n_seg, theta0=np.pi / n_seg)
    body = to_manifold(v, f)
    tv, tf = sweep_thread(sp.r_root, +1, sp.pitch, hl["jar_z_start"], 0.0, sp.turns_jar,
                          thread_profile(sp.h, sp.w_crest, hl["ta"], visual), spt=n_seg)
    thread = to_manifold(tv, tf)
    return from_manifold(body + thread)


def build_lid(sp, visual):
    hl = helix_layout(sp)
    prof, Rr, Ro = lid_profile(sp, visual)
    assert_simple(prof, f"{sp.name} lid")
    if visual and sp.lid_style in ("ribbed", "knurled"):
        n_seg = 4 * sp.n_ribs
    else:
        n_seg = 240 if visual else 120
    # lathe columns half a step off the thread sweep samples (see build_jar)
    dth = 2 * np.pi / n_seg
    theta0 = (hl["lid_theta_start"] + 0.5 * dth) % dth
    ring_mod = None
    if visual and sp.lid_style in ("ribbed", "knurled"):
        z_hi, z_lo = sp.t_top - 1.6, -sp.l_skirt + 1.2

        def ring_mod(i, th, r, z):
            if abs(r - Ro) > 1e-9 or not (z_lo - 1e-9 <= z <= z_hi + 1e-9):
                return np.full_like(th, r)
            groove = 0.5 * (1 - np.cos(sp.n_ribs * (th - theta0)))
            return r - sp.rib_depth * groove

    v, f = lathe(prof, n_seg, theta0=theta0, ring_mod=ring_mod)
    body = to_manifold(v, f)
    tv, tf = sweep_thread(Rr, -1, sp.pitch, hl["lid_z_start"], hl["lid_theta_start"], sp.turns_lid,
                          thread_profile(sp.h, hl["w_lid_crest"], hl["ta"], visual), spt=n_seg)
    thread = to_manifold(tv, tf)
    return from_manifold(body + thread), Rr, Ro


def mesh_stats(v, f):
    tm = trimesh.Trimesh(v, f, process=False)
    e = tm.edges_unique_length
    return dict(n_verts=len(v), n_faces=len(f), watertight=bool(tm.is_watertight),
                winding_consistent=bool(tm.is_winding_consistent), euler=int(tm.euler_number),
                volume_mm3=float(tm.volume), min_edge_mm=float(e.min()), max_edge_mm=float(e.max()),
                min_face_area_mm2=float(tm.area_faces.min()))


def write_obj(path, v_mm, f, header):
    with open(path, "w") as fh:
        fh.write(f"# {header}\n# units: metres, z up\n")
        for x, y, z in v_mm * 1e-3:
            fh.write(f"v {x:.7f} {y:.7f} {z:.7f}\n")
        for a, b, c in f + 1:
            fh.write(f"f {a} {b} {c}\n")


def main():
    os.makedirs(ASSETS, exist_ok=True)
    params = dict(units="metres", flank_deg=FLANK_DEG, embed_mm=EMBED, taper_deg=TAPER_DEG,
                  c_radial_mm=C_RADIAL, g_axial_mm=G_AXIAL, glass_rho=GLASS_RHO, parts={})
    for sp in SPECS:
        hl = helix_layout(sp)
        rec = dict(spec=asdict(sp), helix={k: float(v) for k, v in hl.items()})
        for part in ("jar", "lid"):
            for visual in (False, True):
                if part == "jar":
                    v, f = build_jar(sp, visual)
                else:
                    (v, f), Rr, Ro = build_lid(sp, visual)
                    rec["lid_r_root_mm"], rec["lid_r_outer_mm"] = Rr, Ro
                tag = "vis" if visual else "col"
                st = mesh_stats(v, f)
                assert st["watertight"] and st["winding_consistent"] and st["volume_mm3"] > 0, (sp.name, part, st)
                fn = f"{sp.name}_{part}_{tag}.obj"
                write_obj(os.path.join(ASSETS, fn), v, f, f"{sp.title} {part} ({tag}) - jars station, procedural")
                rec[f"{part}_{tag}"] = dict(file=fn, **st)
                print(f"{fn:22s} verts {st['n_verts']:6d} faces {st['n_faces']:6d} vol {st['volume_mm3'] / 1e3:8.2f} "
                      f"cm3 euler {st['euler']} min edge {st['min_edge_mm']:.4f} mm min area {st['min_face_area_mm2']:.2e}")
        vol_jar = rec["jar_col"]["volume_mm3"] * 1e-9
        vol_lid = rec["lid_col"]["volume_mm3"] * 1e-9
        rec["jar_mass_kg"] = GLASS_RHO * vol_jar
        rec["lid_mass_kg"] = sp.lid_mass_g * 1e-3
        rec["lid_rho_eff"] = rec["lid_mass_kg"] / vol_lid
        rec["rim_z"] = sp.rim_z * 1e-3
        rec["pitch"] = sp.pitch * 1e-3
        print(f"  {sp.name}: jar {rec['jar_mass_kg'] * 1e3:.0f} g, lid {sp.lid_mass_g:.0f} g "
              f"(rho_eff {rec['lid_rho_eff']:.0f} kg/m3); lid thread root/outer R {Rr:.2f}/{Ro:.2f} mm")
        params["parts"][sp.name] = rec
    json.dump(params, open(os.path.join(ASSETS, "params.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
