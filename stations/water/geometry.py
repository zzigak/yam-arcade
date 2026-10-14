"""Procedural containers for the water station: a Brita-style pitcher and three cups.

Every container is a thick-walled shell of revolution (or of a rounded-rectangle plan, for the pitcher) described by
its INNER surface: a plan curve P(theta) scaled by height-dependent semi-axes, plus an optional spout bulge. From that
one description this module builds

  * the visual mesh: a closed, watertight solid (outer wall, rim, inner wall, floor, base / foot), plus handle and lid;
  * the collision pieces: convex hulls of wall wedges (angular slice x height band), the base, handle segments and the
    lid.  Hollow containers cannot be one convex hull, and a grid SDF of a 3 mm shell is fragile, so the shell is cut
    into small convex pieces whose SDFs are exact and fine (Genesis sizes each piece's SDF grid to the piece);
  * a polar lookup of the inner/outer wall (R_in(phi, z), R_out(phi, z)) used to count particles per container,
    test for particles inside walls, and compute the brim capacity;
  * an MJCF body (free joint, or a 6-DoF PD-driven "oracle" mount) with explicit mass / inertia from the visual solid.

Fluid offset: the collision wall's inner face sits FLUID_OFFSET (one particle radius) inside the visual inner face, and
the collision floor FLUID_OFFSET above the visual floor.  The SPH-rigid coupling stops particle CENTRES at the collision
surface, so this keeps every particle's sphere (radius = d/2) inside the visual cavity: the reconstructed water surface
then meets the visual wall instead of poking through it.  Rigid-rigid contact is unaffected (nothing rigid goes inside a
container).  Frames: container-local, origin at the centre of the base on the support plane, +z up; the pitcher's spout
points along +x and its handle along -x.
"""
from dataclasses import dataclass, field
import os

import numpy as np
import trimesh

FLUID_OFFSET = 0.002


def _superellipse(theta, n):
    """Unit superellipse |x|^n + |y|^n = 1, parametrised by angle-like theta (n = 2 gives the circle)."""
    c, s = np.cos(theta), np.sin(theta)
    e = 2.0 / n
    return np.sign(c) * np.abs(c) ** e, np.sign(s) * np.abs(s) ** e


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


@dataclass
class Handle:
    """A tube swept along a planar path in the container's xz plane (y = 0)."""
    path: np.ndarray            # (K, 2) x, z control points (already rounded / densified)
    half_w: float               # half-width in the path plane (normal to the path, in xz)
    half_t: float               # half-thickness along y
    n_exp: float = 3.0          # superellipse exponent of the cross-section (2 = ellipse, larger = boxier)


@dataclass
class Spout:
    length: float               # outward displacement of the lip at theta = 0, z = rim (m)
    half_angle: float           # angular half-width of the bulge (rad)
    z_start: float              # height where the bulge starts growing
    power: float = 1.5          # profile in theta: (1 - |theta|/half_angle)^power, peaked = V channel


@dataclass
class Lid:
    x_cut: float                # lid covers x < x_cut (the spout side stays open)
    t_below: float              # lid slab extends this far below the rim ...
    t_above: float              # ... and this far above it


@dataclass
class ContainerSpec:
    name: str
    z_floor: float              # inner floor height above the base plane
    z_rim: float                # rim height
    prof_z: tuple               # inner profile sample heights (z_floor .. z_rim)
    prof_a: tuple               # inner semi-axis along x at those heights
    prof_b: tuple | None        # inner semi-axis along y (None: same as a -> round)
    wall: float                 # wall thickness (m)
    density: float              # material density (kg/m^3) -> mass and inertia
    plan_n: float = 2.0         # plan superellipse exponent (2 = round)
    foot: tuple | None = None   # (radius, height) of a foot ring below the bowl, or None (flat base)
    spout: Spout | None = None
    lid: Lid | None = None
    handle: Handle | None = None
    n_theta: int = 96           # visual resolution
    n_z: int = 60
    coll_n_theta: int = 32      # collision wedges
    coll_z: tuple | None = None  # collision band boundaries (defaults to prof_z)
    rgba: tuple = (0.9, 0.9, 0.9, 1.0)
    # measured cavities (SAM 3D, see sam3d_fit.py): inner wall radius table instead of the analytic profile
    polar: tuple | None = None           # (zs (nz,), phis (np,) spanning [-pi, pi], R (nz, np))
    base_hull: object = None             # points whose convex hull (with the raised floor) is the base piece
    handle_hulls: list | None = None     # point sets, one convex piece each
    visual_mesh: object = None           # trimesh used as the visual instead of the generated solid
    mass_mesh: object = None             # watertight trimesh for mass / inertia


class Container:
    """Geometry of one container built from a ContainerSpec (see module docstring)."""

    def __init__(self, spec: ContainerSpec):
        self.spec = spec
        s = spec
        self.prof_z = np.asarray(s.prof_z, float)
        self.prof_a = np.asarray(s.prof_a, float)
        self.prof_b = np.asarray(s.prof_b if s.prof_b is not None else s.prof_a, float)
        assert abs(self.prof_z[0] - s.z_floor) < 1e-9 and abs(self.prof_z[-1] - s.z_rim) < 1e-9
        self._build_polar_tables()

    # ------------------------------------------------------------------ surface
    def inner_points(self, theta, z):
        """Inner wall points for broadcastable theta, z arrays -> (..., 3)."""
        s = self.spec
        theta, z = np.broadcast_arrays(np.asarray(theta, float), np.asarray(z, float))
        if s.polar is not None:
            r = self._polar_R(theta, z)
            return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=-1)
        a = np.interp(z, self.prof_z, self.prof_a)
        b = np.interp(z, self.prof_z, self.prof_b)
        ux, uy = _superellipse(theta, s.plan_n)
        x, y = a * ux, b * uy
        if s.spout is not None:
            sp = s.spout
            th = np.angle(np.exp(1j * theta))  # wrap to (-pi, pi]
            w = np.clip(1.0 - np.abs(th) / sp.half_angle, 0.0, 1.0) ** sp.power
            h = _smoothstep((z - sp.z_start) / (s.z_rim - sp.z_start)) ** 1.5
            x = x + sp.length * w * h
        return np.stack([x, y, z], axis=-1)

    def _polar_R(self, theta, z):
        zs, phis, R = self.spec.polar
        th = np.angle(np.exp(1j * theta))
        zi = np.clip((z - zs[0]) / (zs[-1] - zs[0]) * (len(zs) - 1), 0.0, len(zs) - 1.0)
        # zs may be non-uniform: map through interpolation of the index
        zi = np.interp(z, zs, np.arange(len(zs)))
        pi_ = np.interp(th, phis, np.arange(len(phis)))
        z0 = np.minimum(zi.astype(int), len(zs) - 2)
        p0 = np.minimum(pi_.astype(int), len(phis) - 2)
        fz, fp = zi - z0, pi_ - p0
        return ((1 - fz) * (1 - fp) * R[z0, p0] + fz * (1 - fp) * R[z0 + 1, p0]
                + (1 - fz) * fp * R[z0, p0 + 1] + fz * fp * R[z0 + 1, p0 + 1])

    def inner_normals(self, theta, z, eps=1e-4):
        """Outward (into the wall) unit normals of the inner surface, by central differences."""
        theta, z = np.broadcast_arrays(np.asarray(theta, float), np.asarray(z, float))
        dt = self.inner_points(theta + eps, z) - self.inner_points(theta - eps, z)
        zlo = np.maximum(z - eps, self.prof_z[0])
        zhi = np.minimum(z + eps, self.prof_z[-1])
        dz = self.inner_points(theta, zhi) - self.inner_points(theta, zlo)
        n = np.cross(dt, dz)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        p = self.inner_points(theta, z)
        sign = np.sign(np.sum(n[..., :2] * p[..., :2], axis=-1, keepdims=True))
        return n * sign

    def outer_points(self, theta, z):
        return self.inner_points(theta, z) + self.spec.wall * self.inner_normals(theta, z)

    def _warp(self, u):
        # monotone map (-1,1) -> (-pi,pi) with slope ~ pi/3 near 0 and ~ 1.33 pi at the ends
        # theta = pi * (u + c*u^3) / (1 + c)  with c = 2 -> slope at 0 = pi/3
        c = 2.0
        return np.pi * (u + c * u ** 3) / (1.0 + c)

    # ------------------------------------------------------------------ polar tables (counting / wall tests)
    def _build_polar_tables(self, n_phi=720, n_z=240):
        s = self.spec
        zs = np.linspace(s.z_floor, s.z_rim, n_z)
        th = np.linspace(-np.pi, np.pi, 4 * n_phi, endpoint=False)
        phis = np.linspace(-np.pi, np.pi, n_phi + 1)
        R_in = np.zeros((n_z, n_phi + 1))
        R_out = np.zeros((n_z, n_phi + 1))
        for k, z in enumerate(zs):
            for tab, pts in ((R_in, self.inner_points(th, z)), (R_out, self.outer_points(th, z))):
                ph = np.arctan2(pts[:, 1], pts[:, 0])
                r = np.hypot(pts[:, 0], pts[:, 1])
                o = np.argsort(ph)
                ph, r = ph[o], r[o]
                tab[k] = np.interp(phis, np.r_[ph - 2 * np.pi, ph, ph + 2 * np.pi], np.r_[r, r, r])
        self._tab_z, self._tab_phi, self._R_in, self._R_out = zs, phis, R_in, R_out
        # brim capacity: integral over z of the polar area 0.5 * int R^2 dphi
        area = 0.5 * np.trapezoid(R_in ** 2, phis, axis=1)
        self.capacity = float(np.trapezoid(area, zs))   # m^3
        self._area_z = area

    def _lookup(self, tab, phi, z):
        """Bilinear lookup of a (z, phi) table; inputs cast to float64 (float32 particle positions round the
        upper clamp onto the last node)."""
        nz, nphi = tab.shape
        zi = (np.asarray(z, np.float64) - self._tab_z[0]) / (self._tab_z[-1] - self._tab_z[0]) * (nz - 1)
        pi_ = (np.asarray(phi, np.float64) + np.pi) / (2 * np.pi) * (nphi - 1)
        zi = np.clip(np.nan_to_num(zi), 0.0, nz - 1.0)
        pi_ = np.clip(np.nan_to_num(pi_), 0.0, nphi - 1.0)
        z0 = np.minimum(zi.astype(int), nz - 2)
        p0 = np.minimum(pi_.astype(int), nphi - 2)
        fz, fp = zi - z0, pi_ - p0
        return ((1 - fz) * (1 - fp) * tab[z0, p0] + fz * (1 - fp) * tab[z0 + 1, p0]
                + (1 - fz) * fp * tab[z0, p0 + 1] + fz * fp * tab[z0 + 1, p0 + 1])

    def volume_below(self, z):
        """Visual cavity volume between the floor and height z (m^3)."""
        z = np.clip(z, self._tab_z[0], self._tab_z[-1])
        cum = np.r_[0.0, np.cumsum(0.5 * (self._area_z[1:] + self._area_z[:-1]) * np.diff(self._tab_z))]
        return float(np.interp(z, self._tab_z, cum))

    def classify_local(self, p, rim_margin=0.0):
        """p: (N, 3) points in the container frame.

        Returns (in_cavity, in_wall): in_cavity = inside the visual cavity below rim + rim_margin;
        in_wall = inside the visual solid (wall, floor or base; handle / lid excluded)."""
        s = self.spec
        r = np.hypot(p[:, 0], p[:, 1])
        phi = np.arctan2(p[:, 1], p[:, 0])
        z = p[:, 2]
        zc = np.clip(z, s.z_floor, s.z_rim)
        R_in = self._lookup(self._R_in, phi, zc)
        R_out = self._lookup(self._R_out, phi, zc)
        in_cavity = (z > s.z_floor) & (z < s.z_rim + rim_margin) & (r < R_in)
        in_band = (z >= 0.0) & (z <= s.z_rim)
        in_wall = in_band & (r < R_out) & ~((z > s.z_floor) & (r < R_in))
        if s.foot is not None:
            # below the bowl: only the foot ring (cylinder of radius foot[0] up to foot[1]) is solid
            below = z < self._outer_floor_z()
            in_wall = np.where(below, (z >= 0) & (r < s.foot[0]) & (z <= s.foot[1]), in_wall)
        return in_cavity, in_wall

    def wall_clearance_local(self, p):
        """Radial distance from each point inside the cavity to the visual inner wall (>= 0 inside)."""
        s = self.spec
        r = np.hypot(p[:, 0], p[:, 1])
        phi = np.arctan2(p[:, 1], p[:, 0])
        zc = np.clip(p[:, 2], s.z_floor, s.z_rim)
        return np.minimum(self._lookup(self._R_in, phi, zc) - r, p[:, 2] - s.z_floor)

    def _outer_floor_z(self):
        """Height of the outer surface ring under the inner floor (bowl underside for footed cups)."""
        s = self.spec
        th = np.array([0.0])
        return float(self.outer_points(th, s.z_floor)[0, 2]) if s.foot is not None else 0.0

    # ------------------------------------------------------------------ visual meshes
    def body_mesh(self):
        """Watertight solid: outer wall + rim + inner wall + floor + base (with foot when specified)."""
        s = self.spec
        th = self._warp_or_uniform(s.n_theta)
        zs = self._z_samples(s.n_z)
        n = len(th)
        inner = self.inner_points(th[None, :], zs[:, None])          # (nz, n, 3)
        outer = self.outer_points(th[None, :], zs[:, None])
        rings_out = []  # bottom -> top, each (n, 3)
        if s.foot is None:
            base = outer[0].copy()
            base[:, 2] = 0.0
            # small chamfer at the base edge
            ch = min(0.0015, 0.5 * s.z_floor)
            c = base[:, :2].mean(axis=0)
            base_in = base.copy()
            base_in[:, :2] = c + (base[:, :2] - c) * (1 - ch / np.linalg.norm(base[:, :2] - c, axis=1, keepdims=True))
            base_up = outer[0].copy()
            base_up[:, 2] = ch
            rings_out += [base_in, base_up]
        else:
            rf, hf = s.foot
            ring = np.stack([rf * np.cos(th), rf * np.sin(th), np.zeros(n)], axis=-1)
            rings_out += [ring * [0.93, 0.93, 1], ring + [0, 0, 0.0012], ring + [0, 0, hf]]
        rings_out += [outer[k] for k in range(len(zs))]
        rings_in = [inner[k] for k in range(len(zs))][::-1]  # top -> bottom
        V, F = [], []

        def add_ring(r):
            V.append(r)
            return sum(len(v) for v in V) - n

        def strip(i0, i1, flip=False):
            for j in range(n):
                a, b, c, d = i0 + j, i0 + (j + 1) % n, i1 + (j + 1) % n, i1 + j
                F.extend([[a, b, c], [a, c, d]] if not flip else [[a, c, b], [a, d, c]])

        idx = [add_ring(r) for r in rings_out] + [add_ring(r) for r in rings_in]
        for k in range(len(idx) - 1):
            strip(idx[k], idx[k + 1])
        # floor (inner, top -> centre) and base (bottom centre)
        floor_c = len(np.concatenate(V))
        fl = rings_in[-1]
        V.append(np.array([[fl[:, 0].mean(), fl[:, 1].mean(), s.z_floor]]))
        base_c = floor_c + 1
        V.append(np.array([[rings_out[0][:, 0].mean(), rings_out[0][:, 1].mean(), 0.0]]))
        i_fl, i_b = idx[-1], idx[0]
        for j in range(n):
            F.append([i_fl + j, i_fl + (j + 1) % n, floor_c])
            F.append([i_b + (j + 1) % n, i_b + j, base_c])
        m = trimesh.Trimesh(np.concatenate(V), np.array(F), process=True)
        trimesh.repair.fix_normals(m)
        return m

    def _warp_or_uniform(self, n):
        if self.spec.spout is None:
            return np.linspace(-np.pi, np.pi, n, endpoint=False)
        return self._warp(np.linspace(-1.0, 1.0, n, endpoint=False))

    def _z_samples(self, n):
        s = self.spec
        z = np.linspace(s.z_floor, s.z_rim, n)
        return np.unique(np.r_[z, self.prof_z])

    def handle_mesh(self):
        h = self.spec.handle
        if h is None:
            return None
        rings = self._handle_rings(h, n_cs=24)
        n = rings.shape[1]
        V = rings.reshape(-1, 3)
        F = []
        for k in range(len(rings) - 1):
            for j in range(n):
                a, b = k * n + j, k * n + (j + 1) % n
                c, d = (k + 1) * n + (j + 1) % n, (k + 1) * n + j
                F += [[a, b, c], [a, c, d]]
        c0, c1 = len(V), len(V) + 1
        V = np.vstack([V, rings[0].mean(0), rings[-1].mean(0)])
        last = (len(rings) - 1) * n
        for j in range(n):
            F.append([(j + 1) % n, j, c0])
            F.append([last + j, last + (j + 1) % n, c1])
        m = trimesh.Trimesh(V, np.array(F), process=True)
        trimesh.repair.fix_normals(m)
        return m

    def _handle_path(self, h):
        """The handle path with both ends moved onto the wall's mid-surface at their height (handle side, -x), so the
        end caps sit inside the wall and never protrude into the cavity."""
        path = h.path.copy()
        for i in (0, -1):
            z = path[i, 1]
            x_mid = 0.5 * (self.inner_points(np.pi, z)[0] + self.outer_points(np.pi, z)[0])
            path[i, 0] = x_mid
        return path

    def _handle_rings(self, h, n_cs):
        path = self._handle_path(h)
        P = np.c_[path[:, 0], np.zeros(len(path)), path[:, 1]]
        T = np.gradient(P, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        Y = np.array([0.0, 1.0, 0.0])
        N = np.cross(T, Y)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        ph = np.linspace(0, 2 * np.pi, n_cs, endpoint=False)
        cx, cy = _superellipse(ph, h.n_exp)
        return P[:, None, :] + h.half_w * cx[None, :, None] * N[:, None, :] + h.half_t * cy[None, :, None] * Y

    def lid_mesh(self):
        hull = self._lid_hull(offset=0.0)
        return hull

    def _lid_hull(self, offset):
        s = self.spec
        if s.lid is None:
            return None
        th = self._warp_or_uniform(256)
        ring = self.outer_points(th, s.z_rim)[:, :2]
        poly = _clip_halfplane(ring, s.lid.x_cut)
        lo = np.c_[poly, np.full(len(poly), s.z_rim - s.lid.t_below - offset)]
        hi = np.c_[poly, np.full(len(poly), s.z_rim + s.lid.t_above + offset)]
        return trimesh.convex.convex_hull(np.vstack([lo, hi]))

    def visual_parts(self):
        """name -> trimesh for the export / visual geoms."""
        if self.spec.visual_mesh is not None:
            return {"body": self.spec.visual_mesh}
        parts = {"body": self.body_mesh()}
        if self.spec.handle is not None:
            parts["handle"] = self.handle_mesh()
        if self.spec.lid is not None:
            parts["lid"] = self.lid_mesh()
        return parts

    # ------------------------------------------------------------------ collision pieces
    def collision_pieces(self):
        """List of convex trimeshes (container frame)."""
        s = self.spec
        d = FLUID_OFFSET
        pieces = []
        # base / bottom: hull of the raised inner floor ring, the outer ring at the floor and the support ring
        th = self._warp_or_uniform(64)
        floor_in = self.inner_points(th, s.z_floor) - d * self.inner_normals(th, s.z_floor)
        floor_in[:, 2] = s.z_floor + d
        out_f = self.outer_points(th, s.z_floor)
        if s.base_hull is not None:
            pieces.append(trimesh.convex.convex_hull(np.vstack([floor_in, np.asarray(s.base_hull)])))
        elif s.foot is None:
            sup = out_f.copy()
            sup[:, 2] = 0.0
            pieces.append(trimesh.convex.convex_hull(np.vstack([floor_in, out_f, sup])))
        else:
            rf, hf = s.foot
            ring = np.stack([rf * np.cos(th), rf * np.sin(th), np.zeros_like(th)], axis=-1)
            pieces.append(trimesh.convex.convex_hull(np.vstack([ring, ring + [0, 0, hf + 0.001]])))
            pieces.append(trimesh.convex.convex_hull(np.vstack([floor_in, out_f, ring + [0, 0, hf]])))
        # wall wedges
        zb = np.asarray(s.coll_z if s.coll_z is not None else self.prof_z, float)
        thc = self._warp_or_uniform(s.coll_n_theta)
        thc = np.r_[thc, thc[0] + 2 * np.pi]
        # Seam covers.  With abutting wedges, a particle that slips a little into the seam between two of them sees
        # each piece's internal side face as the nearest surface, gets sideways normals and is never pushed back
        # (seen in the oracle run: 4-5 particles parked 1.4 mm deep in the tumbler's seams).  A narrow extra piece
        # straddles every seam (+-35 % of the neighbouring wedge widths): its inner face is the wall there and its own
        # side faces are 2.5-3 mm away, so a penetrating particle is always pushed back along the wall normal.  Its
        # chord is short (sagitta ~0.07 mm); widening the wedges instead deepened their chords into the cavity and
        # kicked the resting water.  Bands overlap 3 mm vertically for the same reason.
        ov, zov = 0.35, 0.003
        n_seg = len(thc) - 1

        def wedge(t0, t1, z0, z1, k):
            T = np.array([t0, t1])
            Z = np.array([z0, z1])
            TT, ZZ = np.meshgrid(T, Z)
            pin = self.inner_points(TT, ZZ)
            nrm = self.inner_normals(TT, ZZ)
            # fluid offset on every water-facing face: inner face d inward, outer face d outward, and the top band
            # d above the rim (a resting particle's centre sits one radius off any surface)
            pts = np.vstack([(pin - d * nrm).reshape(-1, 3), (pin + (s.wall + d) * nrm).reshape(-1, 3)])
            if k == len(zb) - 2:
                top = pts.copy()
                top[:, 2] = np.where(np.isclose(top[:, 2], s.z_rim, atol=1e-4), s.z_rim + d, top[:, 2])
                pts = np.vstack([pts, top])
            # the lowest band reaches down into the base so the floor/wall junction has no seam
            if k == 0:
                low = (pin - d * nrm).reshape(-1, 3)[:2].copy()
                low[:, 2] = max(0.0, s.z_floor - 0.004)
                pts = np.vstack([pts, low])
            return trimesh.convex.convex_hull(pts)

        for k in range(len(zb) - 1):
            z0, z1 = max(zb[k] - zov, s.z_floor), min(zb[k + 1] + zov, s.z_rim)
            for i in range(n_seg):
                pieces.append(wedge(thc[i], thc[i + 1], z0, z1, k))
                dth_prev = thc[i] - thc[i - 1] if i > 0 else thc[n_seg] - thc[n_seg - 1]
                dth_next = thc[i + 1] - thc[i]
                pieces.append(wedge(thc[i] - ov * dth_prev, thc[i] + ov * dth_next, z0, z1, k))
        if s.handle is not None:
            rings = self._handle_rings(s.handle, n_cs=12)
            for k in range(len(rings) - 1):
                pieces.append(trimesh.convex.convex_hull(np.vstack([rings[k], rings[k + 1]])))
        for pts in s.handle_hulls or ():
            pieces.append(trimesh.convex.convex_hull(np.asarray(pts)))
        if s.lid is not None:
            pieces.append(self._lid_hull(offset=d))
        return pieces

    # ------------------------------------------------------------------ mass properties
    def mass_properties(self):
        """(mass, com, inertia_about_com 3x3) of the visual solid (or the spec's mass_mesh) with the spec density."""
        parts = [self.spec.mass_mesh] if self.spec.mass_mesh is not None else list(self.visual_parts().values())
        m_tot, c_acc, mprops = 0.0, np.zeros(3), []
        for p in parts:
            assert p.is_watertight, f"{self.spec.name}: visual part not watertight"
            vol = abs(p.volume)
            m = vol * self.spec.density
            mprops.append((m, p.center_mass, p.moment_inertia * self.spec.density))
            m_tot += m
            c_acc += m * p.center_mass
        com = c_acc / m_tot
        I = np.zeros((3, 3))
        for m, c, Ic in mprops:
            r = c - com
            I += Ic + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        return m_tot, com, I

    # ------------------------------------------------------------------ export
    def write(self, out_dir):
        """Write visual OBJs, collision piece OBJs and an info dict; returns the info dict."""
        os.makedirs(out_dir, exist_ok=True)
        name = self.spec.name
        vis = {}
        for part, m in self.visual_parts().items():
            path = os.path.join(out_dir, f"{name}_{part}.obj")
            trimesh.Trimesh(m.vertices, m.faces, process=False).export(path)   # geometry only (MJCF visual)
            vis[part] = path
        coll = []
        cdir = os.path.join(out_dir, f"{name}_coll")
        os.makedirs(cdir, exist_ok=True)
        for i, m in enumerate(self.collision_pieces()):
            path = os.path.join(cdir, f"c{i:03d}.obj")
            m.export(path)
            coll.append(path)
        mass, com, I = self.mass_properties()
        return dict(visual=vis, collision=coll, mass=mass, com=com, inertia=I, capacity=self.capacity)


def _clip_halfplane(poly, x_cut):
    """Sutherland-Hodgman clip of a closed 2D polygon to x <= x_cut."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ina, inb = a[0] <= x_cut, b[0] <= x_cut
        if ina:
            out.append(a)
        if ina != inb:
            t = (x_cut - a[0]) / (b[0] - a[0])
            out.append(a + t * (b - a))
    return np.array(out)


def rounded_path(corners, radius, n_arc=8, n_seg=4):
    """Polyline through `corners` (x, z) with circular-arc fillets of `radius` at interior corners, densified."""
    corners = np.asarray(corners, float)
    pts = [corners[0]]
    for i in range(1, len(corners) - 1):
        p0, p1, p2 = corners[i - 1], corners[i], corners[i + 1]
        u, v = p0 - p1, p2 - p1
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        u, v = u / lu, v / lv
        ang = np.arccos(np.clip(np.dot(u, v), -1, 1))
        t = min(radius / np.tan(ang / 2), 0.45 * lu, 0.45 * lv)
        a, b = p1 + u * t, p1 + v * t
        seg = np.linspace(pts[-1], a, n_seg + 1)[1:]
        pts.extend(seg)
        # quadratic Bezier fillet a -> b with control p1 (smooth, tangent-continuous)
        for s in np.linspace(0, 1, n_arc + 1)[1:]:
            pts.append((1 - s) ** 2 * a + 2 * (1 - s) * s * p1 + s ** 2 * b)
    seg = np.linspace(pts[-1], corners[-1], n_seg + 1)[1:]
    pts.extend(seg)
    return np.array(pts)
