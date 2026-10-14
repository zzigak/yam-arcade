"""Fit the station's convex-piece container model to a SAM 3D cup (assets/sam3d/<name>/).

The SAM 3D `collision_hollow.obj` is a watertight open-top shell (3 mm walls, voxel-ragged rim).  A single non-convex
mesh is a poor SPH boundary in Genesis (its grid SDF would need sub-millimetre cells across a 3 mm wall), so the
station keeps its own collision model - convex wedge pieces around a cavity - but takes the cavity from SAM 3D:

  * the asset is re-framed to the station's container convention: cavity axis at x = y = 0, base at z = 0, handle
    toward -x (SAM 3D puts the handle at +x and the bbox centre at the origin);
  * R_in(phi, z): the inner wall radius measured by casting rays from the cavity axis against horizontal sections
    of collision_hollow.obj (2 deg x ~1.5 mm grid), lightly smoothed (the shell is voxelised at 0.7-1.25 mm);
  * base piece = convex hull of the shell below the cavity floor (+ the raised floor), handle = convex hull of the
    shell outside the body on the handle side (it also fills the handle's hole: nothing goes there);
  * visual = SAM 3D visual mesh in the same frame, per-vertex colours kept.

`fit(name)` returns a ContainerSpec (polar table) plus the transformed meshes; results are cached in assets/.
"""
import json
import os

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter

from geometry import ContainerSpec

HERE = os.path.dirname(os.path.abspath(__file__))
SAM3D = os.path.abspath(os.path.join(HERE, "..", "..", "assets", "sam3d"))
CACHE = os.path.join(HERE, "assets", "sam3d_fit")

DENSITY = dict(mug=2300.0, tumbler=2500.0, teacup=2400.0, pitcher=1050.0)
RGBA = dict(mug=(0.25, 0.25, 0.27, 1.0), tumbler=(0.90, 0.95, 0.97, 0.25), teacup=(0.95, 0.94, 0.90, 1.0))


def frame_transform(info):
    """4x4: SAM 3D asset frame -> station container frame (cavity axis at origin, handle toward -x)."""
    cx, cy = info["collision"]["cavity_center_xy_m"]
    T = np.eye(4)
    T[:3, 3] = (-cx, -cy, 0.0)
    Rz = np.diag([-1.0, -1.0, 1.0, 1.0])      # yaw 180 deg
    return Rz @ T


def _ray_hits(segs, phis):
    """Nearest positive distance along rays from the origin at angles phis to 2D segments (N, 2, 2)."""
    d = np.stack([np.cos(phis), np.sin(phis)], 1)                    # (P, 2)
    a, b = segs[:, 0], segs[:, 1]                                     # (N, 2)
    e = b - a
    den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]   # (P, N)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (a[None, :, 0] * e[None, :, 1] - a[None, :, 1] * e[None, :, 0]) / den        # ray parameter
        u = (a[None, :, 0] * d[:, None, 1] - a[None, :, 1] * d[:, None, 0]) / den       # segment parameter
    ok = (np.abs(den) > 1e-12) & (t > 1e-6) & (u >= 0) & (u <= 1)
    t = np.where(ok, t, np.inf)
    return t.min(1)


def adaptive_bands(zs, R, tol=0.0004, max_h=0.035):
    """Band boundaries along z so that, at every angle, the straight chord of each band stays within `tol` of the
    measured wall (a band's convex piece has a straight inner face in z; a rounded cup bottom needs short bands)."""
    zf = np.linspace(zs[0], zs[-1], 400)
    Rf = np.stack([np.interp(zf, zs, R[:, j]) for j in range(R.shape[1])], 1)
    out, i0 = [0], 0
    while i0 < len(zf) - 1:
        i1 = i0 + 1
        while i1 + 1 < len(zf) and zf[i1 + 1] - zf[i0] <= max_h:
            t = (zf[i0:i1 + 2] - zf[i0]) / (zf[i1 + 1] - zf[i0])
            chord = Rf[i0][None] + t[:, None] * (Rf[i1 + 1] - Rf[i0])[None]
            if np.abs(chord - Rf[i0:i1 + 2]).max() > tol:
                break
            i1 += 1
        out.append(i1)
        i0 = i1
    return tuple(zf[out])


def fit(name, n_phi=180, n_z=48, smooth=(1.0, 1.5)):
    info = json.load(open(os.path.join(SAM3D, name, "info.json")))
    T = frame_transform(info)
    coll = trimesh.load(os.path.join(SAM3D, name, "collision_hollow.obj"), process=False)
    coll.apply_transform(T)
    vis = trimesh.load(os.path.join(SAM3D, name, "visual_lo.obj"), process=False)
    vis.apply_transform(T)
    z_floor = float(info["collision"]["floor_top_z_m"])
    z_rim = float(info["collision"]["fill_level_z_m"])
    zs = np.linspace(z_floor + 0.0015, z_rim - 0.004, n_z)   # the top ~3 mm of the rim is voxel-ragged
    phis = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    R = np.zeros((n_z, n_phi))
    for k, z in enumerate(zs):
        sec = coll.section(plane_origin=(0, 0, z), plane_normal=(0, 0, 1))
        segs = np.concatenate([sec.vertices[e.points][:, :2][np.stack([np.arange(len(e.points) - 1),
                                                                         np.arange(1, len(e.points))], 1)]
                               for e in sec.entities])
        R[k] = _ray_hits(segs, phis)
        bad = ~np.isfinite(R[k])
        if bad.any():   # a gap in a ragged section: take the level below
            assert k > 0 and bad.mean() < 0.2, f"{name}: rays escape the cavity at z = {z:.4f}"
            R[k, bad] = R[k - 1, bad]
    R = gaussian_filter(R, sigma=smooth, mode=("nearest", "wrap"))
    # extend the table to exactly the floor and the rim
    zs = np.r_[z_floor, zs, z_rim]
    R = np.vstack([R[:1], R, R[-1:]])
    wall = float(info["collision"]["wall_m"])
    r_out_max = float(R.max()) + wall
    # base: shell below the floor; handle: shell outside the body on the -x side
    V = coll.vertices
    rr = np.hypot(V[:, 0], V[:, 1])
    base_pts = V[V[:, 2] <= z_floor + 0.0005]
    handle_pts = V[(rr > r_out_max + 0.002) & (V[:, 0] < 0)]
    spec = ContainerSpec(
        name=name,
        z_floor=z_floor,
        z_rim=z_rim,
        prof_z=(z_floor, z_rim),
        prof_a=(float(R.mean()), float(R.mean())),
        prof_b=None,
        wall=wall,
        density=DENSITY[name],
        polar=(zs, np.r_[phis, np.pi], np.hstack([R, R[:, :1]])),
        base_hull=base_pts,
        handle_hulls=[handle_pts] if len(handle_pts) > 10 else None,
        visual_mesh=vis,
        mass_mesh=coll,
        coll_n_theta=28,
        coll_z=adaptive_bands(zs, R),
        rgba=RGBA[name],
    )
    return spec, info


def export_visual_glb(spec, path):
    """The SAM 3D visual in the container frame as a Y-up glTF (what the compositor and Genesis read), colours kept."""
    m = spec.visual_mesh.copy()
    to_yup = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)  # (x, y, z) -> (x, z, -y)
    m.apply_transform(to_yup)
    m.export(path)
    return path
