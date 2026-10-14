"""Signed minimum distance between the station's collision meshes at logged poses (offline, no physics).

Used by checks.py (run with ~/projects/yam-arcade/.venvs/jars/bin/python, needs trimesh + rtree).
Distance between two closed meshes = min over (vertices of A against B) and (vertices of B against A) of the
point-to-triangle distance, signed negative when the vertex is inside the other mesh (ray-parity containment within
0.05 mm of the surface, closest-face normal beyond). Vertex-based, so
an edge-edge crossing without a vertex inside would be missed. Where parts touch (threads, rims, skirts, bases) the meshes
have 1.1-1.9 mm circumferential spacing (120 segments) and facet sag <= 0.015 mm, so the vertex distances track the
true gap to well within the 0.2 mm d_hat contact gap.
"""
import json
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
PARAMS = json.load(open(os.path.join(ASSETS, "params.json")))
_CACHE = {}
# trimesh's point-triangle closest point uses absolute tolerances: with these meshes in metres, sub-millimetre
# triangles give wrong distances (0.218 mm reported for a true 0.0039 mm). Everything here works in MILLIMETRES
# internally; the public functions take and return metres.
MM = 1000.0


def mesh(fn):
    """The mesh in millimetres."""
    if fn not in _CACHE:
        m = trimesh.load(os.path.join(ASSETS, fn), process=False)
        m.apply_scale(MM)
        _CACHE[fn] = m
    return _CACHE[fn]


def jar_mesh(name):
    return mesh(PARAMS["parts"][name]["jar_col"]["file"])


def lid_mesh(name):
    return mesh(PARAMS["parts"][name]["lid_col"]["file"])


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _signed(m, pts, near=0.05):
    """Point-triangle distance (mm, mesh and points in mm), negative inside. The sign comes from a ray-parity
    containment test for every point within `near` mm (0.05 mm, closer than any resting IPC contact gap) of the surface (a closest-face normal is ambiguous at concave edges, e.g. thread roots), and
    from the closest face's normal for points farther away (unambiguous there)."""
    c, d, tri = trimesh.proximity.closest_point(m, pts)
    s = np.einsum("ij,ij->i", pts - c, m.face_normals[tri])
    inside = s < 0
    close = d < near
    if close.any():
        inside[close] = m.contains(pts[close])
    return np.where(inside, -d, d)


def pair_distance(mA, pA, RA, mB, pB, RB, margin=10.0):
    """Signed min distance (m) between mm-mesh A at (pA [m], RA) and mm-mesh B at (pB [m], RB). Returns (dist_m, where_m)."""
    pA, pB = np.asarray(pA) * MM, np.asarray(pB) * MM
    # B's vertices in A's frame and vice versa; only vertices inside the other's (inflated) bounding box matter
    vB_in_A = (RA.T @ ((RB @ mB.vertices.T).T + pB - pA).T).T
    vA_in_B = (RB.T @ ((RA @ mA.vertices.T).T + pA - pB).T).T
    best, where = np.inf, None
    for m, pts, p0, R0 in ((mA, vB_in_A, pA, RA), (mB, vA_in_B, pB, RB)):
        lo, hi = m.bounds[0] - margin, m.bounds[1] + margin
        sel = np.all((pts >= lo) & (pts <= hi), axis=1)
        if not sel.any():
            continue
        d = _signed(m, pts[sel])
        i = int(np.argmin(d))
        if d[i] < best:
            best, where = float(d[i]), (R0 @ pts[sel][i] + p0)
    return best / MM, (None if where is None else where / MM)


def min_z_world(m, p, R):
    """Lowest point (m) of an mm-mesh at pose (p [m], R): exact from the vertices for a horizontal plane."""
    return float(((R @ m.vertices.T).T / MM + p)[:, 2].min())
