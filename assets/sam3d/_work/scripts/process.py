"""Post-process raw SAM 3D meshes into sim-ready assets (run with the genesis venv: trimesh, scipy, skimage,
fast_simplification, pymeshlab).

usage: python -I process.py NAME[:SEED] [...]      (per-object settings in OBJ below)

Frame convention of the outputs: Z-up, metres, base at z = 0, bounding box centred on x = y = 0.
Objects with a handle are yawed so the handle points to +x (so a pitcher's spout points to -x).
SAM 3D writes its .glb in glTF convention (Y-up, unit-size object); we map (x, y, z)_gltf -> (x, -z, y).
"""
import json, os, sys, time
import numpy as np
import trimesh
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

ROOT = os.path.expanduser("~/projects/yam-arcade/assets/sam3d")
RAW = f"{ROOT}/_work/sam3d_raw"
INP = f"{ROOT}/_work/inputs"

# dim: which dimension is set to `size` metres ('height' = z extent, 'length' = largest horizontal extent)
OBJ = {
    "pitcher": dict(raw="pitcher_brita2", dim="height", size=0.25, handle=True, y_extent=0.135,
                    y_why="photo is a pure side view, so SAM 3D had to guess the depth (17.5 cm, a round body); "
                          "Brita 10-cup pitchers are 26.9 x 14.0 x 26.2 cm (L x W x H), so y is rescaled to 13.5 cm", container=True, mode="carve",
                    wall=0.004, pitch=0.00125, zcut=0.90, R=0.015, faces=8000,
                    why="Brita 6-10 cup pitchers are 25-26 cm tall (e.g. Everyday 10-cup 26.2 cm); 25 cm chosen"),
    "pitcher_glass": dict(raw="pitcher_glass", dim="height", size=0.18, handle=True, container=True, mode="respect",
                          wall=0.003, pitch=0.001, R=0.012, faces=6000,
                          why="bulbous glass water jug; 18 cm tall keeps the brimful capacity near 2 L"),
    "urn": dict(raw="urn", dim="height", size=0.30, handle=False, container=True, mode="carve", wall=0.004,
                pitch=0.0015, zcut=0.86, R=0.02, faces=8000,
                why="~2 gallon (7.5 L) acrylic beverage dispenser, ~30 cm tall without a stand"),
    "mug": dict(raw="mug", dim="height", size=0.095, handle=True, container=True, mode="respect", wall=0.003,
                pitch=0.0007, R=0.009, faces=4000, why="standard 11-12 oz coffee mug, 9.5 cm tall (task spec)"),
    "mug_b": dict(raw="mug_b", dim="height", size=0.095, handle=True, container=True, mode="respect", wall=0.003,
                  pitch=0.0007, R=0.009, faces=4000, why="standard coffee mug, 9.5 cm tall"),
    "tumbler": dict(raw="tumbler", dim="height", size=0.14, handle=False, container=True, mode="respect",
                    wall=0.003, pitch=0.0007, R=0.008, faces=4000,
                    why="tall drinking glass (pint/highball class), 14 cm tall (task spec)"),
    "tumbler_b": dict(raw="tumbler_b", dim="height", size=0.14, handle=False, container=True, mode="respect",
                      wall=0.003, pitch=0.0007, R=0.008, faces=4000, why="conical drinking glass, 14 cm tall"),
    "teacup": dict(raw="teacup", dim="height", size=0.065, handle=True, container=True, mode="respect",
                   wall=0.003, pitch=0.0006, R=0.007, faces=4000,
                   why="porcelain teacup: 6-7 cm tall, ~9 cm rim; 6.5 cm chosen"),
    "wineglass": dict(raw="wineglass", dim="height", size=0.15, handle=False, container=True, mode="respect",
                      wall=0.003, pitch=0.0006, R=0.006, faces=4000,
                      why="small stemmed (sherry / small wine) glass, 15 cm tall"),
    "box": dict(raw="box", dim="length", size=0.32, handle=False, container=True, mode="carve", wall=0.004,
                pitch=0.0015, R=0.004, zcut=0.85, hull=True, faces=3000, align=True,
                why="open cardboard box sized for a 1.2 x 0.7 m table: 32 cm long (photo proportions give the rest)"),
    "jar": dict(raw="jar", dim="height", size=0.10, handle=False, container=True, mode="respect", wall=0.003,
                pitch=0.0007, zcut=0.86, R=0.01, faces=4000, align=True,
                why="hexagonal jam jar (~280 ml) with a 63-70 mm twist-off lid, ~10 cm tall"),
    "jar_lid": dict(raw="jar_lid", dim="length", size=0.070, handle=False, container=False,
                    why="regular-mouth mason jar band/lid: 70 mm outer diameter"),
    "mahjong_tile": dict(raw="mahjong_tile", dim="max", size=0.030, handle=False, container=False, align=True,
                         why="Chinese mahjong tile ~30 x 22 x 16 mm; largest dimension set to 30 mm"),
}

OBJ["pitcher_v1"] = dict(OBJ["pitcher"], raw="pitcher_brita",
                         crop=[((0.076, -0.05, -0.01), (0.2, 0.05, 0.0895))],
                         recolor=[((0.076, -0.05, 0.08), (0.2, 0.05, 0.125), 8)])  # string hanging off the handle base  # first mask (string on the handle, handle hole filled)


def load_raw(raw, seed):
    m = trimesh.load(f"{RAW}/{raw}_s{seed}.glb").to_geometry()
    v = np.asarray(m.vertices, dtype=np.float64)
    v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)  # glTF Y-up -> Z-up
    col = np.asarray(m.visual.vertex_colors)[:, :4] if m.visual.kind == "vertex" else None
    return trimesh.Trimesh(v, m.faces, vertex_colors=col, process=False)


def rot_between(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def upright(m):
    """Rotate so the largest convex-hull facet facing roughly down becomes the base (normal -> -z)."""
    h = m.convex_hull
    n, area = h.facets_normal, h.facets_area
    ok = n[:, 2] < -np.cos(np.radians(30))
    if not ok.any():
        return np.eye(3), 0.0
    i = np.argmax(np.where(ok, area, -1))
    R = rot_between(n[i], np.array([0, 0, -1.0]))
    return R, float(np.degrees(np.arccos(np.clip(-n[i, 2], -1, 1))))


def voxelize_parity(m, pitch, pad=4):
    """Solid occupancy of a watertight mesh by z-column ray parity (vectorised over triangles)."""
    lo = m.bounds[0] - pad * pitch
    shape = np.ceil((m.bounds[1] - m.bounds[0]) / pitch).astype(int) + 2 * pad
    tri = m.triangles  # (F,3,3)
    x, y = tri[:, :, 0], tri[:, :, 1]
    i0 = np.ceil((x.min(1) - lo[0]) / pitch - 0.5).astype(int); i1 = np.floor((x.max(1) - lo[0]) / pitch - 0.5).astype(int)
    j0 = np.ceil((y.min(1) - lo[1]) / pitch - 0.5).astype(int); j1 = np.floor((y.max(1) - lo[1]) / pitch - 0.5).astype(int)
    ni, nj = np.maximum(i1 - i0 + 1, 0), np.maximum(j1 - j0 + 1, 0)
    cnt = ni * nj
    t = np.repeat(np.arange(len(tri)), cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    nj_r = np.maximum(np.repeat(nj, cnt), 1)
    ii = np.repeat(i0, cnt) + off // nj_r; jj = np.repeat(j0, cnt) + off % nj_r
    px = lo[0] + (ii + 0.5) * pitch; py = lo[1] + (jj + 0.5) * pitch
    A, B, C = tri[t, 0], tri[t, 1], tri[t, 2]
    def edge(P, Q, qx, qy):
        return (Q[:, 0] - P[:, 0]) * (qy - P[:, 1]) - (Q[:, 1] - P[:, 1]) * (qx - P[:, 0])
    w0, w1, w2 = edge(B, C, px, py), edge(C, A, px, py), edge(A, B, px, py)
    area = w0 + w1 + w2
    inside = (np.abs(area) > 1e-18) & (((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0)))
    w0, w1, w2, area = w0[inside], w1[inside], w2[inside], area[inside]
    z = (w0 * A[inside, 2] + w1 * B[inside, 2] + w2 * C[inside, 2]) / area
    ii, jj = ii[inside], jj[inside]
    k = np.floor((z - lo[2]) / pitch - 0.5).astype(int) + 1
    k = np.clip(k, 0, shape[2])
    T = np.zeros((shape[0], shape[1], shape[2] + 1), np.int32)
    np.add.at(T, (ii, jj, k), 1)
    occ = (np.cumsum(T, axis=2)[:, :, :shape[2]] % 2) == 1
    odd = (T.sum(2) % 2) == 1
    occ[odd] = False  # columns that grazed an edge an odd number of times: drop, then patch from neighbours
    if odd.any():
        occ = ndi.binary_closing(occ, iterations=1) | occ
    # thin walls can fall between column samples: add every voxel the surface passes through
    pts = np.concatenate([m.vertices, m.triangles_center, (tri[:, 0] + tri[:, 1]) / 2, (tri[:, 1] + tri[:, 2]) / 2,
                          (tri[:, 2] + tri[:, 0]) / 2])
    q = np.clip(np.floor((pts - lo) / pitch).astype(int), 0, np.array(shape) - 1)
    occ[q[:, 0], q[:, 1], q[:, 2]] = True
    return occ, lo


def slice_fill(M):
    E = np.zeros_like(M)
    for k in range(M.shape[2]):
        if M[:, :, k].any():
            E[:, :, k] = ndi.binary_fill_holes(M[:, :, k])
    return E


def ball(r):
    r = int(np.ceil(r)); g = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (g ** 2).sum(0) <= r * r + 0.5


def disk(r):
    r = int(np.ceil(r)); g = np.mgrid[-r:r + 1, -r:r + 1]
    return (g ** 2).sum(0) <= r * r + 0.5


def erode_dist(X, d_vox):
    """Euclidean erosion by d voxels (via EDT of the complement). The top slice is repeated upward first, so
    regions that reach the top of the grid (the opening extension) are treated as unbounded above."""
    n = int(np.ceil(d_vox)) + 3
    Xt = np.concatenate([X, np.repeat(X[:, :, -1:], n, axis=2)], axis=2)
    return ndi.distance_transform_edt(np.pad(Xt, 1))[1:-1, 1:-1, 1:-1][:, :, :X.shape[2]] > d_vox


def build_hollow(M, c, pitch):
    """Return final solid F (open-top container with walls >= wall), the cavity, and diagnostics."""
    t = c["wall"] / pitch
    E = slice_fill(M)
    zs = np.where(E.any((0, 1)))[0]; kb, kt = zs[0], zs[-1]
    body = ndi.binary_opening(E, structure=ball(c["R"] / pitch))
    lab, n = ndi.label(body)
    if n > 1:
        body = lab == 1 + np.argmax(ndi.sum(body, lab, range(1, n + 1)))
    if c["mode"] == "carve":
        kc = int(round(kb + c["zcut"] * (kt - kb)))
        foot = body[:, :, kc]
        B = body.copy(); B[:, :, kc:] |= foot[:, :, None]
        Cn = erode_dist(B, t)
    else:
        Cv = E & ~M
        # keep only the main cavity (the component reachable from the top, largest)
        lab, n = ndi.label(Cv)
        if n > 1:
            Cv = lab == 1 + np.argmax(ndi.sum(Cv, lab, range(1, n + 1)))
        zc = np.where(Cv.any((0, 1)))[0]
        kc = int(round(kb + c["zcut"] * (kt - kb))) if c.get("zcut") else zc[-1] - 1
        foot = Cv[:, :, kc] | (body[:, :, kc] & ndi.binary_dilation(Cv[:, :, kc], iterations=2))
        Eext = E.copy(); Eext[:, :, kc:] |= foot[:, :, None]
        Cext = Cv.copy(); Cext[:, :, kc:] |= foot[:, :, None]
        Cn = Cext & erode_dist(Eext, t)
    Cn[:, :, :kb + max(1, int(round(t)))] = False  # solid floor at least `wall` thick
    F = E & ~Cn
    # tidy: keep the largest solid component, drop specks
    lab, n = ndi.label(F)
    if n > 1:
        F = lab == 1 + np.argmax(ndi.sum(F, lab, range(1, n + 1)))
    return F, Cn, E, body


def capacity(F, Cn, pitch):
    """Largest water volume the open container holds upright: the highest level at which the air below it that
    sits in the cavity is still enclosed laterally (raising it further spills over the lowest rim point)."""
    A = ~F
    def water(k):
        lab, _ = ndi.label(A[:, :, :k])
        border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                           lab[:, :, 0].ravel()]))
        return ~np.isin(lab, border) & (lab > 0)
    zc = np.where(Cn.any((0, 1)))[0]
    lo_k, hi_k = zc[0] + 2, F.shape[2]
    best = water(lo_k)
    if best.sum() == 0:
        return 0.0, 0
    while lo_k < hi_k:
        mid = (lo_k + hi_k + 1) // 2
        w = water(mid)
        if w.sum() > 0:
            lo_k, best = mid, w
        else:
            hi_k = mid - 1
    return float(best.sum() * pitch ** 3), lo_k


def mc_mesh(F, lo, pitch, faces):
    from skimage import measure
    P = np.pad(F, 2).astype(np.float32)
    P = ndi.gaussian_filter(P, 0.6)
    v, f, _, _ = measure.marching_cubes(P, 0.5)
    v = (v - 2 + 0.5) * pitch + lo
    m = trimesh.Trimesh(v, f[:, ::-1], process=True)
    if m.volume < 0:
        m.invert()
    trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=6)
    m = decimate(m, faces)
    return m


def decimate(m, faces):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(m.vertices, m.faces))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(faces), preservetopology=True, preservenormal=True,
                                                qualitythr=0.5, planarquadric=True, autoclean=True)
    mm = ms.current_mesh()
    out = trimesh.Trimesh(mm.vertex_matrix(), mm.face_matrix(), process=True)
    if not out.is_watertight:
        trimesh.repair.fill_holes(out)
    trimesh.repair.fix_normals(out)
    return out


def decimate_colored(m, faces):
    import fast_simplification
    ratio = 1 - faces / len(m.faces)
    if ratio <= 0:
        return m.copy()
    v, f = fast_simplification.simplify(m.vertices.astype(np.float32), m.faces.astype(np.int64), ratio)
    out = trimesh.Trimesh(v, f, process=True)
    _, idx = cKDTree(m.vertices).query(out.vertices)
    out.visual = trimesh.visual.ColorVisuals(out, vertex_colors=np.asarray(m.visual.vertex_colors)[idx])
    return out


def handle_yaw(E, body, lo, pitch):
    H = E & ~ndi.binary_dilation(body, iterations=2)
    lab, n = ndi.label(H)
    if n == 0:
        return 0.0
    H = lab == 1 + np.argmax(ndi.sum(H, lab, range(1, n + 1)))
    hc = np.argwhere(H).mean(0)[:2]; bc = np.argwhere(body).mean(0)[:2]
    d = hc - bc
    return float(np.arctan2(d[1], d[0]))


def process(name, seed):
    t0 = time.time()
    c = OBJ[name]
    outd = os.environ.get("OUTDIR", f"{ROOT}/{name}"); os.makedirs(outd, exist_ok=True)
    m = load_raw(c["raw"], seed)
    R, tilt = upright(m)
    m.vertices = m.vertices @ R.T
    ext = m.extents
    s = {"height": c["size"] / ext[2], "length": c["size"] / max(ext[0], ext[1]), "max": c["size"] / max(ext)}[c["dim"]]
    m.vertices *= s
    yaw = 0.0
    if c.get("align"):
        xy = m.convex_hull.vertices[:, :2]; xy = xy - xy.mean(0)
        w, V = np.linalg.eigh(np.cov(xy.T)); a = V[:, -1]
        yaw = -np.arctan2(a[1], a[0])
    if c.get("handle"):
        pitch = max(c["pitch"], 0.0015)
        M0, lo0 = voxelize_parity(m, pitch)
        E0 = slice_fill(M0); b0 = ndi.binary_opening(E0, structure=ball(c["R"] / pitch))
        yaw = -handle_yaw(E0, b0, lo0, pitch)
    if yaw:
        cz, sz = np.cos(yaw), np.sin(yaw)
        m.vertices = m.vertices @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]).T
    ext = m.extents  # re-apply the size after the yaw (a yaw changes axis-aligned horizontal extents)
    s2 = {"height": c["size"] / ext[2], "length": c["size"] / max(ext[0], ext[1]), "max": c["size"] / max(ext)}[c["dim"]]
    m.vertices *= s2; s *= s2
    yfix = None
    if c.get("y_extent"):  # depth is unobservable from a pure side view: set it from real-product proportions
        yfix = c["y_extent"] / m.extents[1]
        m.vertices[:, 1] *= yfix
    b = m.bounds
    m.vertices -= np.array([(b[0, 0] + b[1, 0]) / 2, (b[0, 1] + b[1, 1]) / 2, b[0, 2]])
    for (x0, y0, z0), (x1, y1, z1) in c.get("crop", []):  # remove photo artefacts (final frame, metres)
        v = m.vertices
        inside = (v[:, 0] > x0) & (v[:, 0] < x1) & (v[:, 1] > y0) & (v[:, 1] < y1) & (v[:, 2] > z0) & (v[:, 2] < z1)
        keep = ~inside[m.faces].any(1)
        m.update_faces(keep); m.remove_unreferenced_vertices()
        parts = m.split(only_watertight=False)
        if len(parts) > 1:  # drop floating crumbs left by the cut
            big = max(parts, key=lambda q: len(q.faces))
            m = trimesh.util.concatenate([q for q in parts if len(q.faces) > 0.01 * len(big.faces)])
        trimesh.repair.fill_holes(m)
    for (x0, y0, z0), (x1, y1, z1), thr in c.get("recolor", []):  # paint photo artefacts (e.g. a string) grey
        v = m.vertices; col = np.asarray(m.visual.vertex_colors).copy()
        box = (v[:, 0] > x0) & (v[:, 0] < x1) & (v[:, 1] > y0) & (v[:, 1] < y1) & (v[:, 2] > z0) & (v[:, 2] < z1)
        bad = box & ((col[:, 0].astype(int) - col[:, 2].astype(int)) >= thr)
        ref = box & ~bad
        col[bad, :3] = np.median(col[ref, :3], axis=0).astype(np.uint8)
        m.visual.vertex_colors = col
    info = dict(name=name, source_raw=f"{c['raw']}_s{seed}.glb", seed=seed, frame="Z-up, metres, base at z=0, bbox centred in x/y"
                + (", handle along +x" if c.get("handle") else ""),
                target=dict(dim=c["dim"], size_m=c["size"], why=c["why"]), sam3d_to_zup="(x,y,z)_gltf -> (x,-z,y)",
                upright_correction_deg=round(tilt, 2), yaw_applied_deg=round(float(np.degrees(yaw)), 1),
                scale_from_sam3d_units=round(float(s), 5),
                y_rescale=None if yfix is None else dict(factor=round(float(yfix), 4), why=c.get("y_why")))
    vis = decimate_colored(m, 200000)
    lo_v = decimate_colored(m, 40000)
    gl = vis.copy(); gl.vertices = np.stack([vis.vertices[:, 0], vis.vertices[:, 2], -vis.vertices[:, 1]], 1)
    gl.export(f"{outd}/visual.glb")  # glTF spec: Y-up, so Blender/three.js import it upright (Z-up in Blender)
    vis.export(f"{outd}/visual.obj", include_color=True)
    lo_v.export(f"{outd}/visual_lo.obj", include_color=True)
    info.update(bbox_m=np.round(vis.extents, 4).tolist(), visual_faces=len(vis.faces), visual_lo_faces=len(lo_v.faces),
                visual_watertight=bool(vis.is_watertight), visual_volume_m3=float(vis.volume) if vis.is_watertight else None,
                raw_faces=len(m.faces), raw_watertight=bool(m.is_watertight))
    if c.get("container"):
        pitch = c["pitch"]
        M, lo = voxelize_parity(m.convex_hull if c.get("hull") else m, pitch)
        F, Cn, E, body = build_hollow(M, c, pitch)
        # sam3d's own cavity, for the report
        Cv = E & ~M
        cap, kfill = capacity(F, Cn, pitch)
        col = mc_mesh(F, lo, pitch, c["faces"])
        col.export(f"{outd}/collision_hollow.obj")
        np.savez_compressed(f"{ROOT}/_work/vox_{name}_s{seed}.npz", F=F, M=M, lo=lo, pitch=pitch)
        zs = np.where(Cn.any((0, 1)))[0]
        cav_xy = (np.argwhere(Cn).mean(0)[:2] + 0.5) * pitch + lo[:2]
        info.update(collision=dict(file="collision_hollow.obj", faces=len(col.faces), watertight=bool(col.is_watertight),
                                   volume_m3=float(col.volume), bbox_m=np.round(col.extents, 4).tolist(),
                                   wall_m=c["wall"], voxel_m=pitch, mode=c["mode"],
                                   sam3d_cavity_frac_of_envelope=round(float(Cv.sum() / max(E.sum(), 1)), 3),
                                   floor_top_z_m=round(float(lo[2] + zs[0] * pitch), 4),
                                   cavity_center_xy_m=np.round(cav_xy, 4).tolist(),
                                   fill_level_z_m=round(float(lo[2] + kfill * pitch), 4)),
                    capacity_l=round(cap * 1000, 3))
    json.dump(info, open(f"{outd}/info.json", "w"), indent=1)
    print(name, seed, f"{time.time()-t0:.0f}s", json.dumps({k: info[k] for k in ("bbox_m", "upright_correction_deg")}),
          info.get("capacity_l"), info.get("collision", {}).get("watertight"), info.get("collision", {}).get("faces"))
    return info


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        name, _, seed = arg.partition(":")
        process(name, int(seed or 42))
