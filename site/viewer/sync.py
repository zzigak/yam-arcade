#!/usr/bin/env python
"""YAM arcade 3D viewer: convert every simulation export under out/ into the compact format the three.js viewer reads.

    ~/simquarry/genesis-world/.venv/bin/python site/viewer/sync.py [--force] [--only RUN_ID] [-q]

Scans out/ recursively for meta.json: out/<station>/ (the export the compositor renders) and every in-progress run in
a subdirectory (out/cables/tests/drop1/, out/jars/twist_test/, ...). A run that is new, or whose files changed (mtime
signature), is converted into

    site/viewer/data/<run_id>/manifest.json         objects, kinds, materials, fps, stride, station, label, checks
    site/viewer/data/<run_id>/<ver>/traj.bin        float32, per moving rigid n x [px py pz qw qx qy qz]
    site/viewer/data/<run_id>/<ver>/dyn.bin         uint16, rods and particles quantised to their bounding box
    site/viewer/data/<run_id>/<ver>/seq/<obj>/NNNNN.bin   meshseq frames (water surfaces), loaded lazily
    site/viewer/data/_meshes/<sha1>.glb             every rigid mesh as GLB, content-addressed, shared by all runs

and site/viewer/data/index.json lists the runs. Positions stay in the station-local frame (z up, metres); quaternions
stay (w, x, y, z); the viewer places each station with layout.STATIONS. Exports still being written (traj.npz fails
to load, frame counts disagree, files touched in the last few seconds) are skipped and retried on the next pass.

Kept up to date by the user timer arcade-viewer-sync (every 2 min, Nice 19, CPUWeight 5, idle I/O):
    systemctl --user list-timers arcade-viewer-sync.timer         # when it runs next
    journalctl --user -u arcade-viewer-sync -n 40                  # what the last passes did
    systemctl --user stop arcade-viewer-sync.timer                 # pause (until the next login/reboot)
    systemctl --user disable --now arcade-viewer-sync.timer        # stop for good
Unit files: ~/.config/systemd/user/arcade-viewer-sync.{service,timer}.
"""
import argparse
import hashlib
import io
import json
import math
import os
import shutil
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "out")
DATA = os.path.join(HERE, "data")
MESHES = os.path.join(DATA, "_meshes")
sys.path.insert(0, ROOT)
import layout as L  # noqa: E402

FORMAT = 3                    # bump to force every run to be reconverted
BUDGET = 34e6                 # bytes of per-frame data per run (traj + dyn + seq) before frames are strided
MESH_FACE_LIMIT = 60000       # rigid meshes above this are decimated ...
MESH_FACE_TARGET = 30000      # ... to this
SEQ_FACE_LIMIT = 40000        # meshseq frames above this are decimated
PARTICLE_LIMIT = 30000        # particles per object (fixed subset, same points every frame)
TEX_MAX = 1024                # texture edge, pixels
SETTLE_S = 4.0                # files modified more recently than this are treated as still being written
VERBOSE = True


def log(*a):
    if VERBOSE:
        print("[sync]", *a, flush=True)


# ============================================================================ small helpers
def sha1_files(paths, extra=""):
    h = hashlib.sha1(f"{FORMAT}|{extra}".encode())
    for p in paths:  # content only (not names), so identical meshes in different runs share one GLB
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:20]


def stat_sig(p):
    try:
        st = os.stat(p)
        return [os.path.relpath(p, OUT), st.st_mtime_ns, st.st_size]
    except FileNotFoundError:
        return [os.path.relpath(p, OUT), None, None]


def write_json_atomic(path, obj, indent=None):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, separators=None if indent else (",", ":"))
    os.replace(tmp, path)


def checks_summary(path):
    """Summarise the station's checks.json (formats differ between stations) -> dict or None."""
    try:
        c = json.load(open(path))
    except Exception:
        return None
    rows = []
    items = c.get("checks", c) if isinstance(c, dict) else c
    if isinstance(items, dict):
        items = [dict(v, name=k) if isinstance(v, dict) else dict(name=k, value=v) for k, v in items.items()]
    if not isinstance(items, list):
        return None
    for r in items:
        if not isinstance(r, dict):
            continue
        ok = next((r[k] for k in ("passed", "pass_", "pass", "ok") if k in r), None)
        rows.append(dict(name=str(r.get("name", "?")), ok=ok))
    allok = c.get("all_ok", c.get("all_pass")) if isinstance(c, dict) else None
    fails = [r["name"] for r in rows if r["ok"] is False]
    return dict(n=len(rows), n_pass=sum(1 for r in rows if r["ok"] is True), fail=fails[:12],
                all_ok=bool(allok) if allok is not None else (len(fails) == 0 and len(rows) > 0))


# ============================================================================ normals with sharp edges
def corner_normals(V, F, angle_deg=35.0):
    """Blender-like 'shade smooth + sharp from angle': per face corner, the area-weighted average of the normals of the
    faces around that vertex that are within angle_deg of this face. -> (3F, 3)"""
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    e1, e2 = V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]
    fnw = np.cross(e1, e2)                                   # area-weighted
    fn = fnw / np.maximum(np.linalg.norm(fnw, axis=1, keepdims=True), 1e-20)
    nF, nV = len(F), len(V)
    cv = F.ravel()
    cf = np.repeat(np.arange(nF), 3)
    order = np.argsort(cv, kind="stable")
    cv_s, cf_s = cv[order], cf[order]
    counts = np.bincount(cv, minlength=nV)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    deg = counts[cv_s]
    if int((deg.astype(np.int64)).sum()) > 40_000_000:      # pathological valence: fall back to flat
        return np.repeat(fn, 3, axis=0)
    i_idx = np.repeat(np.arange(len(cv_s)), deg)
    offs = np.arange(len(i_idx)) - np.repeat(np.cumsum(deg) - deg, deg)
    j_idx = starts[cv_s][i_idx] + offs
    fi, fj = cf_s[i_idx], cf_s[j_idx]
    ok = (fn[fi] * fn[fj]).sum(1) > math.cos(math.radians(angle_deg))
    acc = np.zeros((len(cv_s), 3))
    np.add.at(acc, i_idx[ok], fnw[fj[ok]])
    out = np.empty_like(acc)
    out[order] = acc
    n = np.linalg.norm(out, axis=1, keepdims=True)
    bad = n[:, 0] < 1e-20
    out = out / np.maximum(n, 1e-20)
    out[bad] = np.repeat(fn, 3, axis=0)[bad]
    return out


def split_by_corner_normals(V, F, CN, extra=None):
    """Make per-vertex normals from per-corner normals by splitting vertices whose corners disagree.
    extra: dict of per-vertex arrays carried along. -> V, F, N, extra"""
    q = np.round(CN * 1000).astype(np.int32)
    key = np.concatenate([F.reshape(-1, 1).astype(np.int64), q.astype(np.int64)], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    first = np.zeros(len(uniq), np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    vid = F.ravel()[first]
    N = CN[first]
    ex = {k: v[vid] for k, v in (extra or {}).items()}
    return V[vid], inv.reshape(-1, 3), N, ex


def decimate(V, F, target):
    try:
        import fast_simplification
        red = 1.0 - float(target) / len(F)
        V2, F2 = fast_simplification.simplify(np.asarray(V, np.float32), np.asarray(F, np.int32), target_reduction=red)
        if len(F2) > 0:
            return np.asarray(V2, np.float64), np.asarray(F2, np.int64)
    except Exception as e:  # noqa: BLE001
        log(f"  decimation failed ({e}); keeping full mesh")
    return V, F


# ============================================================================ GLB writer
class GLB:
    def __init__(self):
        self.bin = bytearray()
        self.views, self.accessors, self.meshes_prims, self.materials, self.images, self.textures = [], [], [], [], [], []
        self._img_ids = {}

    def _view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        off = len(self.bin)
        self.bin += data
        v = dict(buffer=0, byteOffset=off, byteLength=len(data))
        if target:
            v["target"] = target
        self.views.append(v)
        return len(self.views) - 1

    def _acc(self, arr, ctype, typ, target=34962, normalized=False, minmax=False):
        arr = np.ascontiguousarray(arr)
        vi = self._view(arr.tobytes(), target)
        a = dict(bufferView=vi, componentType=ctype, count=int(len(arr) if typ != "SCALAR" else arr.size), type=typ)
        if normalized:
            a["normalized"] = True
        if minmax:
            a["min"] = arr.min(0).tolist()
            a["max"] = arr.max(0).tolist()
        self.accessors.append(a)
        return len(self.accessors) - 1

    def image(self, img):
        """PIL image -> texture index (JPEG unless it has alpha), resized to TEX_MAX."""
        key = id(img)
        if key in self._img_ids:
            return self._img_ids[key]
        from PIL import Image
        im = img
        if max(im.size) > TEX_MAX:
            s = TEX_MAX / max(im.size)
            im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.LANCZOS)
        buf = io.BytesIO()
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im.convert("RGBA").save(buf, format="PNG", optimize=True)
            mime = "image/png"
        else:
            im.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
            mime = "image/jpeg"
        vi = self._view(buf.getvalue())
        self.images.append(dict(bufferView=vi, mimeType=mime))
        self.textures.append(dict(source=len(self.images) - 1, sampler=0))
        self._img_ids[key] = len(self.textures) - 1
        return len(self.textures) - 1

    def prim(self, V, F, N, uv=None, colors=None, base=(1, 1, 1, 1), tex=None, own=False):
        attrs = dict(POSITION=self._acc(np.asarray(V, np.float32), 5126, "VEC3", minmax=True),
                     NORMAL=self._acc(np.asarray(N, np.float32), 5126, "VEC3"))
        if uv is not None:
            attrs["TEXCOORD_0"] = self._acc(np.asarray(uv, np.float32), 5126, "VEC2")
        if colors is not None:
            attrs["COLOR_0"] = self._acc(np.asarray(colors, np.uint8), 5121, "VEC4", normalized=True)
        idx = np.asarray(F).ravel()
        if len(V) < 65536:
            ia = self._acc(idx.astype(np.uint16), 5123, "SCALAR", target=34963)
        else:
            ia = self._acc(idx.astype(np.uint32), 5125, "SCALAR", target=34963)
        mat = dict(pbrMetallicRoughness=dict(baseColorFactor=[float(c) for c in base], metallicFactor=0.0,
                                             roughnessFactor=0.5), extras=dict(own=bool(own)))
        if tex is not None:
            mat["pbrMetallicRoughness"]["baseColorTexture"] = dict(index=tex)
        self.materials.append(mat)
        self.meshes_prims.append(dict(attributes=attrs, indices=ia, material=len(self.materials) - 1))

    def bytes(self):
        while len(self.bin) % 4:
            self.bin.append(0)
        gltf = dict(asset=dict(version="2.0", generator="yam-arcade viewer sync"), scene=0,
                    scenes=[dict(nodes=[0])], nodes=[dict(mesh=0)], meshes=[dict(primitives=self.meshes_prims)],
                    accessors=self.accessors, bufferViews=self.views, buffers=[dict(byteLength=len(self.bin))],
                    materials=self.materials)
        if self.images:
            gltf.update(images=self.images, textures=self.textures,
                        samplers=[dict(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)])
        js = json.dumps(gltf, separators=(",", ":")).encode()
        js += b" " * ((4 - len(js) % 4) % 4)
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
                + struct.pack("<II", len(self.bin), 0x004E4942) + bytes(self.bin))


# ============================================================================ mesh conversion
def obj_deps(path):
    """An OBJ's .mtl and the textures it references (paths that exist)."""
    deps = []
    if not path.lower().endswith(".obj"):
        return deps
    d = os.path.dirname(path)
    for line in open(path, errors="ignore"):
        if line.startswith("mtllib"):
            mtl = os.path.join(d, line.split(None, 1)[1].strip())
            if os.path.exists(mtl):
                deps.append(mtl)
                for l2 in open(mtl, errors="ignore"):
                    sp = l2.split()
                    if sp and sp[0].startswith("map_"):
                        tex = os.path.join(d, sp[-1])
                        if os.path.exists(tex):
                            deps.append(tex)
        elif line.startswith(("v ", "f ")):
            break
    return deps


def obj_has(path, prefix):
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith(prefix):
                return True
    return False


def box_uv(V, F):
    """Blender's box projection with Object coordinates (period 1 m): unindex and project each face on its dominant
    axis plane."""
    P = V[F.reshape(-1)]
    e1, e2 = V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]
    ax = np.argmax(np.abs(np.cross(e1, e2)), axis=1).repeat(3)
    uv = np.where(ax[:, None] == 0, P[:, [1, 2]], np.where(ax[:, None] == 1, P[:, [0, 2]], P[:, [0, 1]]))
    return P, np.arange(len(P)).reshape(-1, 3), uv


def convert_mesh(path, texture=None, mesh_up=None):
    """Mesh file (+ optional spec texture) -> data/_meshes/<sha>.glb. Returns (relative url, info dict)."""
    import trimesh
    deps = [path] + obj_deps(path) + ([texture] if texture else [])
    h = sha1_files(deps, extra=f"up={mesh_up}")
    dst = os.path.join(MESHES, f"{h}.glb")
    info_p = dst + ".json"
    if os.path.exists(dst) and os.path.exists(info_p):
        return f"_meshes/{h}.glb", json.load(open(info_p))
    ext = os.path.splitext(path)[1].lower()
    gltf = ext in (".glb", ".gltf")
    yup = (gltf and mesh_up != "z") or (not gltf and mesh_up == "y")
    file_normals = gltf or (ext == ".obj" and obj_has(path, "vn "))
    has_mtl = ext == ".obj" and any(p.lower().endswith(".mtl") for p in deps)
    sc = trimesh.load(path, force="scene", process=False)
    tex_img = None
    if texture:
        from PIL import Image
        tex_img = Image.open(texture)
        tex_img.load()
    g = GLB()
    n_in = n_out = 0
    for node in sc.graph.nodes_geometry:
        T, gname = sc.graph[node]
        geom = sc.geometry[gname]
        if not hasattr(geom, "faces") or len(geom.faces) == 0:
            continue
        V = np.asarray(geom.vertices, np.float64) @ T[:3, :3].T + T[:3, 3]
        F = np.asarray(geom.faces, np.int64)
        n_in += len(F)
        Nf = geom._cache.cache.get("vertex_normals") if file_normals else None
        N = None
        if Nf is not None and len(Nf) == len(V):
            N = np.asarray(Nf, np.float64) @ T[:3, :3].T
            N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        vis = geom.visual
        uv = None
        colors = None
        base = (1.0, 1.0, 1.0, 1.0)
        img = None
        own = has_mtl or gltf
        mat = getattr(vis, "material", None)
        if getattr(vis, "uv", None) is not None and len(vis.uv) == len(V):
            uv = np.asarray(vis.uv, np.float64)
        if mat is not None:
            if hasattr(mat, "baseColorFactor") and mat.baseColorFactor is not None:
                base = tuple(np.asarray(mat.baseColorFactor, float) / 255.0)
                img = getattr(mat, "baseColorTexture", None)
            elif hasattr(mat, "diffuse") and mat.diffuse is not None:
                base = tuple(np.asarray(mat.diffuse, float) / 255.0)
                img = getattr(mat, "image", None)
        if getattr(vis, "kind", None) == "vertex":
            colors = np.asarray(vis.vertex_colors, np.uint8)
            own = True
        elif getattr(vis, "kind", None) == "face" and uv is None:
            fc = np.asarray(vis.face_colors, np.uint8)
            if len(np.unique(fc, axis=0)) == 1:
                base = tuple(fc[0] / 255.0)
            else:  # unindex so every corner carries its face's colour
                V, F, colors = V[F.ravel()], np.arange(F.size).reshape(-1, 3), np.repeat(fc, 3, axis=0)
                N = None
            own = True
        if tex_img is not None:
            img = tex_img
        if img is not None and uv is None:  # texture but no UVs: Blender box projection on object coords
            F0 = F
            V, F, uv = box_uv(V, F)
            colors = colors[F0.ravel()] if colors is not None else None
            N = None
        if uv is None and len(F) > MESH_FACE_LIMIT and colors is None:
            V, F = decimate(V, F, MESH_FACE_TARGET)
            N = None
        if yup:  # glTF Y-up -> station Z-up, as Genesis and Blender do: (x, y, z) -> (x, -z, y)
            R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
            V = V @ R.T
            if N is not None:
                N = N @ R.T
        if N is None:
            CN = corner_normals(V, F)
            extra = {}
            if uv is not None:
                extra["uv"] = uv
            if colors is not None:
                extra["col"] = colors
            V, F, N, ex = split_by_corner_normals(V, F, CN, extra)
            uv, colors = ex.get("uv"), ex.get("col")
        if colors is not None and colors.shape[1] == 3:
            colors = np.concatenate([colors, np.full((len(colors), 1), 255, np.uint8)], 1)
        tex = g.image(img) if img is not None and uv is not None else None
        uvg = None if uv is None else np.stack([uv[:, 0], 1.0 - uv[:, 1]], 1)  # glTF UV origin is top-left
        g.prim(V, F, N, uv=uvg if tex is not None else None, colors=colors, base=base, tex=tex, own=own)
        n_out += len(F)
    if not g.meshes_prims:
        raise ValueError(f"no geometry in {path}")
    os.makedirs(MESHES, exist_ok=True)
    data = g.bytes()
    with open(dst + ".tmp", "wb") as f:
        f.write(data)
    os.replace(dst + ".tmp", dst)
    info = dict(faces_in=int(n_in), faces=int(n_out), bytes=len(data), src=os.path.relpath(path, ROOT))
    write_json_atomic(info_p, info)
    log(f"  mesh {os.path.relpath(path, OUT)} -> {h}.glb ({n_in} -> {n_out} faces, {len(data) / 1e6:.2f} MB)")
    return f"_meshes/{h}.glb", info


# ============================================================================ runs
def find_runs():
    runs = []
    for d, dirs, files in os.walk(OUT):
        dirs[:] = sorted(x for x in dirs if x not in ("meshes", "meshseq", "stills", "frames", "__pycache__")
                         and not x.startswith("."))
        if "meta.json" in files:
            runs.append(d)
    return runs


def run_id_of(d):
    rel = os.path.relpath(d, OUT)
    return rel.replace(os.sep, "__")


def run_signature(d, meta):
    paths = [os.path.join(d, x) for x in ("meta.json", "traj.npz", "checks.json", "DONE")]
    sig = [FORMAT] + [stat_sig(p) for p in paths]
    for name, o in sorted(meta.get("objects", {}).items()):
        if o.get("kind") == "meshseq":
            sd = os.path.join(d, o.get("dir", f"meshseq/{name}"))
            try:
                fs = os.listdir(sd)
                mt = max((os.stat(os.path.join(sd, f)).st_mtime_ns for f in fs), default=0)
                sig.append([name, len(fs), mt])
            except FileNotFoundError:
                sig.append([name, None])
    return hashlib.sha1(json.dumps(sig).encode()).hexdigest()[:16]


def latest_mtime(d):
    m = 0.0
    for x in ("meta.json", "traj.npz", "checks.json", "DONE"):
        p = os.path.join(d, x)
        if os.path.exists(p):
            m = max(m, os.path.getmtime(p))
    return m


def quantize(P):
    """float (..., 3) with possible NaN -> (uint16 array, lo[3], step[3]); NaN -> 65535."""
    fin = np.isfinite(P).all(-1)
    if fin.any():
        lo = np.nanmin(P[fin], axis=0)
        hi = np.nanmax(P[fin], axis=0)
    else:
        lo = hi = np.zeros(3)
    step = np.maximum((hi - lo) / 65534.0, 1e-9)
    Q = np.zeros(P.shape, np.uint16)
    Q[fin] = np.clip(np.round((P[fin] - lo) / step), 0, 65534).astype(np.uint16)
    Q[~fin] = 65535
    return Q, lo.astype(float).tolist(), step.astype(float).tolist()


def seq_frame_bytes(V, F):
    """meshseq frame -> bytes: u32 nV, nF, idx bytes, 0 | f32 lo[3], step[3] | u16 V (padded to 4) | u16/u32 F."""
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    if len(V) == 0 or len(F) == 0:
        return struct.pack("<IIII", 0, 0, 2, 0) + struct.pack("<6f", 0, 0, 0, 1, 1, 1)
    Q, lo, step = quantize(V)
    ib = 2 if len(V) < 65536 else 4
    vb = Q.astype("<u2").tobytes()
    vb += b"\0" * ((4 - len(vb) % 4) % 4)
    fb = F.astype("<u2" if ib == 2 else "<u4").tobytes()
    return struct.pack("<IIII", len(V), len(F), ib, 0) + struct.pack("<6f", *lo, *step) + vb + fb


def convert_run(d, rid, force=False, prev=None):
    meta_p, traj_p = os.path.join(d, "meta.json"), os.path.join(d, "traj.npz")
    now = time.time()
    for p in (meta_p, traj_p):
        if os.path.exists(p) and now - os.path.getmtime(p) < SETTLE_S:
            raise RuntimeError(f"{os.path.relpath(p, OUT)} modified {now - os.path.getmtime(p):.1f}s ago, still writing?")
    meta = json.load(open(meta_p))
    objects = meta.get("objects") or {}
    # Recorder.close() writes traj.npz and then meta.json; meshseq frames are written while recording. Anything newer
    # than meta.json means the station is re-recording into this directory: keep the previous version until it closes.
    meta_t = os.path.getmtime(meta_p)
    newer = [os.path.join(dp, f) for nm, o in objects.items() if o.get("kind") == "meshseq"
             for dp, _, fs in [next(os.walk(os.path.join(d, o.get("dir", f"meshseq/{nm}"))), (None, None, []))]
             for f in sorted(fs)[-3:] if dp]
    newer.append(traj_p)
    for p in newer:
        if os.path.exists(p) and os.path.getmtime(p) > meta_t + 5:
            raise RuntimeError(f"{os.path.relpath(p, OUT)} is newer than meta.json (re-recording in progress?)")
    sig = run_signature(d, meta)
    if prev is not None and prev.get("sig") == sig and not force:
        return prev, False
    n = int(meta.get("n_frames", 0))
    fps = float(meta.get("fps", L.RENDER_FPS))
    traj = {}
    if os.path.exists(traj_p):
        with np.load(traj_p) as z:
            traj = {k: z[k] for k in z.files}     # raises on a truncated archive
    elif any(o.get("kind") in ("rod", "particles") or (o.get("kind") == "rigid" and "static_pose" not in o)
             for o in objects.values()):
        raise RuntimeError("traj.npz missing")
    if "t" in traj and len(traj["t"]) != n:
        raise RuntimeError(f"traj.npz has {len(traj['t'])} frames, meta.json says {n}")
    for name, o in objects.items():
        k = o.get("kind")
        if k == "rigid" and "static_pose" not in o:
            for s in ("pos", "quat"):
                a = traj.get(f"{name}/{s}")
                if a is None or len(a) != n:
                    raise RuntimeError(f"{name}/{s} missing or wrong length")
        elif k in ("rod", "particles"):
            a = traj.get(f"{name}/pts")
            if a is None or len(a) != n:
                raise RuntimeError(f"{name}/pts missing or wrong length")

    # ---- frame stride from the size budget
    rigid_dyn = [nm for nm, o in objects.items() if o.get("kind") == "rigid" and "static_pose" not in o]
    pts_objs = [nm for nm, o in objects.items() if o.get("kind") in ("rod", "particles")]
    per_frame = 28 * len(rigid_dyn)
    part_objs = [nm for nm in pts_objs if objects[nm]["kind"] == "particles"]
    for nm in pts_objs:
        if nm not in part_objs:
            per_frame += 6 * traj[f"{nm}/pts"].shape[1]
    # particles get at most ~45 % of the budget at the stride we would like (every frame up to 900 frames, then
    # every 2nd, ...): a fixed random subset of PARTICLE_LIMIT or fewer, never below 3000 per object
    want_stride = max(1, int(math.ceil(n / 900.0)))
    part_cap = PARTICLE_LIMIT
    if part_objs and n:
        part_cap = int(min(PARTICLE_LIMIT, max(3000, 0.45 * BUDGET / (n / want_stride) / 6 / len(part_objs))))
    sub = {}
    for nm in part_objs:
        k = traj[f"{nm}/pts"].shape[1]
        if k > part_cap:
            sub[nm] = np.sort(np.random.default_rng(0).choice(k, part_cap, replace=False))
            k = part_cap
        per_frame += 6 * k
    seqs = {}
    for nm, o in objects.items():
        if o.get("kind") == "meshseq":
            sd = os.path.join(d, o.get("dir", f"meshseq/{nm}"))
            fs = sorted(f for f in os.listdir(sd) if f.endswith(".npz") and f[:-4].isdigit()) if os.path.isdir(sd) else []
            seqs[nm] = (sd, np.array([int(f[:-4]) for f in fs], int))
    seq_est = 0.0
    for nm, (sd, idx) in seqs.items():
        if len(idx):
            samp = [idx[len(idx) // 2], idx[-1]]
            nf = []
            for j in samp:
                try:
                    with np.load(os.path.join(sd, f"{j:05d}.npz")) as z:
                        nf.append(len(z["F"]))
                except Exception:
                    pass
            fpf = min(max(nf or [0]), SEQ_FACE_LIMIT)
            seq_est += fpf * 9.5 * len(idx) / max(n, 1)     # ~0.5 vertex (6 B) + 3 indices (2-4 B) per face
    total_per_frame = per_frame + seq_est
    stride = max(1, int(math.ceil(total_per_frame * n / BUDGET))) if n else 1
    if seq_est > 0 and stride > want_stride:  # meshseq frames shrink (decimation) before playback gets choppier
        stride = max(want_stride, int(math.ceil((per_frame + min(seq_est, 3000 * 9.5)) * n / BUDGET)))
    frames = np.arange(0, n, stride)
    if n and frames[-1] != n - 1 and stride > 1:
        frames = np.append(frames, n - 1)                     # always keep the last frame
    nf_out = len(frames)

    ver = sig[:10]
    rd = os.path.join(DATA, rid)
    vd = os.path.join(rd, ver)
    if os.path.exists(vd):
        shutil.rmtree(vd)
    os.makedirs(vd)
    man_objs = []
    # ---- rigids
    tb = []
    toff = 0
    mesh_cache = {}
    for nm, o in objects.items():
        k = o.get("kind")
        mo = dict(name=nm, kind=k, material=o.get("material") or {})
        if k == "rigid":
            mp = os.path.join(d, o["mesh"])
            tex = None
            spec_tex = (o.get("material") or {}).get("texture")
            if spec_tex:
                for cand in (spec_tex, os.path.join(d, spec_tex), os.path.join(d, "meshes", os.path.basename(spec_tex))):
                    if os.path.exists(cand):
                        tex = cand
                        break
            key = (mp, tex, o.get("mesh_up"))
            if key not in mesh_cache:
                try:
                    mesh_cache[key] = convert_mesh(mp, tex, o.get("mesh_up"))
                except Exception as e:  # noqa: BLE001
                    log(f"  {rid}: mesh {o['mesh']} failed: {e}")
                    mesh_cache[key] = (None, None)
            url, info = mesh_cache[key]
            mo.update(mesh=url, faces=(info or {}).get("faces"), scale=o.get("scale", 1.0))
            if "static_pose" in o:
                mo["static_pose"] = o["static_pose"]
            else:
                P = traj[f"{nm}/pos"][frames].astype(np.float32)
                Q = traj[f"{nm}/quat"][frames].astype(np.float32)
                Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
                arr = np.concatenate([P, Q], 1).astype("<f4")
                arr[~np.isfinite(arr)] = 0
                tb.append(arr.tobytes())
                mo["traj"] = toff
                toff += arr.size
        elif k in ("rod", "particles"):
            mo["radius"] = float(o.get("radius", 0.003))
        elif k == "meshseq":
            pass
        else:
            mo["unsupported"] = True
        man_objs.append(mo)
    files = {}
    if tb:
        with open(os.path.join(vd, "traj.bin"), "wb") as f:
            f.write(b"".join(tb))
        files["traj"] = f"{ver}/traj.bin"
    # ---- rods + particles
    db = []
    doff = 0
    for mo in man_objs:
        if mo["kind"] in ("rod", "particles"):
            P = traj[f"{mo['name']}/pts"][frames]
            if mo["name"] in sub:
                P = P[:, sub[mo["name"]]]
                mo["subsampled_from"] = int(traj[f"{mo['name']}/pts"].shape[1])
            Q, lo, step = quantize(P.astype(np.float64))
            db.append(Q.astype("<u2").tobytes())
            mo.update(dyn=doff, k=int(P.shape[1]), lo=lo, step=step)
            doff += Q.size
    if db:
        with open(os.path.join(vd, "dyn.bin"), "wb") as f:
            f.write(b"".join(db))
        files["dyn"] = f"{ver}/dyn.bin"
    # ---- meshseq
    seq_bytes = 0
    for mo in man_objs:
        if mo["kind"] != "meshseq":
            continue
        sd, idx = seqs.get(mo["name"], (None, np.array([], int)))
        od = os.path.join(vd, "seq", mo["name"])
        os.makedirs(od, exist_ok=True)
        # source file used at each output frame (latest available <= that frame, as the compositor does)
        need = sorted({int(idx[max(0, np.searchsorted(idx, f, side="right") - 1)]) for f in frames} if len(idx) else set())
        n_seq = max(1, len(need))
        face_cap = int(min(SEQ_FACE_LIMIT, max(3000, (BUDGET - (per_frame * nf_out)) / n_seq / 9.5)))
        done_idx = []
        maxf = 0
        for j in need:
            try:
                with np.load(os.path.join(sd, f"{j:05d}.npz")) as z:
                    V, F = z["V"], z["F"]
            except Exception as e:  # noqa: BLE001
                log(f"  {rid}: meshseq {mo['name']} frame {j} unreadable ({e}), skipped")
                continue
            if len(F) > face_cap:
                V, F = decimate(V, F, face_cap)
            b = seq_frame_bytes(V, F)
            with open(os.path.join(od, f"{j:05d}.bin"), "wb") as f:
                f.write(b)
            seq_bytes += len(b)
            maxf = max(maxf, len(F))
            done_idx.append(j)
        mo.update(seq=f"{ver}/seq/{mo['name']}/", frames=done_idx, max_faces=maxf)

    n_bytes = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(vd) for f in fs)
    mesh_bytes = sum((mesh_cache[k][1] or {}).get("bytes", 0) for k in mesh_cache)
    rel = os.path.relpath(d, OUT)
    parts = rel.split(os.sep)
    station = meta.get("station") or parts[0]
    label = "/".join(parts[1:]) if len(parts) > 1 else "export"
    if station != parts[0] and len(parts) == 1:
        label = parts[0]
    note = meta.get("note") or meta.get("notes") or meta.get("drive") or meta.get("task") or ""
    events = []
    for e in meta.get("events") or []:
        if isinstance(e, dict) and "t" in e and "msg" in e:
            events.append(dict(t=float(e["t"]), msg=str(e["msg"])[:160]))
    checks = checks_summary(os.path.join(d, "checks.json"))
    man = dict(id=rid, sig=sig, ver=ver, station=station, label=label, source=f"out/{rel}", fps=fps,
               n_frames=nf_out, src_frames=n, stride=int(stride), frames_kept=frames.tolist() if stride > 1 else None,
               duration=(n - 1) / fps if n > 1 else 0.0, mtime=latest_mtime(d), synced=time.time(),
               done=os.path.exists(os.path.join(d, "DONE")), checks=checks,
               camera=meta.get("camera") if isinstance(meta.get("camera"), dict) else None,
               table=meta.get("table") if isinstance(meta.get("table"), dict) else None,
               note=str(note)[:600], events=events[:300], files=files, objects=man_objs,
               bytes=dict(frames=n_bytes, meshes=mesh_bytes, seq=seq_bytes))
    write_json_atomic(os.path.join(rd, "manifest.json"), man)
    for x in os.listdir(rd):  # drop older versions of this run
        p = os.path.join(rd, x)
        if os.path.isdir(p) and x != ver:
            shutil.rmtree(p, ignore_errors=True)
    log(f"{rid}: {n} frames (stride {stride} -> {nf_out}), {len(man_objs)} objects, "
        f"frames {n_bytes / 1e6:.1f} MB + meshes {mesh_bytes / 1e6:.1f} MB")
    return man, True


def layout_info():
    return dict(TABLE=L.TABLE, STATIONS={k: dict(world=v["world"], title=v.get("title", k), task=v.get("task", ""),
                                                 arm=v.get("arm")) for k, v in L.STATIONS.items()},
                BACKGROUND_RGB=L.BACKGROUND_RGB, FLOOR_RGB=L.FLOOR_RGB, ROBOT_MOUNT=L.ROBOT_MOUNT,
                WORK_ZONE=L.WORK_ZONE, RENDER_FPS=L.RENDER_FPS)


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="reconvert every run")
    ap.add_argument("--only", default=None, help="only this run id (e.g. cables__tests__drop1)")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()
    VERBOSE = not args.quiet
    os.makedirs(MESHES, exist_ok=True)
    t0 = time.time()
    entries = []
    seen = set()
    changed = 0
    for d in find_runs():
        rid = run_id_of(d)
        seen.add(rid)
        man_p = os.path.join(DATA, rid, "manifest.json")
        prev = None
        if os.path.exists(man_p):
            try:
                prev = json.load(open(man_p))
            except Exception:
                prev = None
        if args.only and rid != args.only:
            man = prev
        else:
            try:
                man, ch = convert_run(d, rid, force=args.force, prev=prev)
                changed += int(ch)
            except Exception as e:  # noqa: BLE001  partially written export: keep the old version, retry next pass
                log(f"{rid}: skipped ({type(e).__name__}: {e})")
                man = prev
        if man:
            entries.append({k: man[k] for k in ("id", "station", "label", "source", "fps", "n_frames", "src_frames",
                                                 "stride", "duration", "mtime", "synced", "done", "checks", "bytes",
                                                 "note")}
                           | dict(n_objects=len(man["objects"]),
                                  kinds=sorted({o["kind"] for o in man["objects"]}),
                                  robot=any(o["name"].startswith("robot/") for o in man["objects"])))
    # runs whose export disappeared
    for x in os.listdir(DATA):
        if x.startswith("_") or x in seen or not os.path.isdir(os.path.join(DATA, x)):
            continue
        shutil.rmtree(os.path.join(DATA, x), ignore_errors=True)
        log(f"{x}: export gone, removed")
    # meshes no run references any more
    used = set()
    for x in os.listdir(DATA):
        mp = os.path.join(DATA, x, "manifest.json")
        if os.path.exists(mp):
            try:
                used |= {o.get("mesh") for o in json.load(open(mp))["objects"]}
            except Exception:
                pass
    for f in os.listdir(MESHES):
        if f.endswith(".glb") and f"_meshes/{f}" not in used and time.time() - os.path.getmtime(os.path.join(MESHES, f)) > 600:
            os.remove(os.path.join(MESHES, f))
            if os.path.exists(os.path.join(MESHES, f + ".json")):
                os.remove(os.path.join(MESHES, f + ".json"))
    entries.sort(key=lambda e: (e["station"], -e["mtime"]))
    index = dict(generated=time.time(), layout=layout_info(), runs=entries)
    write_json_atomic(os.path.join(DATA, "index.json"), index, indent=1)
    log(f"{len(entries)} runs, {changed} converted, {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
