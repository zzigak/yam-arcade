"""Synthetic export for testing the compositor: out/_test/ (station-local frame, table top at z = 0.75).

    ~/simquarry/genesis-world/.venv/bin/python render/make_test_export.py

Contents: a two-tier shelf with one sphere per material kind on top and real-shaped props below (thin-wall glass
jar + brushed lid, ceramic bowl, glass cup with a sloshing water meshseq inside, an OBJ with an .mtl texture, a box
with a `texture` image, the same arrow as OBJ (Z-up) and GLB (Y-up, glTF convention) to check axis handling); in
front a sliding/tumbling cube, a swaying rod, falling beads and water drops, and a wobbling water blob.
Nothing here is physical; it only exercises every export kind and material.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from export import Recorder  # noqa: E402
import layout as L  # noqa: E402

OUT = os.path.join(ROOT, "out", "_test")
SRC = os.path.join(OUT, "_src")
TOP = L.TABLE_TOP_Z
FPS = L.RENDER_FPS
N = 90


# ---------------------------------------------------------------- tiny mesh kit (numpy only)
def box(sx, sy, sz, c=(0, 0, 0)):
    """Box with per-face UVs (24 verts) so textures map one image per face."""
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    faces = [((1, 0, 0), (0, 1, 0)), ((-1, 0, 0), (0, -1, 0)), ((0, 1, 0), (-1, 0, 0)),
             ((0, -1, 0), (1, 0, 0)), ((0, 0, 1), (1, 0, 0)), ((0, 0, -1), (1, 0, 0))]
    V, F, UV = [], [], []
    for n, u in faces:
        n, u = np.array(n, float), np.array(u, float)
        v = np.cross(n, u)
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = n + a * u + b * v
            V.append(p * [hx, hy, hz] + c)
            UV.append(((a + 1) / 2, (b + 1) / 2))
        k = len(V) - 4
        F += [(k, k + 1, k + 2), (k, k + 2, k + 3)]
    return np.array(V), np.array(F), np.array(UV)


def sphere(r, nu=48, nv=24, c=(0, 0, 0), scale=(1, 1, 1)):
    V = [(0, 0, -1)]
    for j in range(1, nv):
        th = np.pi * j / nv - np.pi / 2
        for i in range(nu):
            ph = 2 * np.pi * i / nu
            V.append((np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)))
    V.append((0, 0, 1))
    V = np.array(V) * r * np.array(scale) + c
    F = []
    for i in range(nu):
        F.append((0, 1 + (i + 1) % nu, 1 + i))
    for j in range(nv - 2):
        a, b = 1 + j * nu, 1 + (j + 1) * nu
        for i in range(nu):
            i2 = (i + 1) % nu
            F += [(a + i, a + i2, b + i2), (a + i, b + i2, b + i)]
    top = len(V) - 1
    a = 1 + (nv - 2) * nu
    for i in range(nu):
        F.append((a + i, a + (i + 1) % nu, top))
    return V, np.array(F)


def lathe(profile, n=64):
    """Revolve an (r, z) polyline about +z. r == 0 points become a single pole vertex. Profile order sets normals:
    walk it counter-clockwise in the (r, z) half-plane seen from +y... in practice: outside going up = outward."""
    V, rings = [], []
    for r, z in profile:
        if r < 1e-9:
            rings.append([len(V)] * n)
            V.append((0, 0, z))
        else:
            rings.append(list(range(len(V), len(V) + n)))
            V += [(r * np.cos(2 * np.pi * i / n), r * np.sin(2 * np.pi * i / n), z) for i in range(n)]
    F = []
    for a, b in zip(rings[:-1], rings[1:]):
        for i in range(n):
            i2 = (i + 1) % n
            tri = [(a[i], a[i2], b[i2]), (a[i], b[i2], b[i])]
            F += [t for t in tri if len(set(t)) == 3]
    return np.array(V, float), np.array(F)


def torus(R, r, nu=64, nv=24):
    V, F = [], []
    for i in range(nu):
        u = 2 * np.pi * i / nu
        for j in range(nv):
            v = 2 * np.pi * j / nv
            V.append(((R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)))
    for i in range(nu):
        for j in range(nv):
            a, b = i * nv + j, ((i + 1) % nu) * nv + j
            a2, b2 = i * nv + (j + 1) % nv, ((i + 1) % nu) * nv + (j + 1) % nv
            F += [(a, b, b2), (a, b2, a2)]
    return np.array(V), np.array(F)


def merge(*parts):
    V, F, k = [], [], 0
    for v, f in parts:
        V.append(v)
        F.append(np.asarray(f) + k)
        k += len(v)
    return np.vstack(V), np.vstack(F)


def write_obj(path, V, F, UV=None, mtl=None):
    with open(path, "w") as fh:
        if mtl:
            fh.write(f"mtllib {os.path.basename(mtl[0])}\nusemtl {mtl[1]}\n")
        for v in V:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if UV is not None:
            for t in UV:
                fh.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
            for f in F:
                fh.write("f " + " ".join(f"{i + 1}/{i + 1}" for i in f) + "\n")
        else:
            for f in F:
                fh.write("f " + " ".join(str(i + 1) for i in f) + "\n")
    return path


def checker(path, n=8, px=512, c0=(240, 238, 230), c1=(40, 70, 140), mark=None):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (px, px), c0)
    d = ImageDraw.Draw(im)
    s = px // n
    for i in range(n):
        for j in range(n):
            if (i + j) % 2:
                d.rectangle([i * s, j * s, (i + 1) * s - 1, (j + 1) * s - 1], fill=c1)
    if mark:  # an asymmetric mark so flipped UVs show
        d.rectangle([0, 0, s * 2, s], fill=mark)
    im.save(path)
    return path


def quat_axis(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[np.cos(ang / 2), np.sin(ang / 2) * axis]


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


# ---------------------------------------------------------------- scene
def main():
    os.makedirs(SRC, exist_ok=True)
    for sub in ("meshes", "meshseq"):
        d = os.path.join(OUT, sub)
        if os.path.isdir(d):
            import shutil
            shutil.rmtree(d)
    rec = Recorder(OUT, station="_test", fps=FPS)
    I = (1.0, 0.0, 0.0, 0.0)

    # shelf: two boards on two sides, standing on the table at the back
    sx, sy, t, h = 1.10, 0.16, 0.018, 0.20
    yb = 0.24
    shelf = merge(box(sx, sy, t, (0, 0, t / 2))[:2], box(sx, sy, t, (0, 0, h))[:2],
                  box(t, sy, h, (-sx / 2 + t / 2, 0, h / 2))[:2], box(t, sy, h, (sx / 2 - t / 2, 0, h / 2))[:2])
    rec.add_rigid("shelf", write_obj(os.path.join(SRC, "shelf.obj"), *shelf),
                  material=dict(kind="wood", color=(0.66, 0.50, 0.36)), static_pose=((0, yb, TOP), I))

    # one 3 cm sphere per material kind on the upper board
    kinds = [("paint", (0.80, 0.16, 0.12)), ("plastic", (0.98, 0.78, 0.10)), ("rubber", (0.12, 0.12, 0.13)),
             ("ceramic", (0.93, 0.92, 0.88)), ("glass", None), ("metal", None), ("water", None),
             ("fabric", (0.30, 0.42, 0.62)), ("paper", (0.96, 0.95, 0.91)), ("wood", (0.62, 0.44, 0.28)),
             ("emissive", (1.0, 0.55, 0.15))]
    sph = write_obj(os.path.join(SRC, "sphere.obj"), *sphere(0.03))
    for i, (k, c) in enumerate(kinds):
        x = -0.5 + i * 0.1
        mat = dict(kind=k) if c is None else dict(kind=k, color=c)
        if k == "emissive":
            mat["strength"] = 6.0
        rec.add_rigid(f"sample_{k}", sph, material=mat, static_pose=((x, yb, TOP + h + t / 2 + 0.03), I))

    # lower tier: glass jar with a brushed lid, a ceramic bowl, textured boxes, the axis arrows
    R, H, w = 0.045, 0.12, 0.003
    jar = lathe([(0, 0), (R - 0.006, 0), (R, 0.006), (R, H - 0.015), (R - 0.008, H - 0.008), (R - 0.008, H),
                 (R - 0.008 - w, H), (R - 0.008 - w, H - 0.008 - w), (R - w, H - 0.015), (R - w, 0.006 + w), (0, w)])
    lid = lathe([(0, 0.016), (R - 0.004, 0.016), (R - 0.001, 0.013), (R - 0.001, 0.0), (R - 0.004, 0.0),
                 (R - 0.004, 0.012), (0, 0.012)][::-1])
    z0 = TOP + t
    rec.add_rigid("jar", write_obj(os.path.join(SRC, "jar.obj"), *jar), material=dict(kind="glass", thin=False),
                  static_pose=((-0.42, yb, z0), I))
    rec.add_rigid("jar_lid", write_obj(os.path.join(SRC, "lid.obj"), *lid),
                  material=dict(kind="metal", color=(0.80, 0.80, 0.82), roughness=0.3),
                  static_pose=((-0.42, yb, z0 + H - 0.006), I))
    bowl = lathe([(0, 0), (0.03, 0), (0.06, 0.05), (0.064, 0.055), (0.058, 0.055), (0.027, 0.006), (0, 0.006)])
    rec.add_rigid("bowl", write_obj(os.path.join(SRC, "bowl.obj"), *bowl),
                  material=dict(kind="ceramic", color=(0.55, 0.70, 0.78)), static_pose=((-0.26, yb, z0), I))

    tex = checker(os.path.join(SRC, "checker.png"), mark=(220, 60, 40))
    bV, bF, bUV = box(0.08, 0.08, 0.08)
    rec.add_rigid("tex_box", write_obj(os.path.join(SRC, "tex_box.obj"), bV, bF, bUV),
                  material=dict(kind="paper", texture=os.path.join("meshes", "checker.png")),
                  static_pose=((-0.12, yb, z0 + 0.04), quat_axis((0, 0, 1), 0.4)))
    # OBJ whose .mtl carries its own map_Kd (embedded texture path)
    checker(os.path.join(SRC, "label.png"), n=4, c0=(250, 250, 245), c1=(30, 140, 90))
    with open(os.path.join(SRC, "label_box.mtl"), "w") as fh:
        fh.write("newmtl label\nKd 1 1 1\nmap_Kd label.png\n")
    rec.add_rigid("mtl_box", write_obj(os.path.join(SRC, "label_box.obj"), bV * [1, 0.6, 1.2], bF, bUV,
                                       mtl=("label_box.mtl", "label")),
                  material=dict(kind="plastic"), static_pose=((0.0, yb, z0 + 0.048), quat_axis((0, 0, 1), -0.3)))
    # material["texture"] is not copied by the Recorder, so put it beside the meshes ourselves
    import shutil
    shutil.copy2(tex, os.path.join(OUT, "meshes", "checker.png"))

    # arrow pointing +x with a fin on +z: once as OBJ (Z-up, raw) and once as GLB (glTF Y-up convention)
    arrow = merge(box(0.10, 0.012, 0.012, (0.05, 0, 0))[:2],
                  (np.array([(0.10, -0.02, -0.02), (0.10, 0.02, -0.02), (0.10, 0.02, 0.02), (0.10, -0.02, 0.02),
                             (0.14, 0, 0)]), np.array([(0, 2, 1), (0, 3, 2), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)])),
                  box(0.03, 0.004, 0.03, (0.02, 0, 0.02))[:2])
    rec.add_rigid("arrow_obj", write_obj(os.path.join(SRC, "arrow.obj"), *arrow),
                  material=dict(kind="paint", color=(0.15, 0.45, 0.85)), static_pose=((0.12, yb - 0.03, z0 + 0.02), I))
    import trimesh
    Vy = arrow[0][:, [0, 2, 1]] * [1, 1, -1]  # Z-up -> Y-up: (x, y, z) -> (x, z, -y)
    trimesh.Trimesh(Vy, arrow[1], process=False).export(os.path.join(SRC, "arrow.glb"))
    rec.add_rigid("arrow_glb", os.path.join(SRC, "arrow.glb"),
                  material=dict(kind="paint", color=(0.95, 0.45, 0.10)), static_pose=((0.12, yb + 0.04, z0 + 0.02), I))

    # rubber ring and a brushed metal cylinder on the right of the lower tier
    rec.add_rigid("ring", write_obj(os.path.join(SRC, "ring.obj"), *torus(0.035, 0.01)),
                  material=dict(kind="rubber", color=(0.85, 0.25, 0.20)), static_pose=((0.32, yb, z0 + 0.01), I))
    cyl = lathe([(0, 0), (0.03, 0), (0.03, 0.1), (0, 0.1)])
    rec.add_rigid("can", write_obj(os.path.join(SRC, "can.obj"), *cyl), material=dict(kind="metal"),
                  static_pose=((0.45, yb, z0), I))

    # front: glass cup (fixed) with sloshing water meshseq and falling drops; moving cube; swaying rod; beads; blob
    cR, cH, cw = 0.04, 0.10, 0.0025
    cup = lathe([(0, 0), (cR, 0), (cR, cH), (cR - cw, cH), (cR - cw, 0.006), (0, 0.006)])
    cup_xy = (0.30, -0.12)
    rec.add_rigid("cup", write_obj(os.path.join(SRC, "cup.obj"), *cup), material=dict(kind="glass", thin=False),
                  static_pose=((*cup_xy, TOP), I))
    rec.add_meshseq("cup_water", material=dict(kind="water"))
    rec.add_meshseq("blob", material=dict(kind="water"))
    cube = write_obj(os.path.join(SRC, "cube.obj"), *box(0.05, 0.05, 0.05)[:2])
    rec.add_rigid("cube", cube, material=dict(kind="paint", color=(0.85, 0.18, 0.15), roughness=0.4))
    rec.add_rigid("robot/fake_link", cube, material=dict(kind="plastic", color=(0.2, 0.2, 0.22)), scale=0.6)
    rec.add_rod("rod", radius=0.005, material=dict(kind="rubber", color=(0.10, 0.35, 0.80)))
    rec.add_rod("rod2", radius=0.0035, material=dict(kind="rubber", color=(0.90, 0.30, 0.10)))
    rec.add_particles("beads", radius=0.004, material=dict(kind="plastic", color=(0.95, 0.75, 0.10)))
    rec.add_particles("drops", radius=0.003, material=dict(kind="water"))

    rng = np.random.default_rng(0)
    nb, nd = 300, 200
    b0 = np.c_[rng.uniform(-0.05, 0.05, nb), rng.uniform(-0.05, 0.05, nb), rng.uniform(0.15, 0.45, nb)]
    d0 = np.c_[rng.normal(0, 0.008, nd), rng.normal(0, 0.008, nd), rng.uniform(0.15, 0.5, nd)]
    cyl_lathe_n = 48
    for i in range(N):
        tt = i / FPS
        # cube slides along x and tumbles
        x = -0.40 + 0.5 * (1 - np.cos(np.pi * i / (N - 1))) * 0.35
        q = qmul(quat_axis((0, 0, 1), 1.2 * tt), quat_axis((1, 0, 0), 0.0))
        cube_pose = ((x, -0.18, TOP + 0.025 + 0.03 * max(0, np.sin(4 * tt)) ), q)
        link_pose = ((x, -0.18, TOP + 0.10), quat_axis((1, 1, 0), 2 * tt))
        # rods: base fixed on the table, swaying with a quadratic shape
        s = np.linspace(0, 1, 40)
        A = 0.08 * np.sin(2 * np.pi * 0.8 * tt)
        rod = np.c_[-0.05 + A * s ** 2, -0.10 + 0.02 * A * s ** 3, TOP + 0.005 + 0.30 * s - 0.5 * abs(A) * s ** 3]
        th = np.linspace(0, 3 * np.pi, 80)
        rod2 = np.c_[-0.25 + 0.06 * np.cos(th + tt), -0.05 + 0.06 * np.sin(th + tt) * 0.6, TOP + 0.004 + 0.0 * th]
        rod2[:, 2] += 0.03 * np.maximum(0, np.sin(th / 3 + 2 * tt)) ** 2
        # beads fall onto the table at (0.05, -0.12) and stop
        b = b0.copy()
        b[:, 2] = np.maximum(b0[:, 2] - 0.5 * 9.81 * tt ** 2, 0.004)
        beads = b + [0.08, -0.05, TOP]
        # drops fall into the cup; once below the water line they park at the bottom (hidden in the water)
        d = d0.copy()
        d[:, 2] = d0[:, 2] - 0.5 * 9.81 * tt ** 2
        drops = d + [*cup_xy, TOP]
        drops[:, 2] = np.maximum(drops[:, 2], TOP + 0.02)
        # water in the cup: closed cylinder with a sloshing top
        wr = cR - cw - 0.0005
        Vw, Fw = lathe([(0, 0.0065), (wr, 0.0065), (wr, 0.06), (0, 0.06)], n=cyl_lathe_n)
        top = Vw[:, 2] > 0.05
        Vw[top, 2] = 0.06 + 0.006 * np.sin(3 * tt) * Vw[top, 0] / wr
        Vw = Vw + [*cup_xy, TOP]
        # water blob: wobbling flattened sphere
        Vb, Fb = sphere(0.035, 40, 20, scale=(1.0, 1.0, 0.45))
        Vb[:, 2] *= 1 + 0.15 * np.sin(5 * tt)
        Vb[:, 0] *= 1 - 0.07 * np.sin(5 * tt)
        Vb[:, 2] -= Vb[:, 2].min()
        Vb = Vb + [-0.14, -0.27, TOP]
        rec.frame(tt, rigid={"cube": cube_pose, "robot/fake_link": link_pose}, rods={"rod": rod, "rod2": rod2},
                  particles={"beads": beads, "drops": drops}, meshes={"cup_water": (Vw, Fw), "blob": (Vb, Fb)})
    rec.close(extra_meta=dict(note="synthetic compositor test, not a simulation",
                              camera=dict(eye=[0.55, -1.05, 1.25], target=[0.0, 0.05, 0.86], lens=40)))
    # sample checks.json / NOTES.md so the site builder has something to render for the test section
    import json
    json.dump([dict(name="frames recorded", value=N, limit=N, cmp="=", unit="", pass_=True),
               dict(name="cube top below 1 m", value=float(TOP + 0.08), limit=1.0, cmp="≤", unit="m", passed=True),
               dict(name="deliberately failing row", value=0.5, limit=0.1, cmp="≤", unit="mm", passed=False,
                    note="shows how a failed check renders")], open(os.path.join(OUT, "checks.json"), "w"), indent=1)
    open(os.path.join(OUT, "NOTES.md"), "w").write(
        "# _test notes\n\n## Summary\n\nSynthetic export for the compositor. **Nothing is simulated.** Every "
        "material kind sits on the shelf; `cube`, `rod`, `beads`, `drops`, `cup_water` and `blob` move.\n\n"
        "## Numbers\n\n| thing | value | why |\n|---|---|---|\n| sphere radius | 3 cm | readable at 720p |\n"
        "| rod radius | 5 mm | a thick cable |\n\n- list item one\n- list item two with `code`\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
