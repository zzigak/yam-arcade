"""Generate the mahjong station assets: tile face textures (PIL), rounded-edge UV-mapped tile meshes (OBJ+MTL+PNG),
a slotted wooden tile rack (union of boxes, wood-grain texture), and URDFs (visual mesh + box collision + inertia).

Run with the station's tool venv (manifold3d for the rack union):
    ~/projects/yam-arcade/.venvs/mahjong/bin/python stations/mahjong/make_assets.py
Outputs go to stations/mahjong/assets/.
"""
import json
import math
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import gaussian_filter, shift as nd_shift

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spec  # noqa: E402

OUT = os.path.join(HERE, "assets")
FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Black.ttc"  # face 0 = JP (has 萬 發 東 ...)

# ---------------------------------------------------------------------------------------------- colours (sRGB 0-1)
IVORY = np.array([0.955, 0.937, 0.880])
BACK = np.array([0.075, 0.430, 0.330])  # jade-green back layer
RED = (0.78, 0.09, 0.11)
GREEN = (0.06, 0.45, 0.24)
BLUE = (0.12, 0.25, 0.56)
INK = (0.07, 0.07, 0.09)

PPM = 24        # texture pixels per mm
SS = 2          # supersampling for drawing
ATLAS = 1024
FRONT_PX = (round(spec.TILE_W * 1000 * PPM), round(spec.TILE_H * 1000 * PPM))  # 624 x 840
SIDE_BOX = (640, 0, 1024, 504)    # x0, y0, x1, y1 (21 mm thickness over 504 px)
BACK_BOX = (640, 520, 1024, 1024)


# =========================================================================================================== faces
class Face:
    """Paint layer for one tile face, drawn in mm with the origin at the face centre, +y down (image convention)."""

    def __init__(self):
        self.w, self.h = FRONT_PX[0] * SS, FRONT_PX[1] * SS
        self.img = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)

    def px(self, x, y):
        return (self.w / 2 + x * PPM * SS, self.h / 2 + y * PPM * SS)

    @staticmethod
    def rgba(c):
        return tuple(int(round(255 * v)) for v in c) + (255,)

    def circle(self, x, y, r, c):
        cx, cy = self.px(x, y)
        rr = r * PPM * SS
        self.d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=self.rgba(c))

    def ring(self, x, y, r0, r1, c):
        """annulus by drawing an ellipse outline of width r1-r0"""
        cx, cy = self.px(x, y)
        rr = r1 * PPM * SS
        self.d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=self.rgba(c), width=int(round((r1 - r0) * PPM * SS)))

    def poly(self, pts, c):
        self.d.polygon([self.px(*p) for p in pts], fill=self.rgba(c))

    def rrect(self, x, y, w, h, rad, c, angle=0.0):
        """rounded rectangle centred at (x, y), rotated by angle (deg, clockwise in image)"""
        if angle == 0.0:
            x0, y0 = self.px(x - w / 2, y - h / 2)
            x1, y1 = self.px(x + w / 2, y + h / 2)
            self.d.rounded_rectangle([x0, y0, x1, y1], radius=rad * PPM * SS, fill=self.rgba(c))
            return
        # rotated: polygon approximation of the rounded rectangle
        n = 8
        pts = []
        for cx, cy, a0 in [(w / 2 - rad, h / 2 - rad, 0), (-w / 2 + rad, h / 2 - rad, 90),
                           (-w / 2 + rad, -h / 2 + rad, 180), (w / 2 - rad, -h / 2 + rad, 270)]:
            for i in range(n + 1):
                a = math.radians(a0 + 90 * i / n)
                pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
        ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        self.poly([(x + ca * px - sa * py, y + sa * px + ca * py) for px, py in pts], c)

    def text(self, ch, x, y, box_w, box_h, c):
        """glyph scaled to fit box_w x box_h mm, centred at (x, y) on its ink bounding box"""
        probe = ImageFont.truetype(FONT, 400, index=0)
        l, t, r, b = probe.getbbox(ch)
        s = min(box_w * PPM * SS / (r - l), box_h * PPM * SS / (b - t))
        font = ImageFont.truetype(FONT, max(8, int(400 * s)), index=0)
        l, t, r, b = font.getbbox(ch)
        cx, cy = self.px(x, y)
        self.d.text((cx - (l + r) / 2, cy - (t + b) / 2), ch, font=font, fill=self.rgba(c))


def dot(f, x, y, R, c, big=False):
    """One 'pin' circle: coloured disc, ivory ring, coloured core with an ivory rosette."""
    f.circle(x, y, R, c)
    f.ring(x, y, 0.70 * R, 0.80 * R, IVORY)
    for i in range(8):
        a = 2 * math.pi * i / 8
        f.circle(x + 0.46 * R * math.cos(a), y + 0.46 * R * math.sin(a), 0.11 * R, IVORY)
    f.circle(x, y, 0.20 * R, IVORY)
    f.circle(x, y, 0.09 * R, RED if c != RED else BLUE)
    if big:
        for i in range(16):
            a = 2 * math.pi * (i + 0.5) / 16
            f.circle(x + 0.90 * R * math.cos(a), y + 0.90 * R * math.sin(a), 0.045 * R, IVORY)


def stick(f, x, y, L, c, angle=0.0, w=2.5):
    """One bamboo stick: body, three nodes, a pale centre line."""
    f.rrect(x, y, w, L, 0.9, c, angle)
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    for t in (-0.5 * L + 0.55, 0.0, 0.5 * L - 0.55):
        nx, ny = x - sa * t, y + ca * t
        f.rrect(nx, ny, w + 0.9, 0.75, 0.3, c, angle)
        f.rrect(nx, ny, w - 0.6, 0.22, 0.1, IVORY, angle)
    for t0, t1 in ((-0.5 * L + 1.2, -0.6), (0.6, 0.5 * L - 1.2)):
        tm, lm = 0.5 * (t0 + t1), (t1 - t0)
        f.rrect(x - sa * tm, y + ca * tm, 0.45, lm, 0.2, IVORY, angle)


def draw_face(kind):
    f = Face()
    n, s = int(kind[0]), kind[1]
    if s == "m":
        f.text("一二三四五六七八九"[n - 1], 0.0, -6.8, 15.0, 11.5, INK)
        f.text("萬", 0.0, 6.6, 17.0, 13.5, RED)
    elif s == "z":
        if n <= 4:
            f.text(spec.HONOR_GLYPHS[n], 0.0, 0.0, 18.5, 24.0, INK)
        elif n == 6:
            f.text("發", 0.0, 0.0, 19.0, 24.5, GREEN)
        elif n == 7:
            f.text("中", 0.0, 0.0, 18.5, 24.5, RED)
        # 5z (White dragon) stays blank, as on Japanese sets
    elif s == "p":
        layouts = {
            1: [(0, 0, 8.6, GREEN)],
            2: [(0, -6.6, 4.7, GREEN), (0, 6.6, 4.7, BLUE)],
            3: [(-5.4, -8.6, 3.9, BLUE), (0, 0, 3.9, RED), (5.4, 8.6, 3.9, GREEN)],
            4: [(-4.8, -6.4, 4.1, BLUE), (4.8, -6.4, 4.1, GREEN), (-4.8, 6.4, 4.1, GREEN), (4.8, 6.4, 4.1, BLUE)],
            5: [(-5.4, -8.6, 3.6, BLUE), (5.4, -8.6, 3.6, GREEN), (0, 0, 3.6, RED), (-5.4, 8.6, 3.6, GREEN),
                (5.4, 8.6, 3.6, BLUE)],
            6: [(-4.7, -9.2, 3.4, GREEN), (4.7, -9.2, 3.4, GREEN)]
               + [(sx * 4.7, sy, 3.4, RED) for sy in (1.4, 9.2) for sx in (-1, 1)],
            7: [(-5.6, -10.6, 3.0, GREEN), (0, -8.1, 3.0, GREEN), (5.6, -5.6, 3.0, GREEN)]
               + [(sx * 4.6, sy, 3.0, RED) for sy in (2.6, 10.2) for sx in (-1, 1)],
            8: [(sx * 4.6, sy, 3.05, BLUE) for sy in (-10.5, -3.5, 3.5, 10.5) for sx in (-1, 1)],
            9: [(sx, sy, 2.95, c) for sy, c in ((-9.6, BLUE), (0, RED), (9.6, GREEN)) for sx in (-6.2, 0, 6.2)],
        }
        for x, y, R, c in layouts[n]:
            dot(f, x, y, R, c, big=(n == 1))
        if n == 1:
            f.ring(0, 0, 4.6, 5.2, RED)
    elif s == "s":
        if n == 1:  # the 1-bamboo bird, stylised
            f.poly([(-1.5, 2), (-8.5, 12.5), (-4, 13.5), (0, 6), (4, 13.5), (8.5, 12.5), (1.5, 2)], BLUE)
            f.poly([(-0.8, 3), (-5.0, 12.0), (0, 8), (5.0, 12.0), (0.8, 3)], GREEN)
            f.circle(0, -1.0, 4.8, GREEN)
            f.circle(0.6, -1.4, 3.0, RED)
            f.circle(0.2, -7.5, 2.6, GREEN)
            f.circle(0.9, -7.8, 0.7, IVORY)
            f.poly([(2.4, -7.6), (5.2, -6.6), (2.4, -6.4)], RED)
            for dx in (-1.6, 0.0, 1.6):
                f.rrect(dx * 0.8, -11.6, 0.6, 2.4, 0.3, RED, angle=dx * 12)
        else:
            L3, L2 = 8.6, 11.0
            layouts = {
                2: [(0, -6.6, L2, BLUE), (0, 6.6, L2, GREEN)],
                3: [(0, -6.6, L2, BLUE), (-4.6, 6.6, L2, GREEN), (4.6, 6.6, L2, GREEN)],
                4: [(-4.6, -6.6, L2, GREEN), (4.6, -6.6, L2, BLUE), (-4.6, 6.6, L2, BLUE), (4.6, 6.6, L2, GREEN)],
                5: [(-5.6, -6.6, L2, GREEN), (5.6, -6.6, L2, BLUE), (0, 0, L2, RED), (-5.6, 6.6, L2, BLUE),
                    (5.6, 6.6, L2, GREEN)],
                6: [(sx, sy, L2, GREEN) for sy in (-6.6, 6.6) for sx in (-5.6, 0, 5.6)],
                7: [(0, -10.0, L3 - 1.2, RED)] + [(sx, sy, L3, GREEN) for sy in (0.6, 10.2) for sx in (-5.6, 0, 5.6)],
                9: [(sx, sy, L3, RED if sx == 0 else GREEN) for sy in (-9.8, 0, 9.8) for sx in (-6.0, 0, 6.0)],
            }
            if n == 8:  # M over W
                for sy, sgn in ((-6.8, 1), (6.8, -1)):
                    stick(f, -7.4, sy, L2, GREEN)
                    stick(f, 7.4, sy, L2, GREEN)
                    stick(f, -2.8, sy, L2, GREEN, angle=sgn * 22)
                    stick(f, 2.8, sy, L2, GREEN, angle=-sgn * 22)
            else:
                for x, y, L, c in layouts[n]:
                    stick(f, x, y, L, c)
    return f.img.resize(FRONT_PX, Image.LANCZOS)


def engrave(paint):
    """Composite a paint layer onto the ivory face as if engraved and inked: the upper-left walls of each groove fall
    into shadow, the lower-right walls catch light. Returns float RGB (h, w, 3)."""
    P = np.asarray(paint, np.float32) / 255.0
    A, C = P[..., 3], P[..., :3]
    h, w = A.shape
    rng = np.random.default_rng(0)
    base = IVORY[None, None, :] * (1.0 + 0.012 * gaussian_filter(rng.standard_normal((h, w)), 18)[..., None]
                                   + 0.006 * rng.standard_normal((h, w))[..., None])
    d = 3.0
    A_ul = nd_shift(A, (d, d), order=1, mode="constant")    # value of the pixel up-left of each pixel
    A_dr = nd_shift(A, (-d, -d), order=1, mode="constant")
    shadow = gaussian_filter(A * np.clip(1 - A_ul, 0, 1), 1.2)
    light = gaussian_filter(A * np.clip(1 - A_dr, 0, 1), 1.2)
    ink = C * (0.93 - 0.40 * shadow[..., None]) + 0.10 * light[..., None]
    # faint rim just outside the groove where the cut catches the light (lower-right) / shades (upper-left)
    rim_dark = gaussian_filter(np.clip(A_dr - A, 0, 1), 1.0)
    out = base * (1 - 0.10 * rim_dark[..., None])
    out = out * (1 - A[..., None]) + ink * A[..., None]
    return np.clip(out, 0, 1)


def make_atlas(kind):
    atlas = np.zeros((ATLAS, ATLAS, 3), np.float32)
    atlas[:] = IVORY
    face = engrave(draw_face(kind))
    atlas[: FRONT_PX[1], : FRONT_PX[0]] = face
    rng = np.random.default_rng(1)
    # side band: ivory front layer, thin seam, coloured back layer
    x0, y0, x1, y1 = SIDE_BOX
    split = y0 + round((y1 - y0) * spec.TILE_FACE_LAYER / spec.TILE_T)
    atlas[y0:split, x0:x1] = IVORY
    atlas[split:y1, x0:x1] = BACK
    atlas[split - 1: split + 1, x0:x1] *= 0.80
    # back face
    x0, y0, x1, y1 = BACK_BOX
    atlas[y0:y1, x0:x1] = BACK
    noise = 1.0 + 0.02 * gaussian_filter(rng.standard_normal((ATLAS, ATLAS)), 10)[..., None]
    region = np.zeros((ATLAS, ATLAS, 1), bool)
    region[:, 640:] = True
    atlas = np.where(region, atlas * noise, atlas)
    return Image.fromarray((np.clip(atlas, 0, 1) * 255 + 0.5).astype(np.uint8))


# ============================================================================================================ meshes
def rounded_box(half, r, n_round=4, n_int=3):
    """Rounded box as 6 separately UV'd grids. Returns list of (face_id, V (n,3), N (n,3), grid params (n,2), tris)."""
    def axis_coords(h):
        th = np.linspace(0, math.pi / 4, n_round + 1)[1:]
        edge = (h - r) + r * np.tan(th)
        inner = np.linspace(-(h - r), h - r, n_int + 1)
        return np.concatenate([-edge[::-1], inner, edge])

    E = np.eye(3)
    faces = {"+z": (E[0], E[1], E[2]), "-z": (E[1], E[0], -E[2]), "+x": (E[1], E[2], E[0]),
             "-x": (E[2], E[1], -E[0]), "+y": (E[2], E[0], E[1]), "-y": (E[0], E[2], -E[1])}
    hv = np.asarray(half)
    out = []
    for fid, (a1, a2, nrm) in faces.items():
        i1, i2, i3 = int(np.argmax(np.abs(a1))), int(np.argmax(np.abs(a2))), int(np.argmax(np.abs(nrm)))
        c1, c2 = axis_coords(hv[i1]), axis_coords(hv[i2])
        S1, S2 = np.meshgrid(c1, c2, indexing="ij")
        P = (S1[..., None] * a1 + S2[..., None] * a2 + hv[i3] * nrm).reshape(-1, 3)
        Q = np.clip(P, -(hv - r), hv - r)
        D = P - Q
        N = D / np.linalg.norm(D, axis=1, keepdims=True)
        V = Q + r * N
        n1, n2 = len(c1), len(c2)
        idx = np.arange(n1 * n2).reshape(n1, n2)
        a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
        tris = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
        out.append((fid, V, N, P, tris))
    return out


def tile_uv(fid, P):
    """UV (OBJ convention, v up) of the cube-surface points P (before rounding) of face fid."""
    W, H, T = spec.TILE_W, spec.TILE_H, spec.TILE_T
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    if fid == "+z":
        px, py = (x / W + 0.5) * FRONT_PX[0], (0.5 - y / H) * FRONT_PX[1]
    elif fid == "-z":
        x0, y0, x1, y1 = BACK_BOX
        px, py = x0 + (x / W + 0.5) * (x1 - x0), y0 + (0.5 - y / H) * (y1 - y0)
    else:
        x0, y0, x1, y1 = SIDE_BOX
        along = (y / H + 0.5) if fid in ("+x", "-x") else (x / W + 0.5)
        px, py = x0 + along * (x1 - x0), y0 + (0.5 - z / T) * (y1 - y0)
    return np.stack([px / ATLAS, 1.0 - py / ATLAS], 1)


def write_obj(path, parts, mtl_name, tex_name, color=(1, 1, 1)):
    """parts: list of (V, N, UV, tris). Writes OBJ + MTL next to each other."""
    base = os.path.splitext(path)[0]
    with open(base + ".mtl", "w") as f:
        f.write(f"newmtl {mtl_name}\nKa 0 0 0\nKd {color[0]} {color[1]} {color[2]}\nKs 0.05 0.05 0.05\nNs 40\nd 1\n"
                f"illum 2\nmap_Kd {tex_name}\n")
    lines = [f"mtllib {os.path.basename(base)}.mtl", f"o {mtl_name}"]
    off = 0
    flines = []
    for V, N, UV, tris in parts:
        lines += [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in V]
        lines += [f"vt {a:.6f} {b:.6f}" for a, b in UV]
        lines += [f"vn {a:.5f} {b:.5f} {c:.5f}" for a, b, c in N]
        for t in tris + off + 1:
            flines.append("f " + " ".join(f"{i}/{i}/{i}" for i in t))
        off += len(V)
    lines += [f"usemtl {mtl_name}", "s 1"] + flines
    open(path, "w").write("\n".join(lines) + "\n")


def box_inertia(m, sx, sy, sz):
    return m / 12 * (sy * sy + sz * sz), m / 12 * (sx * sx + sz * sz), m / 12 * (sx * sx + sy * sy)


def make_tiles():
    d = os.path.join(OUT, "tiles")
    os.makedirs(d, exist_ok=True)
    half = (spec.TILE_W / 2, spec.TILE_H / 2, spec.TILE_T / 2)
    grids = rounded_box(half, spec.TILE_EDGE_R)
    parts = [(V, N, tile_uv(fid, P), tris) for fid, V, N, P, tris in grids]
    m = spec.TILE_MASS
    ixx, iyy, izz = box_inertia(m, spec.TILE_W, spec.TILE_H, spec.TILE_T)
    for kind in spec.ALL_KINDS:
        make_atlas(kind).save(os.path.join(d, f"tile_{kind}.png"))
        write_obj(os.path.join(d, f"tile_{kind}.obj"), parts, f"tile_{kind}", f"tile_{kind}.png")
        urdf = f"""<?xml version="1.0"?>
<robot name="tile_{kind}">
  <link name="tile">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{m:.6f}"/>
      <inertia ixx="{ixx:.4e}" iyy="{iyy:.4e}" izz="{izz:.4e}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="tile_{kind}.obj"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="{spec.TILE_W} {spec.TILE_H} {spec.TILE_T}"/></geometry>
    </collision>
  </link>
</robot>
"""
        open(os.path.join(d, f"tile_{kind}.urdf"), "w").write(urdf)
    # a contact sheet of all faces for checking
    faces = [Image.open(os.path.join(d, f"tile_{k}.png")).crop((0, 0) + FRONT_PX) for k in spec.ALL_KINDS]
    sheet = Image.new("RGB", (9 * (FRONT_PX[0] + 40) + 40, 4 * (FRONT_PX[1] + 40) + 40), (200, 200, 205))
    for i, im in enumerate(faces):
        r, c = divmod(i, 9)
        sheet.paste(im, (40 + c * (FRONT_PX[0] + 40), 40 + r * (FRONT_PX[1] + 40)))
    sheet.resize((sheet.width // 3, sheet.height // 3), Image.LANCZOS).save(os.path.join(d, "contact_sheet.png"))
    return m


# ------------------------------------------------------------------------------------------------------------- rack
def wood_texture(path, size=1024):
    """Tileable-ish procedural wood grain: grain lines run along image u (the rack's length)."""
    rng = np.random.default_rng(3)
    v = np.linspace(0, 1, size, endpoint=False)[:, None]
    u = np.linspace(0, 1, size, endpoint=False)[None, :]
    warp = gaussian_filter(rng.standard_normal((size, size)), (40, 120), mode="wrap") * 60
    rings = np.sin(2 * math.pi * (v * 34 + warp) + 0.8 * np.sin(2 * math.pi * u * 2))
    fine = gaussian_filter(rng.standard_normal((size, size)), (0.6, 30), mode="wrap")
    fine /= fine.std()
    t = 0.5 + 0.5 * rings
    light = np.array([0.66, 0.45, 0.28])
    dark = np.array([0.47, 0.29, 0.16])
    col = light * (1 - t[..., None] ** 3 * 0.75) + dark * (t[..., None] ** 3 * 0.75)
    col *= (1 + 0.035 * fine[..., None])
    Image.fromarray((np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def make_rack():
    import manifold3d as mf
    d = os.path.join(OUT, "rack")
    os.makedirs(d, exist_ok=True)
    boxes = spec.rack_boxes()
    solid = None
    for name, c, s, rx in boxes:
        T = np.eye(4)
        ca, sa = math.cos(rx), math.sin(rx)
        T[:3, :3] = [[1, 0, 0], [0, ca, -sa], [0, sa, ca]]
        T[:3, 3] = c
        b = mf.Manifold.cube(s, center=True).transform(T[:3, :])
        solid = b if solid is None else solid + b
    mesh = solid.to_mesh()
    V = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64)
    F = np.asarray(mesh.tri_verts).astype(np.int64)
    tm = trimesh.Trimesh(V, F, process=True)
    assert tm.is_watertight, "rack union is not watertight"
    tm.density = spec.RACK_DENSITY
    mass, com, I = tm.mass, tm.center_mass, tm.moment_inertia
    # triplanar UVs per triangle (unshared vertices): grain along x
    tri = tm.vertices[tm.faces]                       # (n, 3, 3)
    fn = tm.face_normals
    ax = np.argmax(np.abs(fn), axis=1)
    uv = np.zeros(tri.shape[:2] + (2,))
    su, sv = 0.70, 0.18                              # texture span in metres (u along x, v across)
    for k in range(3):
        sel = ax == k
        if k == 0:   # end faces: end grain
            uv[sel, :, 0], uv[sel, :, 1] = tri[sel, :, 1] / sv * 0.25, tri[sel, :, 2] / sv
        elif k == 1:
            uv[sel, :, 0], uv[sel, :, 1] = tri[sel, :, 0] / su, tri[sel, :, 2] / sv
        else:
            uv[sel, :, 0], uv[sel, :, 1] = tri[sel, :, 0] / su, tri[sel, :, 1] / sv
    nV = tri.reshape(-1, 3)
    nN = np.repeat(fn, 3, axis=0)
    nUV = uv.reshape(-1, 2) + 0.5
    nT = np.arange(len(nV)).reshape(-1, 3)
    wood_texture(os.path.join(d, "rack_wood.png"))
    write_obj(os.path.join(d, "rack.obj"), [(nV, nN, nUV, nT)], "rack_wood", "rack_wood.png")
    col = "".join(
        f"""    <collision>
      <origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rpy="{rx:.6f} 0 0"/>
      <geometry><box size="{s[0]:.6f} {s[1]:.6f} {s[2]:.6f}"/></geometry>
    </collision>
""" for name, c, s, rx in boxes)
    urdf = f"""<?xml version="1.0"?>
<robot name="tile_rack">
  <link name="rack">
    <inertial>
      <origin xyz="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" rpy="0 0 0"/>
      <mass value="{mass:.6f}"/>
      <inertia ixx="{I[0, 0]:.5e}" iyy="{I[1, 1]:.5e}" izz="{I[2, 2]:.5e}" ixy="{I[0, 1]:.5e}" ixz="{I[0, 2]:.5e}" iyz="{I[1, 2]:.5e}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="rack.obj"/></geometry>
    </visual>
{col}  </link>
</robot>
"""
    open(os.path.join(d, "rack.urdf"), "w").write(urdf)
    info = dict(mass=mass, com=com.tolist(), volume=tm.volume, bounds=tm.bounds.tolist(), n_boxes=len(boxes))
    json.dump(info, open(os.path.join(d, "rack_info.json"), "w"), indent=1)
    return info


def profile_figure():
    """Side profile of the rack at a slot centre (filled boxes) with a seated tile, for checking by eye."""
    s = 8000  # px per m
    W, H = 900, 520
    img = Image.new("RGBA", (W, H), (245, 245, 247, 255))
    ox, oz = W / 2, H - 60

    def P(y, z):
        return (ox + y * s, oz - z * s)

    def quad(c, sy, sz, rx):
        ca, sa = math.cos(rx), math.sin(rx)
        return [P(c[1] + ca * dy * sy / 2 - sa * dz * sz / 2, c[2] + sa * dy * sy / 2 + ca * dz * sz / 2)
                for dy, dz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]

    cols = dict(shelf=(170, 110, 60), backboard=(120, 80, 40), riser=(150, 130, 90),
                plinth=(110, 90, 70), div07=(230, 170, 60))
    for name, c, sz, rx in spec.rack_boxes():
        if name not in cols:
            continue
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(layer).polygon(quad(c, sz[1], sz[2], rx), fill=cols[name] + (110,), outline=cols[name] + (255,))
        img = Image.alpha_composite(img, layer)
    dr = ImageDraw.Draw(img)
    dr.line([P(-0.06, 0), P(0.06, 0)], fill=(0, 0, 0), width=2)
    (x, y, z), q = spec.seated_tile_pose_rack(7)
    th = 2 * math.atan2(q[1], q[0])
    dr.polygon(quad((x, y, z), spec.TILE_H, spec.TILE_T, th), outline=(20, 110, 80), width=3)
    dr.text((10, 10), "rack profile at slot 7 (y right = away from robot, z up). shelf, backboard, "
                      "divider(yellow), riser, plinth; seated tile green", fill=(0, 0, 0))
    img.convert("RGB").save(os.path.join(OUT, "rack", "profile.png"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    make_tiles()
    info = make_rack()
    profile_figure()
    print(json.dumps(info, indent=1))
