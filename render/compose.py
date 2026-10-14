"""YAM arcade compositor: builds the studio, the four tables and every station export, renders with Cycles.

Runs inside Blender 5.2 (snap). Everything after `--` is ours:

    blender -b -P render/compose.py -- --camera overview --frames 0:300 --res 1280x720 --samples 48 \\
        --out render/frames/overview --encode site/media/overview.mp4

    --camera   overview | orbit | <station> | <station>:over      (default overview)
               <station> is a close-up on that table from the far side, facing the robot;
               <station>:over looks over the robot's shoulder. A station's meta.json may carry
               camera=dict(eye, target, lens) in its local frame, which replaces the close-up preset.
    --frames   a:b[:step] export frame indices, b exclusive (default: all frames of the longest export)
    --still    f1,f2,...  render just these frames (no video)
    --res WxH  --samples N  --out DIR  --encode out.mp4  --fps 30
    --stations mahjong,jars,...   which stations to load (default: every one with out/<station>/meta.json)
    --extra NAME@x,y,yaw          also load out/NAME/ at that world pose with its own table (repeatable;
                                  the compositor test uses `--extra _test@0,0,0`)
    --robot-preview               park a cart at its home pose at stations whose export has no robot/* links
                                  (frame 0 of robots/scratch/export_<arm>, else bare URDF arms; framing only)
    --orbit DEG                   overview sweep in degrees over the frame range (camera `orbit`: 20 by default)
    --eye=x,y,z --target=x,y,z --lens mm   override the camera (world frame for overview/orbit, station-local
                                  for station cameras; use `=` when a value starts with a minus sign)
    --fstop F (close-up depth of field, 0 = off)  --exposure EV  --light K  --bg K  --world K
    --save-blend PATH  --device GPU|CPU  --no-render (with --encode: re-encode existing frames)

Frames are written as DIR/NNNNN.png (export index) plus DIR/render_stats.json (per-frame seconds, settings),
which build_site.py reads. No GPU frame starts while ~/projects/yam-arcade/locks/SAM3D_RUNNING exists, while any
GPU process is running SAM 3D (other projects run it without our lock), or while the card has < 6 GB free.

Material dict (export `material`): kind in wood, paint, plastic, rubber, ceramic, glass, metal, water, fabric,
paper, emissive; color (sRGB 0-1); roughness; texture (image path: absolute, relative to out/<station>/, or a file
in out/<station>/meshes/ -- the Recorder does not copy it). Per kind: glass `thin` (default false = refracting
glass for meshes that model both wall surfaces; true for a single-surface shell), `ior`; water `density`
(absorption per metre, default 6) with `color` as the absorption tint; emissive `strength`; metal `anisotropic`;
wood `grain_scale`. Without `color`, a mesh's own material colours / textures (OBJ .mtl, GLB) are used.
Water `particles` render as thin droplets; bulk water should be exported as a closed meshseq surface.

Conventions (export.py): station-local frame, Z up, metres, quaternions wxyz. Rigid meshes are taken exactly as
Genesis loads them by default: OBJ/STL/PLY raw (Z up, no axis conversion; Blender's OBJ importer would otherwise
swap Y and Z), GLB/GLTF as Y-up glTF converted to Z-up (X, Y, Z) -> (X, -Z, Y), which is what both Genesis
(file_meshes_are_zup=None) and Blender's glTF importer do. A meta object may set "mesh_up": "z"|"y" to force it.
"""
import argparse
import fnmatch
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import layout as L  # noqa: E402
sys.path.insert(0, HERE)
from snapshot import SOURCES as RESULT_SOURCES  # noqa: E402  (cables -> out/cables/newton)

OUT = os.path.join(ROOT, "out")
LOCK = os.path.join(ROOT, "locks", "SAM3D_RUNNING")
I2RT = os.path.join(ROOT, "assets", "i2rt_upstream", "i2rt", "robot_models")


def log(*a):
    print("[compose]", *a, flush=True)


# ============================================================================ colour + materials
def srgb2lin(c):
    c = np.clip(np.asarray(c, float), 0, None)
    return tuple(np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


def rgba(c):
    return (*srgb2lin(c[:3]), 1.0)


KIND_DEFAULTS = {  # sRGB colour, roughness
    "wood": ((0.72, 0.55, 0.38), 0.55),
    "paint": ((0.80, 0.20, 0.18), 0.30),
    "plastic": ((0.90, 0.90, 0.90), 0.35),
    "rubber": ((0.12, 0.12, 0.12), 0.65),
    "ceramic": ((0.93, 0.92, 0.89), 0.45),
    "glass": ((0.965, 0.985, 0.978), 0.0),  # soda-lime green tint per surface (a jar shows 4 surfaces)
    "metal": ((0.80, 0.80, 0.82), 0.30),
    "water": ((0.72, 0.90, 0.97), 0.0),     # volume absorption colour
    "fabric": ((0.55, 0.55, 0.60), 1.0),
    "paper": ((0.95, 0.94, 0.90), 0.85),
    "emissive": ((1.0, 0.85, 0.6), 0.5),
}

_MATS = {}


class NT:
    """Small node-tree builder."""

    def __init__(self, mat):
        if mat.node_tree is None:
            mat.use_nodes = True
        self.t = mat.node_tree
        self.t.nodes.clear()
        self.x = 0

    def n(self, kind, **inputs):
        node = self.t.nodes.new(kind)
        node.location = (self.x, 0)
        self.x += 220
        for k, v in inputs.items():
            if hasattr(node, k) and not isinstance(getattr(node, k), bpy.types.bpy_prop_collection):
                setattr(node, k, v)
            else:
                self.set(node, k, v)
        return node

    def set(self, node, key, v):
        sock = node.inputs[key]
        if isinstance(v, bpy.types.NodeSocket):
            self.t.links.new(v, sock)
        else:
            sock.default_value = v

    def link(self, a, b):
        self.t.links.new(a, b)


def image_node(nt, img, has_uv):
    if has_uv:
        uv = nt.n("ShaderNodeTexCoord")
        tex = nt.n("ShaderNodeTexImage", image=img)
        nt.link(uv.outputs["UV"], tex.inputs["Vector"])
    else:
        co = nt.n("ShaderNodeTexCoord")
        tex = nt.n("ShaderNodeTexImage", image=img, projection="BOX", projection_blend=0.25)
        nt.link(co.outputs["Object"], tex.inputs["Vector"])
    return tex.outputs["Color"]


def wood_color(nt, col, scale=1.0, coord="Object"):
    """Procedural plank grain along local x: wave bands distorted by noise, plus fine pores."""
    tc = nt.n("ShaderNodeTexCoord")
    mp = nt.n("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (0.6 * scale, 6.0 * scale, 6.0 * scale)
    nt.link(tc.outputs[coord], mp.inputs["Vector"])
    noise = nt.n("ShaderNodeTexNoise", **{"Scale": 3.0, "Detail": 4.0, "Roughness": 0.6})
    nt.link(mp.outputs["Vector"], noise.inputs["Vector"])
    wave = nt.n("ShaderNodeTexWave", wave_type="BANDS", bands_direction="Y",
                **{"Scale": 2.2, "Distortion": 4.5, "Detail": 3.0, "Detail Scale": 1.5})
    add = nt.n("ShaderNodeVectorMath", operation="ADD")
    nt.link(mp.outputs["Vector"], add.inputs[0])
    nt.link(noise.outputs["Color"], add.inputs[1])
    nt.link(add.outputs["Vector"], wave.inputs["Vector"])
    ramp = nt.n("ShaderNodeValToRGB")
    lin = np.array(srgb2lin(col))
    ramp.color_ramp.elements[0].color = (*(lin * 0.80), 1)
    ramp.color_ramp.elements[1].color = (*np.clip(lin * 1.08, 0, 1), 1)
    ramp.color_ramp.elements[0].position = 0.25
    nt.link(wave.outputs["Fac"], ramp.inputs["Fac"])
    return ramp.outputs["Color"], wave.outputs["Fac"]


def make_material(spec, image=None, imported_color=None, has_uv=True, tag=""):
    """Material from an export `material` dict. image: bpy image (embedded or spec texture); imported_color: linear
    RGBA from the mesh file, used when the spec gives no colour."""
    kind = spec.get("kind", "plastic")
    if kind not in KIND_DEFAULTS:
        log(f"unknown material kind {kind!r}, using plastic")
        kind = "plastic"
    key = (json.dumps(spec, sort_keys=True), image.name if image else None,
           tuple(np.round(imported_color, 4)) if imported_color is not None else None, has_uv)
    if key in _MATS:
        return _MATS[key]
    dcol, drough = KIND_DEFAULTS[kind]
    if "color" in spec:
        lin = rgba(spec["color"])
    elif imported_color is not None and kind not in ("glass", "water", "metal"):
        lin = tuple(imported_color[:3]) + (1.0,)
    else:
        lin = rgba(dcol)
    rough = float(spec.get("roughness", drough))
    mat = bpy.data.materials.new(f"{kind}{tag}")
    nt = NT(mat)
    out = nt.n("ShaderNodeOutputMaterial")
    out.location = (1400, 0)
    p = nt.n("ShaderNodeBsdfPrincipled")
    p.location = (1100, 0)
    nt.link(p.outputs["BSDF"], out.inputs["Surface"])
    nt.set(p, "Roughness", rough)
    base = image_node(nt, image, has_uv) if image is not None else None

    if kind == "wood":
        if base is None:
            base, fac = wood_color(nt, spec.get("color", dcol), float(spec.get("grain_scale", 1.0)))
            bump = nt.n("ShaderNodeBump", **{"Strength": 0.04, "Distance": 0.0005})
            nt.link(fac, bump.inputs["Height"])
            nt.link(bump.outputs["Normal"], p.inputs["Normal"])
        nt.set(p, "Base Color", base)
        nt.set(p, "Specular IOR Level", 0.45)
        nt.set(p, "Coat Weight", float(spec.get("coat", 0.25)))
        nt.set(p, "Coat Roughness", 0.18)
    elif kind == "paint":
        nt.set(p, "Base Color", base if base is not None else lin)
        nt.set(p, "Coat Weight", 0.5)
        nt.set(p, "Coat Roughness", 0.08)
    elif kind == "plastic":
        nt.set(p, "Base Color", base if base is not None else lin)
        nt.set(p, "Specular IOR Level", 0.5)
    elif kind == "rubber":
        nt.set(p, "Base Color", base if base is not None else lin)
        nt.set(p, "Specular IOR Level", 0.3)
        nt.set(p, "Sheen Weight", 0.15)
    elif kind == "ceramic":
        nt.set(p, "Base Color", base if base is not None else lin)
        nt.set(p, "Coat Weight", 1.0)
        nt.set(p, "Coat Roughness", 0.04)
        nt.set(p, "Subsurface Weight", 0.08)
        nt.set(p, "Subsurface Scale", 0.002)
    elif kind == "glass":
        # Refracting glass by default (meshes that model both wall surfaces); "thin": true for single-surface meshes
        # (Principled Thin Wall: no bending, one sheet). The tint deepens at grazing angles (path through a wall of
        # thickness t is t / cos(theta)), which together with the flags outlines a jar on white.
        geo = nt.n("ShaderNodeNewGeometry")
        dot = nt.n("ShaderNodeVectorMath", operation="DOT_PRODUCT")
        nt.link(geo.outputs["Incoming"], dot.inputs[0])
        nt.link(geo.outputs["Normal"], dot.inputs[1])
        ab = nt.n("ShaderNodeMath", operation="ABSOLUTE")
        nt.link(dot.outputs["Value"], ab.inputs[0])
        mx = nt.n("ShaderNodeMath", operation="MAXIMUM")
        nt.link(ab.outputs[0], mx.inputs[0])
        mx.inputs[1].default_value = 0.06
        inv = nt.n("ShaderNodeMath", operation="DIVIDE")
        inv.inputs[0].default_value = float(spec.get("thickness_gain", 1.0))
        nt.link(mx.outputs[0], inv.inputs[1])
        gam = nt.n("ShaderNodeGamma")
        nt.set(gam, "Color", base if base is not None else lin)
        nt.link(inv.outputs[0], gam.inputs["Gamma"])
        nt.set(p, "Base Color", gam.outputs["Color"])
        nt.set(p, "Transmission Weight", 1.0)
        nt.set(p, "IOR", float(spec.get("ior", 1.5)))
        nt.set(p, "Thin Wall", bool(spec.get("thin", False)))
    elif kind == "metal":
        nt.set(p, "Base Color", base if base is not None else lin)
        nt.set(p, "Metallic", 1.0)
        nt.set(p, "Anisotropic", float(spec.get("anisotropic", 0.55)))
        tg = nt.n("ShaderNodeTangent", direction_type="RADIAL", axis="Z")
        nt.link(tg.outputs["Tangent"], p.inputs["Tangent"])
    elif kind == "water" and spec.get("_droplet"):
        # Water as loose particles: a refracting 3 mm sphere is mostly dark rim at video resolution and reads as a
        # black speck on white, so droplets get a thin-wall (non-bending) water shell with the tint, i.e. a light
        # bead with a Fresnel edge.
        nt.set(p, "Base Color", tuple(np.asarray(lin[:3]) ** 0.35) + (1.0,))
        nt.set(p, "Transmission Weight", 1.0)
        nt.set(p, "IOR", 1.333)
        nt.set(p, "Roughness", 0.05)
        nt.set(p, "Thin Wall", True)
    elif kind == "water":
        nt.set(p, "Base Color", (1.0, 1.0, 1.0, 1.0))
        nt.set(p, "Transmission Weight", 1.0)
        nt.set(p, "IOR", 1.333)
        nt.set(p, "Roughness", float(spec.get("roughness", 0.01)))
        va = nt.n("ShaderNodeVolumeAbsorption")
        va.inputs["Color"].default_value = lin
        va.inputs["Density"].default_value = float(spec.get("density", 6.0))
        nt.link(va.outputs[0], out.inputs["Volume"])
        # Shadow rays pass the surface (absorption still applies): with refractive caustics off, a refracting surface
        # blocks all direct light, and the inside of an opaque cup under water rendered nearly black.
        lp = nt.n("ShaderNodeLightPath")
        tr = nt.n("ShaderNodeBsdfTransparent")
        mx = nt.n("ShaderNodeMixShader")
        nt.link(lp.outputs["Is Shadow Ray"], mx.inputs["Fac"])
        nt.link(p.outputs["BSDF"], mx.inputs[1])
        nt.link(tr.outputs["BSDF"], mx.inputs[2])
        nt.link(mx.outputs["Shader"], out.inputs["Surface"])
    elif kind == "fabric":
        nt.set(p, "Base Color", base if base is not None else lin)
        nt.set(p, "Specular IOR Level", 0.15)
        nt.set(p, "Sheen Weight", 0.8)
        nt.set(p, "Sheen Roughness", 0.45)
    elif kind == "paper":
        nt.set(p, "Base Color", base if base is not None else lin)
        nt.set(p, "Specular IOR Level", 0.25)
        nt.set(p, "Subsurface Weight", 0.12)
        nt.set(p, "Subsurface Scale", 0.001)
    elif kind == "emissive":
        nt.set(p, "Base Color", base if base is not None else lin)
        nt.set(p, "Emission Color", base if base is not None else lin)
        nt.set(p, "Emission Strength", float(spec.get("strength", 5.0)))
    _MATS[key] = mat
    return mat


def simple_material(name, color_lin, rough=0.5, metallic=0.0, spec=0.5, aniso=0.0):
    mat = bpy.data.materials.new(name)
    nt = NT(mat)
    out = nt.n("ShaderNodeOutputMaterial")
    p = nt.n("ShaderNodeBsdfPrincipled", **{"Base Color": (*color_lin[:3], 1), "Roughness": rough,
                                            "Metallic": metallic, "Specular IOR Level": spec, "Anisotropic": aniso})
    nt.link(p.outputs["BSDF"], out.inputs["Surface"])
    return mat, nt, p


# ============================================================================ geometry helpers
def new_mesh_obj(name, V, F, mat=None, smooth=False, parent=None, collection=None):
    me = bpy.data.meshes.new(name)
    set_mesh(me, np.asarray(V, np.float32), np.asarray(F, np.int32), smooth=smooth)
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    if mat is not None:
        me.materials.append(mat)
    if parent is not None:
        ob.parent = parent
    return ob


def set_mesh(me, V, F, smooth=True):
    me.clear_geometry()
    V = np.asarray(V, np.float32).reshape(-1, 3)
    F = np.asarray(F, np.int32).reshape(-1, 3)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.ravel())
    me.loops.add(F.size)
    me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, F.size, 3, dtype=np.int32))
    me.update(calc_edges=True)
    if smooth:
        me.shade_smooth()
    else:
        me.shade_flat()


def box_mesh(sx, sy, sz, c=(0, 0, 0)):
    x, y, z = sx / 2, sy / 2, sz / 2
    V = np.array([(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z), (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]) + c
    F = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6),
         (3, 0, 4), (3, 4, 7)]
    return V, np.array(F)


def lathe(profile, n=128, flip=False):
    V, rings = [], []
    for r, z in profile:
        if r < 1e-9:
            rings.append([len(V)] * n)
            V.append((0, 0, z))
        else:
            rings.append(list(range(len(V), len(V) + n)))
            V += [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z) for i in range(n)]
    F = []
    for a, b in zip(rings[:-1], rings[1:]):
        for i in range(n):
            i2 = (i + 1) % n
            for t in ((a[i], a[i2], b[i2]), (a[i], b[i2], b[i])):
                if len(set(t)) == 3:
                    F.append(t[::-1] if flip else t)
    return np.array(V, float), np.array(F)


def bevel(ob, width, segments=3):
    m = ob.modifiers.new("bevel", "BEVEL")
    m.width = width
    m.segments = segments
    m.limit_method = "ANGLE"
    m.harden_normals = False
    return m


def look_at(cam, eye, target):
    eye, target = Vector(eye), Vector(target)
    cam.location = eye
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = (target - eye).to_track_quat("-Z", "Y")


def world_matrix(x, y, yaw_deg):
    return Matrix.Translation((x, y, 0.0)) @ Matrix.Rotation(math.radians(yaw_deg), 4, "Z")


# ============================================================================ studio
def build_studio(args):
    sc = bpy.context.scene
    # Seamless cove: flat floor disc, a quarter-round fillet, then a wall; floor grey fades to the background
    # grey with height so the horizon never shows.
    R0, rc, H, nseg = 9.0, 3.0, 9.0, 192
    prof = [(0.0, 0.0), (R0, 0.0)]
    for k in range(1, 13):
        a = math.pi / 2 * k / 12
        prof.append((R0 + rc * math.sin(a), rc * (1 - math.cos(a))))
    prof.append((R0 + rc, H))
    V, F = lathe(prof, n=nseg, flip=True)
    mat, nt, p = simple_material("studio", srgb2lin(L.FLOOR_RGB), rough=0.75, spec=0.12)
    sock = lambda coll, ident: next(x for x in coll if x.identifier == ident)
    geo = nt.n("ShaderNodeNewGeometry")
    sep = nt.n("ShaderNodeSeparateXYZ")
    nt.link(geo.outputs["Position"], sep.inputs[0])
    # h: 0 on the floor, 1 up the wall
    mr = nt.n("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP", **{"From Min": 0.05, "From Max": 0.8 * rc})
    nt.link(sep.outputs["Z"], mr.inputs["Value"])
    mix = nt.n("ShaderNodeMix", data_type="RGBA")
    sock(mix.inputs, "A_Color").default_value = rgba(L.FLOOR_RGB)
    sock(mix.inputs, "B_Color").default_value = rgba(L.BACKGROUND_RGB)
    nt.link(mr.outputs["Result"], sock(mix.inputs, "Factor_Float"))
    nt.link(sock(mix.outputs, "Result_Color"), p.inputs["Base Color"])
    # High-key cyc: the wall is "lit separately" -- an emission seen by camera and glossy rays only, so the background
    # reaches BACKGROUND_RGB under AgX without flooding the objects with light.
    lp = nt.n("ShaderNodeLightPath")
    cg = nt.n("ShaderNodeMath", operation="MAXIMUM")
    nt.link(lp.outputs["Is Camera Ray"], cg.inputs[0])
    nt.link(lp.outputs["Is Glossy Ray"], cg.inputs[1])
    k = nt.n("ShaderNodeMath", operation="MULTIPLY")
    nt.link(cg.outputs[0], k.inputs[0])
    nt.link(mr.outputs["Result"], k.inputs[1])
    em = nt.n("ShaderNodeEmission")
    em.inputs["Color"].default_value = rgba(L.BACKGROUND_RGB)
    st = nt.n("ShaderNodeMath", operation="MULTIPLY")
    st.inputs[1].default_value = args.bg
    nt.link(k.outputs[0], st.inputs[0])
    nt.link(st.outputs[0], em.inputs["Strength"])
    add = nt.n("ShaderNodeAddShader")
    outn = next(n for n in nt.t.nodes if n.type == "OUTPUT_MATERIAL")
    nt.link(p.outputs["BSDF"], add.inputs[0])
    nt.link(em.outputs["Emission"], add.inputs[1])
    nt.link(add.outputs["Shader"], outn.inputs["Surface"])
    cove = new_mesh_obj("studio_cove", V, F, mat, smooth=True)

    # World: neutral grey sky seen through the open top (fill light + what upward reflections see).
    w = bpy.data.worlds.new("studio_world")
    sc.world = w
    wnt = NT(w)
    bg = wnt.n("ShaderNodeBackground")
    bg.inputs["Color"].default_value = rgba(L.BACKGROUND_RGB)
    bg.inputs["Strength"].default_value = args.world
    wo = wnt.n("ShaderNodeOutputWorld")
    wnt.link(bg.outputs[0], wo.inputs["Surface"])

    def area(name, loc, target, size, power, shape="RECTANGLE", size_y=None, spread=180):
        ld = bpy.data.lights.new(name, "AREA")
        ld.shape = shape
        ld.size = size
        if size_y:
            ld.size_y = size_y
        ld.energy = power
        ld.spread = math.radians(spread)
        ob = bpy.data.objects.new(name, ld)
        sc.collection.objects.link(ob)
        look_at(ob, loc, target)
        return ob

    k = args.light
    # Big overhead soft box over the whole floor, a softer key from the camera side, a weak rim from behind.
    area("L_top", (0.0, 0.0, 7.5), (0.0, 0.0, 0.0), 16.0, 30000 * k, size_y=16.0)
    area("L_key", (5.5, 6.5, 5.0), (0.0, 0.0, 0.8), 4.0, 7500 * k, size_y=3.0)
    area("L_fill", (-6.5, 2.0, 3.5), (0.0, 0.0, 0.8), 4.0, 1500 * k, size_y=3.0)
    area("L_rim", (0.0, -7.0, 4.5), (0.0, 0.0, 1.0), 6.0, 1800 * k, size_y=2.0)

    # Flags: dark panels only glass/metal reflections can see (no camera, diffuse or shadow rays). On white walls
    # thin glass and water reflect nothing but white; these give them the dark edge lines of a product shot.
    fmat, _, _ = simple_material("flag", (0.015, 0.015, 0.016), rough=0.9, spec=0.0)
    for i in range(8):
        a = 2 * math.pi * (i + 0.5) / 8
        V, F = box_mesh(1.4, 0.02, 3.2)
        ob = new_mesh_obj(f"flag{i}", V, F, fmat)
        ob.location = (6.0 * math.cos(a), 6.0 * math.sin(a), 2.0)
        ob.rotation_euler = (0, 0, a + math.pi / 2)
        ob.visible_camera = False
        ob.visible_diffuse = False
        ob.visible_shadow = False
        ob.visible_volume_scatter = False
    return cove


# ============================================================================ furniture
FINISH = {  # per-station table finish: top kind, colours (sRGB), leg finish
    "mahjong": dict(top="wood", wood=(0.40, 0.26, 0.17), coat=0.7, legs="black"),  # sim: lacquered top, mu 0.35
    "jars": dict(top="wood", wood=(0.80, 0.66, 0.48), legs="black"),
    "cables": dict(top="white", wood=(0.80, 0.66, 0.48), legs="alu"),
    "water": dict(top="wood", wood=(0.86, 0.75, 0.60), legs="alu"),
}
DEFAULT_FINISH = dict(top="wood", wood=(0.78, 0.63, 0.45), legs="black")
_LEGMATS = {}


def leg_material(kind):
    if kind not in _LEGMATS:
        if kind == "alu":
            m, nt, p = simple_material("legs_alu", srgb2lin((0.80, 0.81, 0.83)), rough=0.28, metallic=1.0, aniso=0.5)
        else:
            m, nt, p = simple_material("legs_black", srgb2lin((0.06, 0.06, 0.065)), rough=0.38, spec=0.5)
        _LEGMATS[kind] = m
    return _LEGMATS[kind]


def build_table(name, Mw, finish):
    T = L.TABLE
    sx, sy, h, th = T["size_x"], T["size_y"], T["height"], T["top_thickness"]
    leg, inset = T["leg_size"], T["leg_inset"]
    root = bpy.data.objects.new(f"table_{name}", None)
    bpy.context.scene.collection.objects.link(root)
    root.matrix_world = Mw
    if finish["top"] == "white":
        top_mat, _, _ = simple_material(f"top_white_{name}", srgb2lin((0.93, 0.93, 0.92)), rough=0.45, spec=0.4)
    else:
        top_mat = make_material(dict(kind="wood", color=finish["wood"], coat=finish.get("coat", 0.35)),
                                tag=f"_table_{name}")
    V, F = box_mesh(sx, sy, th, (0, 0, h - th / 2))
    top = new_mesh_obj(f"top_{name}", V, F, top_mat, parent=root)
    bevel(top, 0.004, 3)
    if finish["top"] == "felt":  # felt inlay flush with the top surface, a wooden rim around it
        fm = make_material(dict(kind="fabric", color=finish["felt"]), tag=f"_felt_{name}")
        rim = 0.045
        V, F = box_mesh(sx - 2 * rim, sy - 2 * rim, 0.002, (0, 0, h - 0.0009))
        new_mesh_obj(f"felt_{name}", V, F, fm, parent=root)
    lm = leg_material(finish["legs"])
    xs, ys = sx / 2 - inset - leg / 2, sy / 2 - inset - leg / 2
    for i, (a, b) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        V, F = box_mesh(leg, leg, h - th - 0.008, (a * xs, b * ys, (h - th + 0.008) / 2))
        lo = new_mesh_obj(f"leg{i}_{name}", V, F, lm, parent=root)
        bevel(lo, 0.006, 3)
        V, F = box_mesh(leg * 0.9, leg * 0.9, 0.008, (a * xs, b * ys, 0.004))
        new_mesh_obj(f"foot{i}_{name}", V, F, leg_material("black"), parent=root)
    # apron rails tying the legs together, just under the top
    ah = 0.06
    for i, (cx, cy, lx, ly) in enumerate(((0, ys, 2 * xs, 0.02), (0, -ys, 2 * xs, 0.02),
                                          (xs, 0, 0.02, 2 * ys), (-xs, 0, 0.02, 2 * ys))):
        V, F = box_mesh(lx, ly, ah, (cx, cy, h - th - ah / 2))
        ro = new_mesh_obj(f"apron{i}_{name}", V, F, lm, parent=root)
        bevel(ro, 0.003, 2)
    return root


# ============================================================================ mesh import
_MESHCACHE = {}


def _find_base(mat):
    """(linear colour, image) feeding a node material's base colour, if any."""
    if mat is None or not mat.node_tree:
        return None, None
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            inp = node.inputs["Base Color"]
            stack = [l.from_node for l in inp.links]
            seen = set()
            while stack:
                nd = stack.pop()
                if nd in seen:
                    continue
                seen.add(nd)
                if nd.type == "TEX_IMAGE" and nd.image is not None:
                    return None, nd.image
                for i in nd.inputs:
                    stack += [l.from_node for l in i.links]
            return tuple(inp.default_value), None
    return None, None


def import_mesh(path, mesh_up=None):
    """-> list of (mesh datablock, local Matrix, [(colour, image) per slot]); cached per file."""
    key = (path, mesh_up)
    if key in _MESHCACHE:
        return _MESHCACHE[key]
    before = set(bpy.data.objects)
    ext = os.path.splitext(path)[1].lower()
    gltf = ext in (".glb", ".gltf")
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z", use_split_objects=False,
                              use_split_groups=False)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=path, forward_axis="Y", up_axis="Z")
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path, forward_axis="Y", up_axis="Z")
    elif gltf:
        bpy.ops.import_scene.gltf(filepath=path)
    else:
        raise ValueError(f"unsupported mesh format {path}")
    bpy.context.view_layer.update()
    new = [o for o in bpy.data.objects if o not in before]
    fix = Matrix.Identity(4)
    if gltf and mesh_up == "z":  # undo the importer's Y-up -> Z-up conversion
        fix = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))
    elif not gltf and mesh_up == "y":
        fix = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    parts = []
    for o in new:
        if o.type != "MESH":
            continue
        me = o.data
        if not me.has_custom_normals:
            me.shade_smooth()
            me.set_sharp_from_angle(angle=math.radians(35))
        slots = [_find_base(m) for m in me.materials] or [(None, None)]
        if len(me.materials) == 0:
            me.materials.append(None)
        parts.append((me, fix @ o.matrix_world.copy(), slots))
    for o in new:
        bpy.data.objects.remove(o)
    _MESHCACHE[key] = parts
    return parts


def load_image(path):
    for im in bpy.data.images:
        if im.filepath and os.path.abspath(bpy.path.abspath(im.filepath)) == os.path.abspath(path):
            return im
    return bpy.data.images.load(path)


# ============================================================================ stations
class Station:
    """One export directory placed in the world. apply(f) poses everything for export frame f."""

    def __init__(self, name, d, Mw, coll, only_prefix=None, hide=()):
        self.name, self.dir, self.Mw = name, d, Mw
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.n = int(self.meta.get("n_frames", 0))
        self.fps = float(self.meta.get("fps", L.RENDER_FPS))
        traj_p = os.path.join(d, "traj.npz")
        self.traj = np.load(traj_p) if os.path.exists(traj_p) else {}  # lazy: hidden objects are never read
        self.coll = coll
        self.root = bpy.data.objects.new(f"st_{name}", None)
        coll.objects.link(self.root)
        self.root.matrix_world = Mw
        self.updaters = []
        self.has_robot = any(k.startswith("robot/") for k in self.meta["objects"])
        t0 = time.time()
        counts = {}
        for oname, o in self.meta["objects"].items():
            if only_prefix and not oname.startswith(only_prefix):
                continue
            if any(fnmatch.fnmatch(oname, h) for h in hide):
                log(f"  {name}: hiding {oname}")
                continue
            kind = o["kind"]
            counts[kind] = counts.get(kind, 0) + 1
            getattr(self, f"_add_{kind}")(oname, o)
        log(f"station {name}: {self.n} frames, {counts}, built in {time.time() - t0:.1f}s")

    # -- helpers
    def _link(self, ob, parent=None):
        self.coll.objects.link(ob)
        ob.parent = parent or self.root
        return ob

    def _material(self, o, slot=(None, None), has_uv=True):
        spec = dict(o.get("material") or {})
        img = None
        if spec.get("texture"):
            tp = spec["texture"]
            for cand in (tp, os.path.join(self.dir, tp), os.path.join(self.dir, "meshes", os.path.basename(tp))):
                if os.path.exists(cand):
                    img = load_image(cand)
                    break
            else:
                log(f"  texture {tp} not found for {self.name}")
        if img is None:
            img = slot[1]
        return make_material(spec, image=img, imported_color=slot[0], has_uv=has_uv, tag=f"_{self.name}")

    # -- kinds
    def _add_rigid(self, oname, o):
        path = os.path.join(self.dir, o["mesh"])
        parts = import_mesh(path, o.get("mesh_up"))
        pose = bpy.data.objects.new(f"{self.name}:{oname}", None)
        self._link(pose)
        s = o.get("scale", 1.0)
        pose.scale = tuple(s) if isinstance(s, (list, tuple)) else (s, s, s)
        pose.rotation_mode = "QUATERNION"
        for k, (me, M, slots) in enumerate(parts):
            ob = bpy.data.objects.new(f"{self.name}:{oname}#{k}", me)
            self._link(ob, pose)
            ob.matrix_parent_inverse = Matrix.Identity(4)
            ob.matrix_basis = M
            has_uv = len(me.uv_layers) > 0
            for i, sl in enumerate(ob.material_slots):
                sl.link = "OBJECT"
                sl.material = self._material(o, slots[i] if i < len(slots) else (None, None), has_uv)
        if "static_pose" in o:
            p, q = o["static_pose"]
            pose.location, pose.rotation_quaternion = p, q
        else:
            P, Q = self.traj[f"{oname}/pos"], self.traj[f"{oname}/quat"]

            def upd(f, pose=pose, P=P, Q=Q):
                pose.location = P[f].tolist()
                pose.rotation_quaternion = Q[f].tolist()
            self.updaters.append(upd)

    def _add_rod(self, oname, o):
        P = self.traj[f"{oname}/pts"]
        cu = bpy.data.curves.new(f"{self.name}:{oname}", "CURVE")
        cu.dimensions = "3D"
        cu.bevel_mode = "ROUND"
        cu.bevel_depth = float(o["radius"])
        cu.bevel_resolution = 4
        cu.use_fill_caps = True
        cu.resolution_u = 4
        cu.twist_mode = "MINIMUM"
        sp = cu.splines.new("BEZIER")
        sp.bezier_points.add(P.shape[1] - 1)
        sp.use_smooth = True
        for bp in sp.bezier_points:
            bp.handle_left_type = bp.handle_right_type = "FREE"
        cu.materials.append(self._material(o))
        ob = bpy.data.objects.new(f"{self.name}:{oname}", cu)
        self._link(ob)

        def upd(f, cu=cu, P=P):
            p = P[f].astype(np.float64)
            m = np.empty_like(p)
            m[1:-1] = 0.5 * (p[2:] - p[:-2])
            m[0], m[-1] = p[1] - p[0], p[-1] - p[-2]
            bps = cu.splines[0].bezier_points
            bps.foreach_set("co", p.ravel())
            bps.foreach_set("handle_left", (p - m / 3).ravel())
            bps.foreach_set("handle_right", (p + m / 3).ravel())
            cu.update_tag()
        self.updaters.append(upd)

    def _add_particles(self, oname, o):
        P = self.traj[f"{oname}/pts"]
        me = bpy.data.meshes.new(f"{self.name}:{oname}")
        ob = bpy.data.objects.new(f"{self.name}:{oname}", me)
        self._link(ob)
        mod = ob.modifiers.new("points", "NODES")
        o = dict(o, material=dict(o.get("material") or {}, _droplet=True))
        mod.node_group = points_group(float(o["radius"]), self._material(o))

        def upd(f, me=me, P=P):
            p = P[f]
            p = p[np.isfinite(p).all(1)]
            if len(me.vertices) != len(p):
                me.clear_geometry()
                me.vertices.add(len(p))
            me.vertices.foreach_set("co", p.astype(np.float32).ravel())
            me.update()
        self.updaters.append(upd)

    def _add_meshseq(self, oname, o):
        d = os.path.join(self.dir, o["dir"])
        files = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
        idx = np.array([int(f[:-4]) for f in files])
        me = bpy.data.meshes.new(f"{self.name}:{oname}")
        mat = self._material(o, has_uv=False)
        me.materials.append(mat)
        ob = bpy.data.objects.new(f"{self.name}:{oname}", me)
        self._link(ob)
        state = {"cur": None}

        def upd(f, me=me, d=d, idx=idx, state=state, mat=mat):
            if len(idx) == 0:
                return
            j = int(idx[max(0, np.searchsorted(idx, f, side="right") - 1)])
            if state["cur"] == j:
                return
            z = np.load(os.path.join(d, f"{j:05d}.npz"))
            set_mesh(me, z["V"], z["F"], smooth=True)
            if len(me.materials) == 0:
                me.materials.append(mat)
            state["cur"] = j
        self.updaters.append(upd)

    def apply(self, f):
        f = min(max(int(f), 0), max(self.n - 1, 0))
        for u in self.updaters:
            u(f)


_PTGROUPS = {}


def points_group(radius, mat):
    key = (round(radius, 6), mat.name)
    if key in _PTGROUPS:
        return _PTGROUPS[key]
    ng = bpy.data.node_groups.new(f"points_{mat.name}_{radius:g}", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi, go = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    m2p = ng.nodes.new("GeometryNodeMeshToPoints")
    m2p.inputs["Radius"].default_value = radius
    sm = ng.nodes.new("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = mat
    ng.links.new(gi.outputs[0], m2p.inputs["Mesh"])
    ng.links.new(m2p.outputs["Points"], sm.inputs["Geometry"])
    ng.links.new(sm.outputs["Geometry"], go.inputs[0])
    _PTGROUPS[key] = ng
    return ng


# ============================================================================ robot preview (framing only)
def _rpy(xyz, rpy):
    r, p, y = rpy
    return Matrix.Translation(xyz) @ (Matrix.Rotation(y, 4, "Z") @ Matrix.Rotation(p, 4, "Y") @ Matrix.Rotation(r, 4, "X"))


def urdf_links(urdf):
    """Zero-pose link visuals of a URDF: [(mesh path, Matrix)]."""
    import xml.etree.ElementTree as ET
    root = ET.parse(urdf).getroot()
    base = os.path.dirname(urdf)

    def origin(el):
        o = el.find("origin")
        if o is None:
            return Matrix.Identity(4)
        return _rpy([float(v) for v in o.get("xyz", "0 0 0").split()], [float(v) for v in o.get("rpy", "0 0 0").split()])
    child_of = {}
    for j in root.findall("joint"):
        child_of[j.find("child").get("link")] = (j.find("parent").get("link"), origin(j))
    cache = {}

    def T(link):
        if link not in cache:
            cache[link] = (T(child_of[link][0]) @ child_of[link][1]) if link in child_of else Matrix.Identity(4)
        return cache[link]
    out = []
    for lk in root.findall("link"):
        for vis in lk.findall("visual"):
            m = vis.find("geometry/mesh")
            if m is not None:
                out.append((os.path.join(base, m.get("filename")), T(lk.get("name")) @ origin(vis)))
    return out


ROBOT_PREVIEW_DIRS = [os.path.join(ROOT, "robots", "scratch", "export_{arm}"),
                      os.path.join(ROOT, "robots", "scratch", "export_yam")]


def robot_preview(st_name, Mw, coll):
    """A parked cart at its home pose for framing: frame 0 of the robot agent's own Recorder test export if it is
    there (real meshes and materials), else the bare I2RT URDF arms at zero pose on a box."""
    arm = L.STATIONS[st_name]["arm"] if st_name in L.STATIONS else "yam"
    for d in ROBOT_PREVIEW_DIRS:
        d = d.format(arm=arm)
        if os.path.exists(os.path.join(d, "meta.json")):
            S = Station(f"preview_{st_name}", d, Mw, coll, only_prefix="robot/")
            S.apply(0)
            return S
    cands = sorted(d for d in os.listdir(os.path.join(I2RT, "arm", arm)) if d.startswith("v"))
    urdf = os.path.join(I2RT, "arm", arm, cands[-1], f"{arm}.urdf")
    links = urdf_links(urdf)
    RM = L.ROBOT_MOUNT
    body, _, _ = simple_material("preview_arm", srgb2lin((0.16, 0.16, 0.17)), rough=0.4)
    cart, _, _ = simple_material("preview_cart", srgb2lin((0.55, 0.56, 0.58)), rough=0.5)
    root = bpy.data.objects.new(f"robot_preview_{st_name}", None)
    coll.objects.link(root)
    root.matrix_world = Mw
    for side in (-1, 1):
        Mb = Matrix.Translation((side * RM["arm_spacing"] / 2, RM["arm_base_y"], RM["arm_base_z"])) @ \
            Matrix.Rotation(math.radians(RM["arm_yaw_deg"]), 4, "Z")
        for mesh, M in links:
            for me, Mp, _ in import_mesh(mesh):
                ob = bpy.data.objects.new(f"rp_{st_name}_{os.path.basename(mesh)}", me)
                coll.objects.link(ob)
                ob.parent = root
                ob.matrix_basis = Mb @ M @ Mp
                for sl in ob.material_slots:
                    sl.link = "OBJECT"
                    sl.material = body
    V, F = box_mesh(0.52, 0.52, RM["arm_base_z"] - 0.10, (0, RM["cart_center_y"], (RM["arm_base_z"] + 0.10) / 2 - 0.01))
    ob = new_mesh_obj(f"rp_cart_{st_name}", V, F, cart, parent=root)
    bevel(ob, 0.02, 3)


# ============================================================================ cameras
OVERVIEW = dict(target=(0.30, 0.05, 0.45), az=62.0, el=35.0, dist=7.4, lens=35.0)
CLOSE = dict(eye=(0.66, 1.12, 1.60), target=(0.0, -0.10, 0.88), lens=30.0)
OVER = dict(eye=(0.30, -1.55, 1.95), target=(0.0, 0.02, 0.78), lens=32.0)
# Per-station close-up defaults (station-local). Mahjong stands the tiles facing the robot, so its close-up looks
# from the robot's side, outside the arm span (behind the cart the mast and arms block the table).
CLOSE_BY_STATION = {
    "mahjong": dict(eye=(1.15, -0.80, 1.38), target=(-0.05, 0.08, 0.79), lens=35.0),
}


def vec3(s):
    return tuple(float(v) for v in s.split(","))


def make_camera(args, stations, places, frames):
    """-> (camera object, per-frame callback)."""
    cd = bpy.data.cameras.new("cam")
    cd.sensor_width = 36.0
    cd.clip_start, cd.clip_end = 0.03, 100.0
    cam = bpy.data.objects.new("cam", cd)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    name = args.camera
    if name == "flythrough":
        fp = get_fp(args)
        cd.lens = 35.0
        if args.fstop > 0:
            cd.dof.use_dof = True
            cd.dof.aperture_fstop = args.fstop

        def upd(f):
            eye, tgt, lens = fp.camera(f / fp.FPS)
            look_at(cam, eye, tgt)
            cd.lens = lens
            cd.dof.focus_distance = (Vector(eye) - Vector(tgt)).length
        return cam, upd
    if name in ("overview", "orbit"):
        o = dict(OVERVIEW)
        tgt = vec3(args.target) if args.target else o["target"]
        cd.lens = args.lens or o["lens"]
        sweep = args.orbit if args.orbit is not None else (20.0 if name == "orbit" else 0.0)
        if args.eye:
            eye = Vector(vec3(args.eye))
            rel = eye - Vector(tgt)
            az0 = math.degrees(math.atan2(rel.y, rel.x))
            el = math.degrees(math.atan2(rel.z, rel.xy.length))
            dist = rel.length
        else:
            az0, el, dist = o["az"], o["el"], o["dist"]
        f0, f1 = frames[0], frames[-1]

        def upd(f):
            u = 0.0 if f1 == f0 else (f - f0) / (f1 - f0)
            az = math.radians(az0 + sweep * (0.5 - 0.5 * math.cos(math.pi * u) - 0.5))
            e = math.radians(el)
            eye = Vector(tgt) + dist * Vector((math.cos(e) * math.cos(az), math.cos(e) * math.sin(az), math.sin(e)))
            look_at(cam, eye, tgt)
        return cam, upd
    st, _, variant = name.partition(":")
    if st not in places:
        raise SystemExit(f"camera {name!r}: unknown station {st!r} (have {sorted(places)})")
    S = stations.get(st)
    Mw = places[st]
    preset = dict(OVER if variant == "over" else CLOSE_BY_STATION.get(st, CLOSE))
    if S is not None and variant != "over" and isinstance(S.meta.get("camera"), dict):
        preset.update(S.meta["camera"])
    eye = vec3(args.eye) if args.eye else preset["eye"]
    tgt = vec3(args.target) if args.target else preset["target"]
    cd.lens = args.lens or preset["lens"]
    look_at(cam, Mw @ Vector(eye), Mw @ Vector(tgt))
    if args.fstop > 0:  # mild depth of field: the station sharp, the rest of the room soft
        cd.dof.use_dof = True
        cd.dof.focus_distance = (Vector(eye) - Vector(tgt)).length
        cd.dof.aperture_fstop = args.fstop
    return cam, (lambda f: None)


_FP = {}


def get_fp(args):
    """The camera-path module (fly-through), or one of its clips (--clip)."""
    if "fp" not in _FP:
        mod = load_path(args.path)
        _FP["fp"] = mod.get_clip(args.clip) if args.clip else mod
    return _FP["fp"]


def load_path(p):
    spec = importlib.util.spec_from_file_location("flypath", os.path.abspath(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_path(args, stations, frames, cam, cam_upd, station_frame):
    """No render: per frame, the eye's distance to the nearest scene surface (tables, robots, masts, props; not the
    studio) and whether something stands between the eye and the look-at point. Writes <out>/path_check.json."""
    from mathutils.bvhtree import BVHTree
    sc = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    dyn = {S.name for S in stations.values()}
    trees = {}
    for o in sc.objects:
        if o.type != "MESH" or o.name.startswith(("studio_cove", "flag")) or o.modifiers.get("points"):
            continue
        if len(o.data.polygons) == 0:  # meshseq placeholders, filled per frame: skipped
            continue
        trees[o.name] = (o, BVHTree.FromObject(o.evaluated_get(dg), dg))
    log(f"path check: {len(trees)} meshes, {len(frames)} frames")
    rows = []
    for f in frames:
        for S in stations.values():
            S.apply(station_frame(S, f))
        cam_upd(f)
        bpy.context.view_layer.update()
        eye = cam.matrix_world.translation.copy()
        best, who = 1e9, ""
        for name, (o, tree) in trees.items():
            M = o.matrix_world
            loc, _, _, _ = tree.find_nearest(M.inverted() @ eye)
            if loc is None:
                continue
            d = (M @ loc - eye).length
            if d < best:
                best, who = d, name
        fwd = cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))
        hit, hloc, _, _, hob, _ = sc.ray_cast(bpy.context.evaluated_depsgraph_get(), eye, fwd, distance=100.0)
        rows.append(dict(f=f, t=round(f / args.fps, 3), eye=[round(v, 3) for v in eye], clear=round(best, 3),
                         nearest=who, look_hits=(hob.name if hit else None),
                         look_dist=(round((hloc - eye).length, 3) if hit else None)))
    out = os.path.join(args.out, "path_check.json")
    os.makedirs(args.out, exist_ok=True)
    json.dump(rows, open(out, "w"), indent=0)
    worst = sorted(rows, key=lambda r: r["clear"])[:10]
    log(f"path check -> {out}; min clearance {worst[0]['clear']:.3f} m at frame {worst[0]['f']} ({worst[0]['nearest']})")
    for r in worst:
        log(f"   frame {r['f']:5d} t {r['t']:6.1f}s clear {r['clear']:.3f} m  nearest {r['nearest']}  looks at {r['look_hits']}")


# ============================================================================ render setup
def setup_render(args):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    w, h = (int(v) for v in args.res.lower().split("x"))
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = w, h, 100
    sc.render.fps = args.fps
    cy = sc.cycles
    if args.device == "GPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "OPTIX"
        cy.device = "GPU"
    cy.samples = args.samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.015
    cy.use_denoising = True
    try:
        cy.denoiser = "OPTIX" if args.device == "GPU" else "OPENIMAGEDENOISE"
    except TypeError:
        cy.denoiser = "OPENIMAGEDENOISE"
    cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    cy.max_bounces = 16
    cy.diffuse_bounces = 3
    cy.glossy_bounces = 6
    cy.transmission_bounces = 16
    cy.transparent_max_bounces = 16
    cy.volume_bounces = 0
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.sample_clamp_indirect = 6.0
    cy.blur_glossy = 1.0
    sc.render.film_transparent = False
    vs = sc.view_settings
    vs.view_transform = "AgX"
    try:
        vs.look = "AgX - Base Contrast"
    except TypeError:
        pass
    vs.exposure = args.exposure
    vs.gamma = 1.0
    sc.display_settings.display_device = "sRGB"
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    sc.render.use_persistent_data = False
    try:
        n = len(os.sched_getaffinity(0))
        sc.render.threads_mode = "FIXED"
        sc.render.threads = max(1, n)
    except Exception:
        pass


def gpu_free_mb():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                       "--format=csv,noheader,nounits"]).decode().split(",")
        return int(out[1]) - int(out[0])
    except Exception:
        return 1 << 20


def sam3d_running():
    """A GPU process running SAM 3D (other projects run it too, without our lock file)."""
    import re
    try:
        pids = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).decode().split()
    except Exception:
        return False
    for pid in pids:
        try:
            cmd = open(f"/proc/{pid.strip()}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if re.search(r"sam3d|sam-3d|sam_3d", cmd, re.I):
            return True
    return False


def wait_gpu(args):
    """Do not start a GPU frame while SAM 3D runs (lock file or process) or the card is nearly full."""
    if args.device != "GPU":
        return
    warned = False
    while os.path.exists(LOCK) or sam3d_running() or gpu_free_mb() < args.min_free_mb:
        if not warned:
            log(f"waiting: SAM3D lock={os.path.exists(LOCK)} process={sam3d_running()}, GPU free {gpu_free_mb()} MB")
            warned = True
        time.sleep(20)


def encode(frames_dir, frames, fps, dst):
    frames_dir = os.path.abspath(frames_dir)
    frames = [f for f in frames if os.path.exists(os.path.join(frames_dir, f"{f:05d}.png"))]
    tmp = os.path.join(frames_dir, "_seq")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for i, f in enumerate(frames):
        os.symlink(os.path.join(frames_dir, f"{f:05d}.png"), os.path.join(tmp, f"{i:05d}.png"))
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    cmd = ["nice", "-n", "10", "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
           os.path.join(tmp, "%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-crf", "18",
           "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst]
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"encoded {dst} ({len(frames)} frames @ {fps} fps)")


# ============================================================================ main
def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="compose.py")
    ap.add_argument("--camera", default="overview")
    ap.add_argument("--frames", default=None)
    ap.add_argument("--still", default=None)
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--out", default=os.path.join(HERE, "frames", "default"))
    ap.add_argument("--encode", default=None)
    ap.add_argument("--fps", type=int, default=L.RENDER_FPS)
    ap.add_argument("--stations", default=None)
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--robot-preview", action="store_true")
    ap.add_argument("--orbit", type=float, default=None)
    ap.add_argument("--eye", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--lens", type=float, default=None)
    ap.add_argument("--exposure", type=float, default=0.0)
    ap.add_argument("--fstop", type=float, default=4.0, help="close-up depth of field (0 = off)")
    ap.add_argument("--light", type=float, default=0.10, help="scale on all area-light powers")
    ap.add_argument("--bg", type=float, default=5.0, help="cove wall emission (camera/glossy rays only)")
    ap.add_argument("--world", type=float, default=0.35, help="world (sky) strength")
    ap.add_argument("--save-blend", default=None)
    ap.add_argument("--device", default="GPU", choices=("GPU", "CPU"))
    ap.add_argument("--min-free-mb", type=int, default=6000)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--source", action="append", default=[], help="NAME=DIR: read station NAME from DIR")
    ap.add_argument("--snap", default=None, help="read every station from DIR/<station> when it exists "
                                                  "(render/snapshot.py writes render/snap)")
    ap.add_argument("--hide", action="append", default=[], help="STATION:OBJECT glob, not drawn")
    ap.add_argument("--path", default=os.path.join(HERE, "flypath.py"), help="camera path module (flythrough)")
    ap.add_argument("--clip", default=None, help="per-station clip from --path (render/clips.py): that station "
                                                 "alone at the room centre")
    ap.add_argument("--skip-existing", action="store_true", help="skip frames whose PNG exists (resumable chunks)")
    ap.add_argument("--stats-name", default="render_stats.json")
    ap.add_argument("--check-path", type=int, default=0, help="no render: clearance check every N frames")
    ap.add_argument("--no-wait", action="store_true", help="do not wait for SAM 3D / free GPU memory")
    return ap.parse_args(argv)


def main():
    args = parse()
    args.out = os.path.abspath(args.out)
    if args.clip:
        args.camera = "flythrough"
        if os.path.basename(args.path) == "flypath.py":
            args.path = os.path.join(HERE, "clips.py")
    t_start = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    setup_render(args)
    build_studio(args)

    # where each station's export lives: --source NAME=DIR > --snap DIR/NAME > out/NAME
    srcs = {}
    for s in L.STATIONS:
        d = os.path.join(ROOT, RESULT_SOURCES.get(s, os.path.join("out", s)))
        if args.snap and os.path.exists(os.path.join(args.snap, s, "meta.json")):
            d = os.path.join(args.snap, s)
        srcs[s] = d
    for kv in args.source:
        k, _, v = kv.partition("=")
        srcs[k] = v if os.path.isabs(v) else os.path.join(ROOT, v)
    hide = {}
    for h in args.hide:
        k, _, v = h.partition(":")
        hide.setdefault(k, []).append(v)
    # tables for all four stations (always), then whatever exports exist
    names = args.stations.split(",") if args.stations else [s for s in L.STATIONS
                                                             if os.path.exists(os.path.join(srcs[s], "meta.json"))]
    stations = {}
    coll = bpy.data.collections.new("stations")
    sc.collection.children.link(coll)
    places = {s: world_matrix(*L.STATIONS[s]["world"]) for s in L.STATIONS}
    if args.clip:  # one station alone, at the room centre, so every clip gets the same light
        places = {args.clip: world_matrix(0.0, 0.0, 0.0)}
        names = [args.clip]
    for e in args.extra:
        nm, _, pose = e.partition("@")
        places[nm] = world_matrix(*(vec3(pose) if pose else (0, 0, 0)))
        names.append(nm)
        srcs.setdefault(nm, os.path.join(OUT, nm))
    for s, Mw in places.items():
        fin = dict(FINISH.get(s, DEFAULT_FINISH))
        mp = os.path.join(srcs.get(s, os.path.join(OUT, s)), "meta.json")
        if os.path.exists(mp):  # a station may ask for a finish that matches its simulated friction
            fin.update(json.load(open(mp)).get("table") or {})
        build_table(s, Mw, fin)
    for s in names:
        d = srcs.get(s, os.path.join(OUT, s))
        if not os.path.exists(os.path.join(d, "meta.json")):
            log(f"no export for {s}, table only")
            continue
        stations[s] = Station(s, d, places[s], coll, hide=hide.get(s, ()))
        log(f"  {s} <- {os.path.relpath(d, ROOT)}")
    if args.robot_preview:
        for s in L.STATIONS:
            if not (s in stations and stations[s].has_robot):
                robot_preview(s, places[s], coll)

    n_max = max([S.n for S in stations.values()] or [1])
    fp = get_fp(args) if args.camera == "flythrough" else None
    if fp is not None:
        n_max = fp.N_FRAMES

        def station_frame(S, f):  # video frame -> the station's own export frame, through its clock
            return int(round(fp.sim_time(S.name, f / fp.FPS) * S.fps))
    else:
        def station_frame(S, f):
            return f
    if args.still:
        frames = [int(v) for v in args.still.split(",")]
    elif args.frames:
        parts = [int(v) if v else None for v in args.frames.split(":")]
        a = parts[0] or 0
        b = parts[1] if len(parts) > 1 and parts[1] is not None else n_max
        st = parts[2] if len(parts) > 2 and parts[2] else 1
        frames = list(range(a, b, st))
    else:
        frames = list(range(n_max))
    cam, cam_upd = make_camera(args, stations, places, frames)
    log(f"scene built in {time.time() - t_start:.1f}s; {len(frames)} frames, camera {args.camera}")
    if args.check_path:
        check_path(args, stations, frames[::args.check_path], cam, cam_upd, station_frame)
        return

    if args.save_blend:
        for S in stations.values():
            S.apply(station_frame(S, frames[0]))
        cam_upd(frames[0])
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save_blend))
    if args.no_render:
        if args.encode:
            encode(args.out, frames, args.fps, args.encode)
        return
    os.makedirs(args.out, exist_ok=True)
    stats_p = os.path.join(args.out, args.stats_name)
    snaps = {}
    for s, S in stations.items():
        sp = os.path.join(S.dir, "SNAPSHOT.json")
        snaps[s] = json.load(open(sp)) if os.path.exists(sp) else dict(
            source=os.path.relpath(S.dir, ROOT),
            source_meta_mtime_iso=time.strftime("%Y-%m-%d %H:%M:%S",
                                                time.localtime(os.path.getmtime(os.path.join(S.dir, "meta.json")))))
    stats = dict(camera=args.camera, res=args.res, samples=args.samples, device=args.device,
                 blender=bpy.app.version_string, stations=sorted(stations), extra=args.extra, fps=args.fps,
                 sources=snaps, hide=args.hide, host=os.uname().nodename,
                 gpu=next((d.name for d in bpy.context.preferences.addons["cycles"].preferences.devices
                           if d.use and d.type == "OPTIX"), None) if args.device == "GPU" else "CPU",
                 frames={}, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    if fp is not None:
        stats["path"] = dict(file=os.path.relpath(os.path.abspath(args.path), ROOT),
                             mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(args.path))),
                             clocks={k: [list(p) for p in v] for k, v in fp.CLOCKS.items()},
                             duration=fp.DURATION, n_frames=fp.N_FRAMES)
    if os.path.exists(stats_p):
        try:
            old = json.load(open(stats_p))
            if (old.get("camera"), old.get("res"), old.get("samples")) == (args.camera, args.res, args.samples):
                stats["frames"] = old.get("frames", {})
        except Exception:
            pass
    for f in frames:
        final = os.path.join(args.out, f"{f:05d}.png")
        if args.skip_existing and os.path.exists(final) and os.path.getsize(final) > 0:
            continue
        if not args.no_wait:
            wait_gpu(args)
        t0 = time.time()
        for S in stations.values():
            S.apply(station_frame(S, f))
        cam_upd(f)
        sc.frame_set(f)
        tmp = os.path.join(args.out, f"_tmp_{os.getpid()}_{f:05d}.png")
        sc.render.filepath = tmp
        bpy.ops.render.render(write_still=True)
        os.replace(tmp, final)  # a preempted job never leaves a half-written frame under the final name
        dt = time.time() - t0
        stats["frames"][str(f)] = round(dt, 2)
        log(f"frame {f} {dt:.1f}s -> {final}")
        tmp_s = stats_p + f".{os.getpid()}"
        json.dump(stats, open(tmp_s, "w"), indent=1)
        os.replace(tmp_s, stats_p)
    if args.encode and not args.still:
        encode(args.out, frames, args.fps, args.encode)
    log(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
