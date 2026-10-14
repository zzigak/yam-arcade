"""blender -b -P blender_views.py -- OUTDIR NAME [NAME...]: renders views of assets/sam3d/NAME/visual.glb and the
collision section (_work/qa/NAME_section.obj) with Workbench, vertex colours, orthographic cameras."""
import bpy, sys, os, math
from mathutils import Vector
argv = sys.argv[sys.argv.index("--") + 1:]
out, names = argv[0], argv[1:]
ROOT = os.path.expanduser("~/projects/yam-arcade/assets/sam3d")

def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)

def setup():
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "VERTEX"
    sc.display.shading.show_cavity = True
    sc.display.shading.show_shadows = False
    sc.render.resolution_x = sc.render.resolution_y = 480
    sc.render.film_transparent = False
    sc.world = sc.world or bpy.data.worlds.new("w")
    sc.view_settings.view_transform = "Standard"
    return sc

def bbox(objs):
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)]); hi = Vector([max(p[i] for p in pts) for i in range(3)])
    return lo, hi

def shoot(sc, objs, az, el, path, color_type="VERTEX"):
    sc.display.shading.color_type = color_type
    lo, hi = bbox(objs); c = (lo + hi) / 2; r = (hi - lo).length / 2
    cam_data = bpy.data.cameras.new("c"); cam_data.type = "ORTHO"; cam_data.ortho_scale = 2.15 * r
    cam = bpy.data.objects.new("c", cam_data); sc.collection.objects.link(cam); sc.camera = cam
    d = Vector((math.cos(math.radians(el)) * math.sin(math.radians(az)), -math.cos(math.radians(el)) * math.cos(math.radians(az)),
                math.sin(math.radians(el))))
    cam.location = c + d * (4 * r + 0.1)
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    cam_data.clip_end = 100
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam, do_unlink=True)

sc = setup()
for n in names:
    clear()
    bpy.ops.import_scene.gltf(filepath=f"{ROOT}/{n}/visual.glb")
    objs = [o for o in bpy.data.objects if o.type == "MESH"]
    shoot(sc, objs, -35, 30, (f"{out}/" + n.replace("/", "__") + "_v1.png"))
    shoot(sc, objs, 0, 3, (f"{out}/" + n.replace("/", "__") + "_v2.png"))
    shoot(sc, objs, 25, 65, (f"{out}/" + n.replace("/", "__") + "_v3.png"))
    sec = f"{ROOT}/_work/qa/" + n.replace("/", "__") + "_section.obj"
    if os.path.exists(sec):
        clear()
        bpy.ops.wm.obj_import(filepath=sec, forward_axis="Y", up_axis="Z")
        objs = [o for o in bpy.data.objects if o.type == "MESH"]
        shoot(sc, objs, 0, 15, (f"{out}/" + n.replace("/", "__") + "_v4.png"), color_type="SINGLE")
