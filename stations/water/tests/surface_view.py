"""Render exported water-surface frames (meshseq) with the containers at their recorded poses (CPU rasterizer).

usage: python surface_view.py EXPORT_DIR FRAME [FRAME ...]
"""
import json
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import station as st  # noqa: E402
import genesis as gs  # noqa: E402
import imageio.v3 as iio  # noqa: E402

exp = sys.argv[1]
frames = [int(f) for f in sys.argv[2:]]
meta = json.load(open(os.path.join(exp, "meta.json")))
with np.load(os.path.join(exp, "traj.npz")) as _z:
    traj = {k: _z[k] for k in _z.files}
gs.init(backend=gs.cpu, logging_level="warning")
for f in frames:
    scene = gs.Scene(show_viewer=False, vis_options=gs.options.VisOptions(
        background_color=(0.92, 0.92, 0.93), ambient_light=(0.4, 0.4, 0.4), show_world_frame=False,
        lights=[dict(type="directional", dir=(-0.4, 0.6, -1.0), color=(1.0, 1.0, 1.0), intensity=3.5)]))
    scene.add_entity(gs.morphs.Box(pos=(0, 0, 0.735), size=(1.2, 0.7, 0.03), fixed=True, collision=False),
                     surface=gs.surfaces.Default(color=(0.86, 0.80, 0.70)))
    for name, o in meta["objects"].items():
        if o["kind"] != "rigid" or name.startswith("robot/"):
            continue
        pos, quat = traj[f"{name}/pos"][f], traj[f"{name}/quat"][f]
        mesh = os.path.join(exp, o["mesh"])
        glass = o["material"]["kind"] == "glass"
        scene.add_entity(gs.morphs.Mesh(file=mesh, pos=tuple(pos), quat=tuple(quat), fixed=True, collision=False,
                                        file_meshes_are_zup=not mesh.endswith(".glb")),
                         surface=gs.surfaces.Default(color=(0.85, 0.9, 0.95, 0.25) if glass else
                                                     tuple(o["material"].get("color", (0.8, 0.8, 0.8))) + (1.0,)))
    d = np.load(os.path.join(exp, "meshseq", "water_surface", f"{f:05d}.npz"))
    wpath = os.path.join(HERE, "out_surface_view", f"surf_{f:05d}.obj")
    os.makedirs(os.path.dirname(wpath), exist_ok=True)
    trimesh.Trimesh(d["V"], d["F"], process=False).export(wpath)
    scene.add_entity(gs.morphs.Mesh(file=wpath, fixed=True, collision=False),
                     surface=gs.surfaces.Default(color=(0.30, 0.55, 0.95, 0.85)))
    cams = [scene.add_camera(res=(1280, 720), pos=(-0.55, -0.35, 1.02), lookat=(0.02, 0.05, 0.83), fov=40, GUI=False),
            scene.add_camera(res=(1280, 720), pos=(0.15, 0.10, 1.25), lookat=(0.0, 0.05, 0.80), fov=55, GUI=False)]
    scene.build()
    for i, cam in enumerate(cams):
        iio.imwrite(os.path.join(HERE, "out_surface_view", f"view{i}_{f:05d}.png"), cam.render()[0])
    scene.destroy() if hasattr(scene, "destroy") else None
    print("rendered", f, flush=True)
