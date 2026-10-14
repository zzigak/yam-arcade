"""Export-frame check (CPU, dry scene): for every container, Genesis' own visual-geom vertices (world) must coincide
with the exported OBJ transformed by the pose we record (link pos/quat).  Writes out/water/frame_check.json."""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import station as st  # noqa: E402
import trimesh  # noqa: E402
from export_water import export_meshes  # noqa: E402
from genesis.utils.misc import tensor_to_array as A  # noqa: E402

env = st.build(backend="cpu", pitcher_mount="free", water_ml=0, substeps=4)
for _ in range(30):            # let the cups settle on the table so the poses are non-trivial
    env.scene.step()
paths, _ = export_meshes(os.path.join(os.path.dirname(HERE), "assets", "export"))
res = {}
for name in ("pitcher", "tumbler", "mug", "teacup"):
    pos, quat = env.pose(name)
    R = st.quat_to_R(quat)
    files = [paths[name]] + ([paths["pitcher_lid"]] if name == "pitcher" else [])
    def load_as_compositor(f):
        m = trimesh.load(f, process=False, force="mesh")
        v = m.vertices
        if f.endswith(".glb"):   # glTF is Y-up: the compositor (and Genesis) map (X, Y, Z) -> (X, -Z, Y)
            v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)
        return v
    V = np.concatenate([load_as_compositor(f) for f in files]) @ R.T + pos
    tree = cKDTree(V)
    link = env.links[name]
    gv = np.concatenate([A(g.get_vverts()) for g in link.vgeoms])
    d, _ = tree.query(gv)
    res[name] = dict(genesis_verts=int(len(gv)), max_dist_mm=float(d.max() * 1e3), pos=pos.tolist(), quat=quat.tolist())
    print(name, res[name])
os.makedirs(os.path.join(st.ROOT, "out", "water"), exist_ok=True)
json.dump(res, open(os.path.join(st.ROOT, "out", "water", "frame_check.json"), "w"), indent=1)
