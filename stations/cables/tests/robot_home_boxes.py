"""CPU: world AABBs of every YAM Ultra cart link at the home pose -> stations/cables/assets/robot_home_aabbs.json
(keep-out volume for the initial cable placement)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import station as S
import genesis as gs
from genesis.utils.misc import tensor_to_array as A
from yam_cart import add_cart
gs.init(backend=gs.cpu, logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=S.DT), show_viewer=False)
cart = add_cart(scene, gs, S.layout.STATIONS["cables"]["arm"], ipc=False)
scene.build(); cart.after_build()
out = {}
for link in cart.entity.links:
    vs = [A(g.get_verts()) for g in link.geoms] + [A(v.get_vverts()) for v in link.vgeoms]
    vs = [v for v in vs if len(v)]
    if vs:
        V = np.vstack(vs)
        out[link.name] = [V.min(0).tolist(), V.max(0).tolist()]
json.dump(out, open(os.path.join(S.ASSETS, "robot_home_aabbs.json"), "w"), indent=1)
print(len(out), "links;", {k: [np.round(v[0], 3).tolist(), np.round(v[1], 3).tolist()] for k, v in out.items() if "tip" in k or "gripper" in k})
