"""Export check: register a cart with export.Recorder, record a short move, and verify that each Genesis link frame
equals the MuJoCo body frame the exported OBJs are expressed in (same qpos), so the compositor places meshes right.

    ~/simquarry/genesis-world/.venv/bin/python ~/projects/yam-arcade/robots/test_export.py [variant]
"""
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import genesis as gs  # noqa: E402
import layout  # noqa: E402
from export import Recorder  # noqa: E402
from yam_cart import SIDES, add_cart, model_path  # noqa: E402
from genesis.utils.misc import tensor_to_array as A  # noqa: E402

variant = sys.argv[1] if len(sys.argv) > 1 else "yam"
gs.init(backend=gs.cpu, logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False)
scene.add_entity(gs.morphs.Plane())
cart = add_cart(scene, gs, variant)
scene.build()
cart.after_build()
out = os.path.join(HERE, "scratch", f"export_{variant}")
rec = Recorder(out, station="robot_test", fps=layout.RENDER_FPS)
cart.register(rec)
q_goal, tilt = cart.ik_grasp("left", (-0.2, 0.0, layout.TABLE_TOP_Z + 0.05))
path = cart.joint_path("left", q_goal, duration=1.0, dt=0.01)
cart.set_base_target(0.05, 0.0, 0.05)  # drive the base a little too, to exercise the virtual joints
for k, q in enumerate(path):
    cart.set_arm_target("left", q)
    cart.update()
    scene.step()
    if k % 3 == 0:
        cart.record(rec, k * 0.01)
rec.close()

# Genesis link frames vs MuJoCo body frames at the same qpos
m = mujoco.MjModel.from_xml_path(model_path(variant))
d = mujoco.MjData(m)
qg = cart.qpos()
for j in cart.entity.joints:
    if j.n_dofs:
        d.qpos[m.joint(j.name).qposadr[0]] = qg[j.dofs_idx_local[0]]
mujoco.mj_forward(m, d)
base = cart.base_pos
err_p, err_q = 0.0, 0.0
for link in cart.entity.links:
    b = m.body(link.name)
    pg, qgq = A(link.get_pos()), A(link.get_quat())
    pm, qm = d.xpos[b.id] + base, d.xquat[b.id]
    err_p = max(err_p, float(np.linalg.norm(pg - pm)))
    err_q = max(err_q, float(1 - abs(np.dot(qgq, qm))))
meta = json.load(open(os.path.join(out, "meta.json")))
res = dict(n_objects=len(meta["objects"]), n_frames=meta["n_frames"], max_link_pos_err=err_p, max_link_quat_err=err_q,
           objects=sorted(meta["objects"])[:8], base_q=qg[cart.base_dofs].round(4).tolist(), tilt=tilt)
print(json.dumps(res, indent=1))
assert err_p < 1e-4 and err_q < 1e-6
json.dump(res, open(os.path.join(HERE, "checks", f"export_{variant}.json"), "w"), indent=1)
