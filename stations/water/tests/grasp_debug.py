"""Grasp-only debug (CPU): run RobotDriver.prepare(), print robot<->pitcher contacts and render close views."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import station as st  # noqa: E402
import robot_pour  # noqa: E402
from containers import ALL  # noqa: E402
from genesis.utils.misc import tensor_to_array as A  # noqa: E402

ALL["pitcher"].rgba = (0.55, 0.75, 0.9, 1.0)   # opaque for the debug views
out = os.path.join(HERE, "out_grasp_debug")
os.makedirs(out, exist_ok=True)
hx = st.PLACEMENT["pitcher"][0] + 0.118
hy = st.PLACEMENT["pitcher"][1]
cams = dict(
    side=dict(pos=(hx + 0.35, hy - 0.05, 0.92), lookat=(hx, hy, 0.89), fov=40, res=(800, 600)),
    top=dict(pos=(hx + 0.01, hy - 0.01, 1.30), lookat=(hx, hy, 0.89), fov=35, res=(800, 600)),
    back=dict(pos=(hx - 0.02, hy + 0.40, 0.95), lookat=(hx, hy, 0.89), fov=40, res=(800, 600)),
)
env = st.build(backend="cpu", pitcher_mount="free", water_ml=0, ballast_kg=1.2, robot_builder=robot_pour.builder,
               cameras=cams, substeps=10)
drv = robot_pour.RobotDriver(env)
cart = env.robot.cart
gen = drv.prepare()
k = 0
pent = env.entities["pitcher"]
links = env.scene.rigid_solver.links
import imageio.v3 as iio  # noqa: E402
for _ in gen:
    drv.update()
    env.scene.step()
    k += 1
    if k % 60 == 0:
        c = cart.entity.get_contacts(with_entity=pent)
        la, lb = A(c["link_a"]), A(c["link_b"])
        pos = A(c["position"])
        names = sorted({(links[a].name, links[b].name) for a, b in zip(la, lb)})
        print(f"t={k * env.scene.sim_options.dt:.2f} width={cart.gripper_width('right') * 1e3:.1f}mm "
              f"contacts={len(la)} {names} pos={np.round(pos[:3], 3).tolist()}", flush=True)
    if k % 150 == 0:
        for cn, cam in env.cams.items():
            iio.imwrite(os.path.join(out, f"{cn}_{k:05d}.png"), cam.render()[0])
tcp, fq = cart.tcp_pose("right")
print("TCP", tcp, "pitcher grip point", env.pose("pitcher")[0] + st.quat_to_R(env.pose("pitcher")[1]) @ robot_pour.GRIP_LOCAL)
print(env.robot.log)
for cn, cam in env.cams.items():
    iio.imwrite(os.path.join(out, f"{cn}_final.png"), cam.render()[0])
from yam_cart import mat_from_quat  # noqa: E402
R = mat_from_quat(fq)
print("flange x (closing)", np.round(R[:, 0], 3), " y", np.round(R[:, 1], 3), " z (approach)", np.round(R[:, 2], 3))
print("commanded R", np.round(drv.R_grasp0, 3).tolist())
print("arm q", np.round(cart.arm_q("right"), 3), "target", np.round(cart.q_target[cart.arm_dofs["right"]], 3))
for ln in ("right_tip_left", "right_tip_right", "right_gripper"):
    link = cart.entity.get_link(ln)
    for g in link.geoms:
        v = A(g.get_verts())
        print(ln, "geom verts min", np.round(v.min(0), 4), "max", np.round(v.max(0), 4), "centre", np.round(v.mean(0), 4))
