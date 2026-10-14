"""Grasp-and-lift test: one cart, a fixed table top at station height, a 4 cm cube on it; the left arm grasps the
cube top-down and lifts it 10 cm.

    ~/simquarry/genesis-world/.venv/bin/python ~/projects/yam-arcade/robots/test_grasp.py --variant yam          # rigid, gs.cpu
    ~/simquarry/genesis-world/.venv/bin/python ~/projects/yam-arcade/robots/test_grasp.py --variant yam --ipc    # libuipc, gs.gpu

IPC mode: cube and table are `ipc_only` rigid bodies; the robot is coupled `two_way_soft_constraint` through its
fingertip pads and palms (Cart.default_coup_links), Franka settings (memory: franka-deformables-genesis-newton).
Writes robots/checks/grasp_<rigid|ipc>_<variant>.json (+ an mp4 with --video).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import genesis as gs  # noqa: E402
import layout  # noqa: E402
from yam_cart import add_cart  # noqa: E402
from genesis.utils.misc import tensor_to_array as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="yam")
ap.add_argument("--ipc", action="store_true")
ap.add_argument("--dt", type=float, default=0.01)
ap.add_argument("--cube", type=float, nargs=2, default=(-0.25, -0.10), help="cube x y (station frame)")
ap.add_argument("--force", type=float, default=10.0, help="grip force per finger, N")
ap.add_argument("--coupling", type=float, default=100.0)
ap.add_argument("--video", action="store_true")
ap.add_argument("--coup", default="default", help="'default' (pads + palms) | 'tips' (pads only) | comma list")
args = ap.parse_args()

gs.init(backend=gs.gpu if args.ipc else gs.cpu, logging_level="warning")
T = layout.TABLE
TOP = T["height"]
CUBE = 0.04
kw = {}
if args.ipc:
    kw["coupler_options"] = gs.options.IPCCouplerOptions(
        contact_d_hat=0.001, contact_resistance=1e7,
        constraint_strength_translation=args.coupling, constraint_strength_rotation=args.coupling,
        newton_semi_implicit_enable=False, enable_rigid_rigid_contact=True, enable_rigid_ground_contact=True)
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=args.dt), show_viewer=False,
                 vis_options=gs.options.VisOptions(background_color=layout.BACKGROUND_RGB, show_world_frame=False),
                 **kw)
ipc_mat = (lambda mu: gs.materials.Rigid(coup_type="ipc_only", coup_friction=mu)) if args.ipc else \
    (lambda mu: gs.materials.Rigid(friction=mu))
scene.add_entity(gs.morphs.Plane(), material=ipc_mat(0.6), surface=gs.surfaces.Default(color=layout.FLOOR_RGB))
table = scene.add_entity(gs.morphs.Box(pos=(0, 0, TOP - T["top_thickness"] / 2), size=(T["size_x"], T["size_y"], T["top_thickness"]),
                                       fixed=True), material=ipc_mat(0.5), surface=gs.surfaces.Default(color=(0.75, 0.62, 0.45)))
cube_z0 = TOP + CUBE / 2 + (0.002 if args.ipc else 0.0)  # IPC rejects zero-distance contact at init
cmat = gs.materials.Rigid(rho=700.0, coup_type="ipc_only", coup_friction=0.6) if args.ipc else gs.materials.Rigid(rho=700.0, friction=0.6)
cube = scene.add_entity(gs.morphs.Box(pos=(*args.cube, cube_z0), size=(CUBE,) * 3), material=cmat,
                        surface=gs.surfaces.Default(color=(0.85, 0.25, 0.2)))
coup = None if args.coup == "default" else (
    tuple(f"{s}_{n}" for s in ("left", "right") for n in ("tip_left", "tip_right")) if args.coup == "tips"
    else tuple(args.coup.split(",")))
cart = add_cart(scene, gs, args.variant, ipc=args.ipc, coup_links=coup)
cam = scene.add_camera(res=(640, 480), pos=(args.cube[0] + 0.45, args.cube[1] - 0.55, TOP + 0.35),
                       lookat=(args.cube[0], args.cube[1], TOP + 0.06), fov=45, GUI=False) if args.video else None
scene.build()
cart.after_build()
dt = args.dt
side = "left"
cx, cy = args.cube
log, frames = [], []
t_wall = time.time()
step_i = [0]


def step(phase):
    cart.update()
    try:
        scene.step()
    except Exception:
        print("FAILED at step", step_i[0], phase, "qpos", cart.qpos().round(4).tolist())
        print("last log", log[-3:] if log else None)
        raise
    step_i[0] += 1
    if args.ipc and step_i[0] % 10 == 0:
        print("step", step_i[0], phase, "cube", A(cube.get_pos()).round(4), "tcp", cart.tcp_pose(side)[0].round(4),
              "w", round(cart.gripper_width(side), 4), flush=True)
    if step_i[0] % 3 == 0:
        p = A(cube.get_pos())
        tcp = cart.tcp_pose(side)[0]
        log.append(dict(t=round(step_i[0] * dt, 3), phase=phase, cube=p.round(5).tolist(), tcp=tcp.round(5).tolist(),
                        width=round(cart.gripper_width(side), 5), finger_force=cart.finger_force(side).round(3).tolist(),
                        robot_zmin=round(min(float(A(l.get_AABB())[0, 2]) for l in cart.entity.links
                                             if l.name.startswith(side) and len(l.geoms)), 4)))
        if cam is not None:
            frames.append(np.asarray(A(cam.render(rgb=True)[0]), dtype=np.uint8))


def move(target, duration, phase):
    q, tilt = cart.ik_grasp(side, target, yaw=0.0)
    assert q is not None, f"unreachable {target}"
    for qk in cart.joint_path(side, q, duration, dt):
        cart.set_arm_target(side, qk)
        step(phase)


for _ in range(20):
    step("settle")
cube_start = A(cube.get_pos()).copy()
TCP_GRASP_Z = TOP + 0.006  # fingertip ends 6 mm above the table: pads cover the cube's full height
move((cx, cy, TOP + 0.15), 1.2, "approach")
move((cx, cy, TCP_GRASP_Z), 1.0, "descend")
for _ in range(20):
    step("descend")
cart.grasp(side, force=args.force)
for _ in range(80):
    step("close")
width_grip = cart.gripper_width(side)
move((cx, cy, TCP_GRASP_Z + 0.10), 1.5, "lift")
for _ in range(100):
    step("hold")
cube_end = A(cube.get_pos())
tcp_end = cart.tcp_pose(side)[0]

lift = float(cube_end[2] - cube_start[2])
slip = float((tcp_end[2] - cube_end[2]) - (TCP_GRASP_Z - cube_start[2]))
res = dict(variant=args.variant, mode="ipc" if args.ipc else "rigid", dt=dt, grip_force_N=args.force,
           cube_start=cube_start.round(4).tolist(), cube_end=cube_end.round(4).tolist(), lift_m=round(lift, 4),
           slip_m=round(slip, 4), width_at_grip=round(width_grip, 4),
           finger_force_hold=cart.finger_force(side).round(3).tolist(),
           robot_zmin_min=min(r["robot_zmin"] for r in log), wall_s=round(time.time() - t_wall, 1))
res["checks"] = dict(lifted=dict(value=res["lift_m"], limit=">= 0.095", ok=lift >= 0.095),
                     slip=dict(value=res["slip_m"], limit="|.| <= 0.005", ok=abs(slip) <= 0.005),
                     width=dict(value=res["width_at_grip"], limit="0.035..0.041 (4 cm cube, pads squeeze)",
                                ok=0.035 <= width_grip <= 0.041))
print(json.dumps(res, indent=1))
os.makedirs(os.path.join(HERE, "checks"), exist_ok=True)
name = f"grasp_{res['mode']}_{args.variant}"
json.dump(dict(res, log=log), open(os.path.join(HERE, "checks", f"{name}.json"), "w"), indent=1)
if frames:
    import imageio.v2 as imageio
    imageio.mimwrite(os.path.join(HERE, "checks", f"{name}.mp4"), frames, fps=round(1 / (3 * dt)), quality=8,
                     macro_block_size=8)
