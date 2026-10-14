"""Rigid-only rehearsal of the robot pour (CPU OK): Big YAM grasps the pitcher handle, the pitcher carries 1.2 kg of
ballast at the water's centre of mass instead of SPH water, and the whole pour program runs with a fixed 100-degree
tilt per cup.  Logs grasp, tracking error and slip.

usage: python robot_dry_test.py [--gpu] [--substeps 10] [--frames]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import station as st  # noqa: E402
import robot_pour  # noqa: E402
from pour import PourController, DEG  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--substeps", type=int, default=10)
ap.add_argument("--frames", action="store_true")
ap.add_argument("--max-secs", type=float, default=40.0)
ap.add_argument("--out", default=os.path.join(HERE, "out_robot_dry"))
ap.add_argument("--rigid", default="{}", help="json dict of RigidOptions overrides")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

cams = dict(front=dict(pos=(0.55, -1.05, 1.28), lookat=(0.06, 0.0, 0.84), fov=38, res=(960, 540)),
            close=dict(pos=(0.75, 0.25, 1.05), lookat=(0.25, -0.05, 0.9), fov=45, res=(960, 540)))
t0 = time.time()
env = st.build(backend="gpu" if args.gpu else "cpu", pitcher_mount="free", water_ml=0, ballast_kg=1.2,
               robot_builder=robot_pour.builder, cameras=cams if args.frames else None, substeps=args.substeps,
               rigid=json.loads(args.rigid))
print(f"build {time.time() - t0:.1f}s pitcher mass {env.infos['pitcher']['mass']:.3f}", flush=True)
driver = robot_pour.RobotDriver(env)
dt = env.scene.sim_options.dt
ctrl = PourController(env, driver, dt=dt, slow=1.5)
print("robot log:", env.robot.log, flush=True)

log = []
rel0 = None
tw = time.time()
k = 0
cup0 = {n: env.pose(n)[0] for n in st.CUP_NAMES}
while not ctrl.done and k * dt < args.max_secs:
    ctrl.step()
    driver.update()
    env.scene.step()
    k += 1
    if k % 30 == 0:
        pos, quat = env.pose("pitcher")
        cmd_pos, cmd_q = ctrl.geo.base_pose(ctrl.lip, ctrl.beta)
        R = st.quat_to_R(quat)
        # actual tilt: angle between pitcher z and world z
        tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
        slip = None
        if driver.T_tcp_in_pitcher is not None:
            tcp, fq = env.robot.cart.tcp_pose("right")
            rel = (tcp - pos) @ R          # TCP in the pitcher frame
            if rel0 is None:
                rel0 = rel                 # at the grasp (the driver re-estimates its own copy, so compare to this)
            slip = float(np.linalg.norm(rel - rel0) * 1e3)
        rec = dict(t=k * dt, pos_err_mm=float(np.linalg.norm(cmd_pos - pos) * 1e3) if ctrl.commanding else None,
                   tilt_cmd=float(np.degrees(ctrl.beta)), tilt=float(tilt), slip_mm=slip,
                   width_mm=env.robot.cart.gripper_width("right") * 1e3,
                   f_finger=env.robot.cart.finger_force("right").tolist(),
                   cups_disp_mm={n: float(np.linalg.norm(env.pose(n)[0] - cup0[n]) * 1e3) for n in st.CUP_NAMES},
                   pitcher_z=float(pos[2]))
        log.append(rec)
        if k % 150 == 0 or (args.frames and k * dt > 32.0 and k % 45 == 0):
            print(f"[{time.time() - tw:6.1f}s] " + json.dumps(rec, default=lambda x: round(x, 2)) + f" ev={ctrl.events[-1:]}",
                  flush=True)
            if args.frames:
                import imageio.v3 as iio
                for cn, cam in env.cams.items():
                    iio.imwrite(os.path.join(args.out, f"{cn}_{k:06d}.png"), cam.render()[0])
el = np.array(driver.err_log) if driver.err_log else np.zeros((1, 2))
print(f"tracking: pitcher pos err mean {el[:, 0].mean() * 1e3:.1f} mm max {el[:, 0].max() * 1e3:.1f} mm; "
      f"IK residual mean {el[:, 1].mean() * 1e3:.2f} mm max {el[:, 1].max() * 1e3:.2f} mm")
json.dump(dict(log=log, events=ctrl.events, robot=env.robot.log), open(os.path.join(args.out, "log.json"), "w"),
          indent=1, default=float)
print("events:", ctrl.events)
print("done", ctrl.done, "wall", time.time() - tw)
