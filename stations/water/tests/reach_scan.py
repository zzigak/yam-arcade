"""Reach scan for the handle grasp: right arm, flange z = approach (horizontal), x = closing axis, then rolled about
the approach by the pour angle.  Prints the IK position/rotation error over a grid.  CPU, robot only.

usage: python reach_scan.py [--approach y|-x]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "robots"))
import genesis as gs  # noqa: E402
import layout  # noqa: E402
from yam_cart import add_cart, quat_from_mat  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--approach", default="y")
args = ap.parse_args()
gs.init(backend=gs.cpu, logging_level="warning")
scene = gs.Scene(show_viewer=False)
cart = add_cart(scene, gs, "big_yam")
scene.build()
cart.after_build()


def frame(approach, close, roll):
    z = np.asarray(approach, float)
    x = np.asarray(close, float)
    y = np.cross(z, x)
    R0 = np.stack([x, y, z], axis=1)
    c, s = np.cos(roll), np.sin(roll)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R0 @ Rz


if args.approach == "y":
    approach, closes = (0, 1, 0), [(1, 0, 0), (-1, 0, 0)]
else:
    approach, closes = (-1, 0, 0), [(0, 1, 0), (0, -1, 0)]
for close in closes:
    print("close", close)
    for z in (0.90, 1.00, 1.10):
        for y in (-0.15, -0.05, 0.05, 0.15, 0.25):
            row = []
            for x in (0.0, 0.1, 0.2, 0.3, 0.4):
                errs = []
                seed = None
                for roll in np.radians([0, 40, 80, 120]):
                    sign = 1
                    R = frame(approach, close, -roll)
                    q, e = cart.ik("right", (x, y, z), quat=quat_from_mat(R), init_qpos=seed, return_error=True)
                    errs.append(np.linalg.norm(e[:3]) * 1e3 + np.linalg.norm(e[3:]) * 100)
                row.append("".join("." if er < 3 else ("o" if er < 15 else "X") for er in errs))
            print(f"z={z:.2f} y={y:+.2f}: " + "  ".join(row))
