"""CPU: plan the plug pick-and-drop joint paths (IK only) for synthetic plug positions, both arms."""
import os, sys, types, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import station as S
import genesis as gs
from yam_cart import add_cart
gs.init(backend=gs.cpu, logging_level="warning")
import run as R
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=S.DT), show_viewer=False)
cart = add_cart(scene, gs, "yam_ultra", ipc=False); scene.build(); cart.after_build()
env = types.SimpleNamespace(robot=cart)
for cable, (x, y), yaw in (("ext_coil", (-0.10, 0.10), 0.4), ("ext_coil", (-0.10, 0.10), 1.4), ("ext_coil", (-0.10, 0.10), -0.6),
                           ("barrel_arc", (0.20, 0.05), 0.0), ("power_coil", (-0.03, -0.165), 0.5), ("cat6_scurve", (-0.21, -0.22), 1.0)):
    side = S.arm_for_color(S.CABLE_BY_NAME[cable]["color"])
    pick = dict(cable=cable, point=[x, y, S.ZT + 0.0005], yaw=yaw, side=side, plug=S.CABLE_BY_NAME[cable]["plug"])
    pl = R.demo_plan_plug(env, pick)
    print(cable, side, (x, y), yaw, "ok" if pl["ok"] else "REJECTED", pl["report"])
