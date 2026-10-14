"""CPU replay of run.plan_feed on a recorded frame (IK on a rigid-only cart), printing why candidates are rejected."""
import json, os, sys, types
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import station as S
import genesis as gs
from yam_cart import add_cart
gs.init(backend=gs.cpu, logging_level="warning")
import run as R
out = os.path.join(S.ROOT, "out", "cables")
tr = np.load(os.path.join(out, "traj.npz"))
L = json.load(open(os.path.join(out, "log.json")))
k = max(i for i, r in enumerate(L["log"]) if r.get("demo_phase") == "retreat")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=S.DT), show_viewer=False)
cart = add_cart(scene, gs, "yam_ultra", ipc=False); scene.build(); cart.after_build()
lay = S.initial_layout()
nin = {n: v["n_inner"] for n, v in lay.items()}
# exported rods start at the boot: pad the moulded-in part back with the boot point (inside the plug anyway)
pts = {c["name"]: np.vstack([np.repeat(tr[f"{c['name']}/pts"][k][:1], nin[c["name"]], 0), tr[f"{c['name']}/pts"][k]]).astype(float)
       for c in S.CABLES}
bins = {col: (np.array(b["pos"]), np.array(b["quat"])) for col, b in L["log"][k]["bins"].items()}
env = types.SimpleNamespace(robot=cart, cable_points=lambda: pts, bin_poses=lambda: bins,
                            cables={n: dict(n_inner=nin[n], framed=types.SimpleNamespace(seg=S.CABLE_BY_NAME[n]["seg"])) for n in pts})
orig = R.plan_cartesian
def traced(cart_, side, keys, dt, q0, ik_every=3):
    path, rep = orig(cart_, side, keys, dt, q0, ik_every)
    print("   plan", [tuple(np.round(kk["pos"], 3)) for kk in keys], {kk: round(v, 3) for kk, v in rep.items()})
    return path, rep
R.plan_cartesian = traced
fp, fpl = R.plan_feed(env, "ext_coil", "left", cart.tcp_pose("left"))
print("result", fp)
