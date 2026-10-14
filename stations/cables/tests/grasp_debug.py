"""Robot pick-and-drop on a single cable lying on the table (fast debug of the demo): per-step wall time and gripper
state. usage: grasp_debug.py CABLE [--krb-only] [--settle 1.0]"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import station as S
import run as R
import genesis as gs
ap = argparse.ArgumentParser()
ap.add_argument("cable", default="barrel_arc", nargs="?")
ap.add_argument("--krb-only", action="store_true")
ap.add_argument("--settle", type=float, default=0.6)
ap.add_argument("--xy", type=float, nargs=2, default=(0.12, -0.05))
ap.add_argument("--force", type=float, default=6.0)
ap.add_argument("--coupling", type=float, default=100.0)
ap.add_argument("--pinch", action="store_true")
ap.add_argument("--coupling-rot", type=float, default=1000.0)
ap.add_argument("--tag", default="")
ap.add_argument("--extra", default="{}")
a = ap.parse_args()
gs.init(backend=gs.gpu, logging_level="warning")
pl = {c["name"]: dict(xy=(5.0, 5.0), yaw=0.0, dz=0.1) for c in S.CABLES}
pl[a.cable] = dict(xy=tuple(a.xy), yaw=20.0, dz=0.002)
env = S.build(robot=True, placement=pl, only=[a.cable], krb_only=a.krb_only, coupling=a.coupling, coupling_rot=a.coupling_rot,
              coupler_extra=json.loads(a.extra))
cart = env.robot
t0 = time.time()
for _ in range(int(a.settle / S.DT)):
    env.step()
print("settle wall", round(time.time() - t0, 1), flush=True)
pts = env.cable_points()
cands = R.plan_pick(env, pts) if a.pinch else R.plan_pick_plugs(env, pts)
print("candidates", cands, flush=True)
pick = cands[0]
plan = R.demo_plan(env, pick) if a.pinch else R.demo_plan_plug(env, pick)
print("pick", pick, plan["report"], plan["ok"], flush=True)
side = pick["side"]
log = []
def run_path(path, phase, extra=0):
    for qk in list(path) + [None] * extra:
        if qk is not None:
            cart.set_arm_target(side, qk)
        ts = time.time()
        env.step()
        dtw = time.time() - ts
        P = env.cable_points()[a.cable]
        tcp = cart.tcp_pose(side)[0]
        log.append(dict(t=round(env.t, 4), phase=phase, wall=dtw, tcp=tcp.tolist(), width=float(cart.gripper_width(side)),
                        force=cart.finger_force(side).tolist(), gv=P[pick.get("vertex", 0)].tolist(), zmin=float(P[:, 2].min()),
                        grasped=bool(cart.grasped(side))))
        if len(log) % 15 == 0:
            r = log[-1]
            print(f"t={r['t']:.2f} {phase:9s} wall/step {np.mean([x['wall'] for x in log[-15:]]):.2f}s tcp {np.round(tcp, 4)} "
                  f"w {r['width']*1e3:.1f}mm f {np.round(r['force'], 2)} gv {np.round(r['gv'], 4)} grasped {r['grasped']}", flush=True)
cart.set_gripper(side, 0.045 if a.pinch else min(0.09, pick["width"] + 0.03))
run_path(plan["approach"], "approach", 20)
if a.pinch:
    R.grip_close(cart, side, S.CABLE_BY_NAME[a.cable]["radius"])
else:
    cart.grasp(side, force=R.PLUG_GRIP_FORCE)
run_path([], "close", int(1.2 / S.DT))
run_path(plan["lift"], "lift", int(0.5 / S.DT))
cart.open(side)
run_path([], "release", int(1.0 / S.DT))
run_path(plan["retreat"], "retreat", 30)
sc = env.success()
print("success", json.dumps(sc["per_cable"]), flush=True)
json.dump(dict(log=log, pick=pick, success=sc), open(os.path.join(S.ROOT, "out", "cables", "tests", f"grasp_debug_{a.cable}{'_krb' if a.krb_only else ''}{a.tag}.json"), "w"))
