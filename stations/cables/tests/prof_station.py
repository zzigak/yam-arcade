"""Per-step wall time of the cables station, with libuipc's own timer breakdown.

    CABLES_DT=... CABLES_SEG_SCALE=... gpu_run.sh LOG tests/profile.py --tag NAME [--contact al-ipc] [--robot]

Drops the pile (no camera), times every step; after `--settle` s enables uipc.Timer for `--n` steps of the settled
pile. Writes out/cables/tests/profile_<tag>.json: s/step in the drop (0-1 s) and settled, Newton iterations per
step, top timers, cable vertex count, other GPU processes."""
import argparse, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import station as S
import genesis as gs
import uipc

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--contact", default=None)
ap.add_argument("--robot", action="store_true")
ap.add_argument("--settle", type=float, default=2.0)
ap.add_argument("--n", type=int, default=30)
a = ap.parse_args()


def gpu_jobs():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                             text=True).stdout.split()
        return len(out)
    except Exception:
        return None


gs.init(backend=gs.gpu, logging_level="warning")
uipc.Timer.enable_all()  # must be on before the libuipc engine exists; the report is cumulative over the run
extra = {} if a.contact is None else dict(contact_constitution=a.contact)
env = S.build(robot=a.robot, coupler_extra=extra)
uipc.Logger.set_level(uipc.Logger.Level.Info)  # libuipc prints Newton iteration counts per frame at Info
print("PHASE drop", flush=True)
nverts = sum(c["framed"].n for c in env.cables.values())
ntot = sum(c["framed"].X0.shape[0] for c in env.cables.values())
walls = []
jobs = []
for i in range(int(round(a.settle / S.DT))):
    t0 = time.time()
    env.step()
    walls.append(time.time() - t0)
    if i % 30 == 0:
        jobs.append(gpu_jobs())
        print(f"t={env.t:.2f} wall/step {np.mean(walls[-30:]):.3f}s", flush=True)
print("PHASE settled", flush=True)
tw = []
for i in range(a.n):
    t0 = time.time()
    env.step()
    tw.append(time.time() - t0)
print("PHASE end", flush=True)
uipc.Logger.set_level(uipc.Logger.Level.Warn)
rep = uipc.Timer.report_as_json()
rep = json.loads(rep) if isinstance(rep, str) else rep
n_steps_total = len(walls) + len(tw)


def flatten(node, path="", out=None):
    out = [] if out is None else out
    if isinstance(node, dict):
        name = node.get("name", "")
        p = f"{path}/{name}" if name else path
        if "duration" in node or "count" in node:
            out.append(dict(path=p, duration=node.get("duration"), count=node.get("count")))
        for ch in node.get("children", []) or []:
            flatten(ch, p, out)
    elif isinstance(node, list):
        for ch in node:
            flatten(ch, path, out)
    return out


flat = flatten(rep)
newton = [f for f in flat if "newton" in f["path"].lower().split("/")[-1].lower() and "iteration" in f["path"].lower().split("/")[-1].lower()]
top = sorted([f for f in flat if f["duration"] is not None], key=lambda f: -f["duration"])[:25]
n_drop = int(round(1.0 / S.DT))
res = dict(tag=a.tag, dt=S.DT, seg_scale=S.SEG_SCALE, contact=a.contact or "ipc (default)", robot=a.robot,
           cable_vertices=nverts, total_rod_vertices=ntot,
           s_per_step_drop=float(np.mean(walls[:n_drop])), s_per_step_settling=float(np.mean(walls[n_drop:])),
           s_per_step_settled=float(np.mean(tw)), sim_s_per_wall_s_settled=float(S.DT / np.mean(tw)),
           newton_timer=newton[:3], newton_iters_per_step=(newton[0]["count"] / n_steps_total) if newton else None,
           n_steps_timed=n_steps_total,
           top_timers=top, other_gpu_jobs=jobs, tangle=env.tangle_metric().get("crossings"))
os.makedirs(os.path.join(S.ROOT, "out", "cables", "tests"), exist_ok=True)
json.dump(dict(res=res, timer_raw=rep), open(os.path.join(S.ROOT, "out", "cables", "tests", f"profile_{a.tag}.json"), "w"),
          indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k != "top_timers"}, default=str))
for f in top[:12]:
    print(f"  {f['duration']:.3f}s x{f['count']}  {f['path']}")
