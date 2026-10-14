"""Smoke test of the sc Genesis env for yam-arcade. Each check runs in its own process (one gs.init per process):

    rigid       robots/test_rigid.py yam                         cart loads, holds home, gripper travel, IK reach (gs.cpu)
    ipc_grasp   robots/test_grasp.py --variant yam --ipc         cart through libuipc grasps + lifts a 4 cm cube (gs.gpu)
    ipc_rod     genesis-world demos/ipc/ipc_rod_rope.py -s 100   FEM.Rod polylines through the IPC coupler (gs.gpu)
    sph         stations/water/run.py --mode oracle, 1 s settle  the real water scene: SPH + rigid props + 2 cameras
    egl_nvidia  tiny scene rendered through NVIDIA's EGL
    egl_mesa    same through Mesa's software EGL (what CPU-only jobs get)

    tools/run_on_sc.sh tools/sc_smoke.py [--only sph,ipc_grasp]

Writes out/sc_smoke/summary.json and one log per check in out/sc_smoke/. Peefton reference numbers: tools/SC_README.md.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import genesis

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(genesis.__file__)))
OUT = os.path.join(ROOT, "out", "sc_smoke")
PY = sys.executable
EGL_DIR = "/usr/share/glvnd/egl_vendor.d"

RENDER = """
import genesis as gs
gs.init(backend=gs.cpu, logging_level="warning")
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.Box(pos=(0, 0, 0.2), size=(0.2, 0.2, 0.2)))
cam = scene.add_camera(res=(320, 240), pos=(1.0, 1.0, 1.0), lookat=(0, 0, 0.1), fov=40)
scene.build()
scene.step()
img = cam.render()[0]
print("RENDER", img.shape, round(float(img.mean()), 1), round(float(img.std()), 1))
"""

CHECKS = {
    "rigid": ([PY, "robots/test_rigid.py", "yam"], {}),
    "ipc_grasp": ([PY, "robots/test_grasp.py", "--variant", "yam", "--ipc", "--coupling", "10"], {}),
    "ipc_rod": ([PY, os.path.join(REPO, "demos", "ipc", "ipc_rod_rope.py"), "-s", "100"], {}),
    "sph": ([PY, "stations/water/run.py", "--mode", "oracle", "--settle", "1.0", "--max-secs", "1.0",
             "--out", "out/sc_smoke/water_oracle"], {}),
    "egl_nvidia": ([PY, "-c", RENDER], {"__EGL_VENDOR_LIBRARY_FILENAMES": f"{EGL_DIR}/10_nvidia.json"}),
    "egl_mesa": ([PY, "-c", RENDER], {"__EGL_VENDOR_LIBRARY_FILENAMES": f"{EGL_DIR}/50_mesa.json"}),
}
# result files a check leaves behind; read only when written by this run
RESULTS = {"rigid": "robots/checks/rigid_yam.json", "ipc_grasp": "robots/checks/grasp_ipc_yam.json",
           "sph": "out/sc_smoke/water_oracle/events.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(CHECKS), help="comma list of checks")
    ap.add_argument("--timeout", type=float, default=1800.0, help="per check, s")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    summary = dict(host=os.uname().nodename, gpu=gpu, partition=os.environ.get("SLURM_JOB_PARTITION"),
                   egl_vendors=sorted(os.listdir(EGL_DIR)) if os.path.isdir(EGL_DIR) else [], checks={})
    print(json.dumps(summary, indent=1), flush=True)
    for name in args.only.split(","):
        cmd, env = CHECKS[name]
        log = os.path.join(OUT, f"{name}.log")
        t0 = time.time()
        with open(log, "w") as f:
            try:
                rc = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env}, stdout=f, stderr=subprocess.STDOUT,
                                    timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        res = dict(rc=rc, wall_s=round(time.time() - t0, 1))
        tail = open(log, errors="replace").read().splitlines()
        res["tail"] = [l for l in tail if not l.startswith(" ")][-6:]
        rfile = os.path.join(ROOT, RESULTS.get(name, "-"))
        if os.path.exists(rfile) and os.path.getmtime(rfile) >= t0:
            r = json.load(open(rfile))
            res["result"] = {k: v for k, v in r.items() if k not in ("joints", "reach", "log", "events", "cups")}
        summary["checks"][name] = res
        print(f"== {name}: rc={rc} wall={res['wall_s']}s", *res["tail"], sep="\n  ", flush=True)
        json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
