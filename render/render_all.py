"""Render every camera for the page, one Blender job at a time, then rebuild the site.

    python3 render/render_all.py                       # overview orbit + one close-up per station with an export
    python3 render/render_all.py --only-done           # only stations with out/<station>/DONE
    python3 render/render_all.py --cams jars,water --samples 64 --res 1920x1080
    python3 render/render_all.py --over                # also <station>:over (over the robot's shoulder)

Frames go to render/frames/<name>/ (PNG + render_stats.json), clips to site/media/<name>.mp4. compose.py itself
waits while SAM 3D runs or the GPU is nearly full, so this can be started and left alone (use nohup).
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import layout as L  # noqa: E402
from snapshot import SOURCES  # noqa: E402  (cables -> out/cables/newton, never the Genesis comparison)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", default=None, help="comma list: overview, orbit, <station>, <station>:over")
    ap.add_argument("--only-done", action="store_true")
    ap.add_argument("--over", action="store_true")
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--frames", default=None, help="a:b[:step], passed to compose.py")
    ap.add_argument("--orbit", type=float, default=20.0)
    ap.add_argument("--extra-args", default="", help="appended to every compose.py call")
    args = ap.parse_args()

    src = {s: os.path.join(ROOT, SOURCES.get(s, os.path.join("out", s))) for s in L.STATIONS}
    have = [s for s in L.STATIONS if os.path.exists(os.path.join(src[s], "meta.json"))]
    done = [s for s in have if os.path.exists(os.path.join(src[s], "DONE"))]
    load = done if args.only_done else have
    cams = args.cams.split(",") if args.cams else (["orbit"] + load + ([f"{s}:over" for s in load] if args.over else []))
    print(f"stations loaded: {load}; cameras: {cams}", flush=True)
    for cam in cams:
        name = "overview" if cam in ("overview", "orbit") else cam.replace(":", "_")
        cmd = ["blender", "-b", "-P", os.path.join(HERE, "compose.py"), "--", "--camera", cam,
               "--res", args.res, "--samples", str(args.samples), "--out", os.path.join(HERE, "frames", name),
               "--encode", os.path.join(ROOT, "site", "media", f"{name}.mp4")]
        if load:
            cmd += ["--stations", ",".join(load)]
        if cam == "orbit":
            cmd += ["--orbit", str(args.orbit)]
        if args.frames:
            cmd += ["--frames", args.frames]
        cmd += args.extra_args.split()
        t0 = time.time()
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd)
        print(f"{cam}: exit {r.returncode} after {(time.time() - t0) / 60:.1f} min", flush=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "site", "build_site.py")], check=True)


if __name__ == "__main__":
    main()
