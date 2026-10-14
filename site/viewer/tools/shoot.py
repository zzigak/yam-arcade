"""Headless check of the viewer: load pages in the installed Google Chrome (software GL, no GPU), wait for the
debug hook to report ready, set a time, screenshot, and compare a rigid's pose against traj.npz.

    ~/projects/yam-arcade/.venvs/viewer/bin/python site/viewer/tools/shoot.py --base http://127.0.0.1:8897/viewer/ \
        --out site/viewer/tools/shots [--only room,mahjong] [--mobile]
"""
import argparse
import json
import os
import sys
import time

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8897/viewer/")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots"))
ap.add_argument("--only", default=None)
ap.add_argument("--mobile", action="store_true")
ap.add_argument("--size", default="1400x860")
ap.add_argument("--fracs", default="0.15,0.6")
ap.add_argument("--data", default=None, help="data dir matching --base (default: the one beside this script)")
ap.add_argument("--no-pose", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
W, H = map(int, args.size.split("x"))
data = args.data or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
index = json.load(open(os.path.join(data, "index.json")))
only = set(args.only.split(",")) if args.only else None

jobs = []
if not only or "room" in only:
    jobs.append(("room", "#room=1", ["overview", "mahjong"]))
for r in index["runs"]:
    if only and r["id"] not in only and r["station"] not in only:
        continue
    jobs.append((r["id"], f"#run={r['id']}&play=0", ["close"]))

with sync_playwright() as p:
    # The shell exports VK_DRIVER_FILES=nvidia_icd.json, which overrides ANGLE's SwiftShader ICD and puts WebGL on the
    # RTX 5090 (where headless shadow passes lose the context). Drop VK_* so WebGL really is software (SwiftShader).
    env = {k: v for k, v in os.environ.items() if not k.startswith("VK_")}
    br = p.chromium.launch(channel="chrome", headless=True, env=env,
                           args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    ctx = br.new_context(viewport=dict(width=390, height=844) if args.mobile else dict(width=W, height=H),
                         device_scale_factor=2 if args.mobile else 1, is_mobile=args.mobile, has_touch=args.mobile)
    for name, hsh, cams in jobs:
        pg = ctx.new_page()
        logs = []
        pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        pg.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
        t0 = time.time()
        pg.goto(args.base + hsh)
        try:
            pg.wait_for_function("window.__viewer && window.__viewer.ready === true", timeout=180000)
        except Exception as e:  # noqa: BLE001
            print(name, "NOT READY", e, logs[-10:])
            pg.screenshot(path=os.path.join(args.out, f"{name}_notready.png"))
            pg.close()
            continue
        pg.evaluate("window.__viewer.setPlaying(false)")
        gl = pg.evaluate("(() => { const g = window.__viewer.renderer.getContext(); const d = g.getExtension('WEBGL_debug_renderer_info'); return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : '?'; })()")
        if "SwiftShader" not in gl:
            sys.exit(f"WebGL is not software ({gl}); refusing to run on the GPU")
        dur = pg.evaluate("window.__viewer.S.duration")
        print(f"{name}: ready in {time.time() - t0:.1f}s, duration {dur:.2f}s")
        for cam in cams:
            pg.evaluate(f"window.__viewer.applyPreset('{cam}')")
            for fr in map(float, args.fracs.split(",")):
                pg.evaluate(f"window.__viewer.setTime({fr * dur})")
                pg.wait_for_timeout(400 if "seq" not in name else 1500)
                pg.evaluate("window.__viewer.render()")
                fn = os.path.join(args.out, f"{name}_{cam}_{int(fr * 100):02d}{'_m' if args.mobile else ''}.png")
                pg.screenshot(path=fn)
                print("  ", fn)
        if name != "room" and not args.no_pose:  # pose check: viewer node vs traj.npz, at the frame on screen
            import numpy as np
            man = json.load(open(os.path.join(data, name, "manifest.json")))
            src = os.path.join(os.path.dirname(data), "..", "..", man["source"])
            z = np.load(os.path.join(src, "traj.npz"))
            dyn = [o["name"] for o in man["objects"] if o["kind"] == "rigid" and "traj" in o]
            pick = [n for n in dyn if not n.startswith("robot/")][:1] + [n for n in dyn if n.startswith("robot/")][-1:]
            for nm in pick:
                ps = pg.evaluate(f"window.__viewer.pose({json.dumps(name)}, {json.dumps(nm)})")
                f = ps["src_frame"]
                dp = np.abs(np.array(ps["local"]) - z[f"{nm}/pos"][f]).max()
                q0 = z[f"{nm}/quat"][f] / np.linalg.norm(z[f"{nm}/quat"][f])
                dq = 1 - abs(float(np.dot(np.array(ps["quat_wxyz"]), q0)))
                print(f"   pose {nm} @ frame {f}: |dpos| {dp * 1000:.4f} mm, 1-|q.q| {dq:.2e}, world {np.round(ps['world'], 4).tolist()}")
            mesh_rigids = [o for o in man["objects"] if o["kind"] == "rigid"]
            print(f"   {len(mesh_rigids)} rigids, {sum(1 for o in mesh_rigids if o.get('mesh'))} with meshes")
        errs = [l for l in logs if l.startswith(("error", "pageerror", "warning"))]
        if errs:
            print("   console:", errs[:8])
        pg.close()
    br.close()
