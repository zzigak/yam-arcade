"""Freeze the station exports the fly-through renders from: render/snap/<station>/ <- meta.json, traj.npz, meshes/,
meshseq/, checks.json, DONE, so a station agent rewriting out/ cannot change data between frame chunks.

    python3 render/snapshot.py                       # all four, default sources
    python3 render/snapshot.py water                 # refresh one (e.g. when the full-length pour lands)
    python3 render/snapshot.py water=out/water/sc_full

Writes render/snap/<station>/SNAPSHOT.json: source path, newest source mtime, file count, bytes, snapshot time.
compose.py copies these records into render_stats.json.
"""
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SNAP = os.path.join(HERE, "snap")
SOURCES = {"mahjong": "out/mahjong", "jars": "out/jars", "cables": "out/cables/newton", "water": "out/water"}
ITEMS = ("meta.json", "traj.npz", "meshes", "meshseq", "checks.json", "DONE")


def snap(station, src):
    src = os.path.join(ROOT, src)
    dst = os.path.join(SNAP, station)
    tmp = dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    meta_mtime = os.path.getmtime(os.path.join(src, "meta.json"))
    newest, n, nbytes = 0.0, 0, 0
    for it in ITEMS:
        p = os.path.join(src, it)
        if os.path.isdir(p):
            shutil.copytree(p, os.path.join(tmp, it))
        elif os.path.exists(p):
            shutil.copy2(p, os.path.join(tmp, it))
        else:
            continue
        for dp, _, fs in os.walk(os.path.join(tmp, it)) if os.path.isdir(p) else [(tmp, None, [it])]:
            for f in fs:
                q = os.path.join(dp, f)
                newest = max(newest, os.path.getmtime(q))
                n += 1
                nbytes += os.path.getsize(q)
    # a station must not have rewritten meta.json while we copied
    if abs(os.path.getmtime(os.path.join(src, "meta.json")) - meta_mtime) > 1e-6:
        raise SystemExit(f"{station}: {src}/meta.json changed during the copy, run again")
    m = json.load(open(os.path.join(tmp, "meta.json")))
    rec = dict(station=station, source=os.path.relpath(src, ROOT), source_meta_mtime=meta_mtime,
               source_meta_mtime_iso=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(meta_mtime)),
               newest_file_mtime_iso=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(newest)),
               n_frames=m.get("n_frames"), fps=m.get("fps"), files=n, bytes=nbytes,
               done=os.path.exists(os.path.join(tmp, "DONE")),
               snapshot_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(rec, open(os.path.join(tmp, "SNAPSHOT.json"), "w"), indent=1)
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp, dst)
    print(f"{station}: {rec['source']} ({rec['n_frames']} frames, meta {rec['source_meta_mtime_iso']}, "
          f"{n} files, {nbytes / 1e6:.0f} MB)")


def main():
    args = sys.argv[1:] or list(SOURCES)
    os.makedirs(SNAP, exist_ok=True)
    for a in args:
        st, _, src = a.partition("=")
        snap(st, src or SOURCES[st])


if __name__ == "__main__":
    main()
