"""Run the pile profiling variants one after another, each in its own process (fresh Genesis / libuipc):
base, al-ipc contact, 1.5x coarser segments, dt 1/60. For a dedicated GPU (sc): tools/run_on_sc.sh this file."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
PROF = os.path.join(HERE, "prof_station.py")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "out", "cables", "tests")
os.makedirs(OUT, exist_ok=True)
variants = [("sc_base", {}, []), ("sc_alipc", {}, ["--contact", "al-ipc"]),
            ("sc_seg15", {"CABLES_SEG_SCALE": "1.5"}, []), ("sc_dt60", {"CABLES_DT": str(1 / 60)}, [])]
only = sys.argv[1:]
for tag, env, args in variants:
    if only and tag not in only:
        continue
    log = os.path.join(OUT, f"profile_{tag}.log")
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, PROF, "--tag", tag, "--settle", "1.5"] + args, stdout=f, stderr=subprocess.STDOUT,
                           env={**os.environ, **env})
    print(tag, "exit", r.returncode, flush=True)
