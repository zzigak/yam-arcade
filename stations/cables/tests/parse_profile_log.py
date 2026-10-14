"""Newton iterations / line-search and CCD events per phase from a profile log (libuipc Info output)."""
import re, sys, json
import numpy as np
phase, it = None, {}
ev = {}
for line in open(sys.argv[1], errors="ignore"):
    if line.startswith("PHASE "):
        phase = line.split()[1]
        continue
    m = re.search(r"Newton Iteration Converged with Iteration Count: (\d+)", line)
    if m and phase:
        it.setdefault(phase, []).append(int(m.group(1)))
    for key in ("Newton Iteration Exits with Max", "Line Search Exits with Max", "CCD Filter", "CFL Filter"):
        if key in line and phase:
            ev.setdefault(phase, {}).setdefault(key, 0)
            ev[phase][key] += 1
out = {ph: dict(frames=len(v), newton_mean=float(np.mean(v)), newton_max=int(np.max(v)), events=ev.get(ph, {}))
       for ph, v in it.items()}
print(json.dumps(out))
