"""Snapshot the Genesis cables station's specification (READ-ONLY import of stations/cables/station.py) into
spec_genesis.json, so the Newton version uses exactly the same cables, rest shapes, bins and drop layout, and does not
break if the other agent edits station.py while a Newton run is going.

    cd ~/sims/newton && .venv/bin/python ~/projects/yam-arcade/stations/cables_newton/snapshot_spec.py
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GEN = os.path.join(ROOT, "stations", "cables")
sys.path.insert(0, GEN)
sys.path.insert(0, ROOT)
import station as S  # noqa: E402  (pure numpy/scipy at import time; genesis is only imported inside build())


def main():
    polys = S.initial_polylines()
    cables = []
    for c in S.CABLES:
        P0 = S.rest_local(c)
        cables.append(dict(
            name=c["name"], color=c["color"], what=c["what"], radius=c["radius"], seg=c["seg"], EI=c["EI"],
            rho_line=c["rho_line"], friction=c["friction"], length=float(c["shape"]["length"]),
            pieces=[list(p) for p in c["shape"]["pieces"]],
            rest_local=P0.tolist(), initial=polys[c["name"]].tolist()))
    gaps, table = S.init_clearances(polys)
    src = {f: hashlib.sha1(open(os.path.join(GEN, f), "rb").read()).hexdigest()[:12]
           for f in ("station.py", "cable_model.py", "placement.json")}
    out = dict(snapshot_time=time.strftime("%Y-%m-%d %H:%M:%S"), source_sha1=src,
               colors={k: list(v) for k, v in S.COLORS.items()}, bin_tint={k: list(v) for k, v in S.BIN_TINT.items()},
               cables=cables, placement=S.PLACEMENT, bin=dict(S.BIN, size=list(S.BIN["size"])),
               bins={k: dict(xy=list(v["xy"])) for k, v in S.BINS.items()}, bin_lift=S.BIN_LIFT, genesis_dt=S.DT,
               genesis_init_min_gap=float(min(gaps.values())), genesis_init_min_table_gap=float(min(table.values())))
    json.dump(out, open(os.path.join(HERE, "spec_genesis.json"), "w"), indent=1)
    for c in cables:
        P = np.array(c["rest_local"])
        print(f"{c['name']:13s} {c['color']:6s} L={c['length']:.3f} n={len(P)} seg={np.linalg.norm(np.diff(P, axis=0), axis=1).mean() * 1e3:.2f} mm "
              f"r={c['radius'] * 1e3:.1f} EI={c['EI']:.1e} rho={c['rho_line']:.3f}")
    print("min init gap", out["genesis_init_min_gap"], "min table gap", out["genesis_init_min_table_gap"], src)


if __name__ == "__main__":
    main()
