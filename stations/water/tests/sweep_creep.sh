#!/bin/bash
PY=~/simquarry/genesis-world/.venv/bin/python
run() { tag=$1; shift; systemd-run --user --scope -q -p MemoryMax=16G timeout 2500 $PY robot_dry_test.py --substeps 40 --max-secs 20 --out out_cr_$tag "$@" > cr_$tag.log 2>&1;
  echo "== $tag $*"; python3 -c "
import json; d=json.load(open('out_cr_$tag/log.json'))
for r in d['log'][::30]:
    if r['slip_mm'] is not None: print(round(r['t'],1), 'tilt', round(r['tilt'],1), 'cmd', round(r['tilt_cmd'],1), 'slip_mm', round(r['slip_mm'],2), 'z', round(r['pitcher_z'],3))
"; }
run ns5 --rigid '{"noslip_iterations": 5}'
run ns20 --rigid '{"noslip_iterations": 20}'
