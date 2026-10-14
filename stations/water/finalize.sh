#!/bin/bash
# Fetch the sc runs, export the main run (robot if it succeeded, else the oracle) to out/water/, run all checks,
# write previews.  Heavy steps run in a memory-capped scope (a runaway job is killed alone).
# usage: stations/water/finalize.sh [robot|oracle]
set -e
cd ~/projects/yam-arcade/stations/water
PY=~/simquarry/genesis-world/.venv/bin/python
CAP="systemd-run --user --scope -q -p MemoryMax=24G"
OUT=~/projects/yam-arcade/out/water
SC=/viscam/group/ziga/sim/out/yam-water
MAIN=${1:-robot}
for m in robot oracle; do
  if ssh sc "test -f $SC/raw_$m/raw.npz"; then
    rm -rf "$OUT/raw_$m"; rsync -a "sc:$SC/raw_$m/" "$OUT/raw_$m/"
  fi
done
# clean the previous compositor export (keep raw runs, logs, and the inputs of checks)
rm -rf "$OUT/meshes" "$OUT/meshseq" "$OUT/meta.json" "$OUT/traj.npz" "$OUT/DONE" "$OUT/surface_stats.json"
$CAP $PY export_water.py --raw "$OUT/raw_$MAIN" --out "$OUT"
$CAP $PY tests/frame_check.py > /dev/null 2>&1 || true
ARGS="--raw $OUT/raw_oracle --main $MAIN --extra oracle_v1=$OUT/raw_oracle_v1 --extra robot_elliptic=$OUT/raw_robot_elliptic"
if [ -f "$OUT/raw_robot/raw.npz" ]; then ARGS="$ARGS --raw-robot $OUT/raw_robot"; fi
$CAP $PY checks.py $ARGS
cp "$OUT/preview_$MAIN.mp4" "$OUT/preview.mp4"
