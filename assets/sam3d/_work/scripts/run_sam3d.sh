#!/bin/bash
# usage: run_sam3d.sh jobs.json OUTDIR   -- takes the shared GPU lock, waits for a quiet GPU, runs, releases the lock
LOCK=~/projects/yam-arcade/locks/SAM3D_RUNNING
touch $LOCK; trap 'rm -f $LOCK; echo lock released' EXIT
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "$(date +%T) gpu used ${used} MiB"; [ "$used" -le 5120 ] && break
  if [ $i -eq 30 ]; then free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -lt 27648 ] && { echo "GPU never cleared, abort"; exit 1; }; fi
  sleep 30
done
source ~/tools/sam-3d-objects/env.sh
cd ~/projects/yam-arcade/assets/sam3d/_work   # never run from _src
python ~/projects/yam-arcade/assets/sam3d/_work/scripts/run_sam3d.py "$1" "$2"
