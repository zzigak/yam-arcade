#!/bin/bash
# Start a GPU job only when SAM 3D is not running (SPEC.md GPU etiquette): waits for locks/SAM3D_RUNNING to go away,
# then runs the given python script (paths may be relative to the caller's directory) with the Genesis venv, from
# the genesis-world checkout.
#   stations/cables/gpu_run.sh LOG script.py [args...]
LOG="$(realpath -m "$1")"; shift
SCRIPT="$(realpath "$1")"; shift
LOCK=~/projects/yam-arcade/locks/SAM3D_RUNNING
while [ -e "$LOCK" ]; do sleep 20; done
cd ~/simquarry/genesis-world || exit 1
echo "[gpu_run] start $(date +%T)" > "$LOG"
exec .venv/bin/python "$SCRIPT" "$@" >> "$LOG" 2>&1
