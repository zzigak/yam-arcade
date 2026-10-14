#!/usr/bin/env bash
# Run one jars-station job as a 1-GPU batch job on the Stanford sc cluster (adapted from
# ~/projects/franka-deformables/sc/run_on_sc.sh; memory: sc-cluster-gpu-access.md).
#
#   stations/jars/sc_run.sh robot1 run.py robot --jars mason      # TAG, then run.py args; output -> $B/out/yam-jars/TAG
#   stations/jars/sc_run.sh twist  run.py twist
#   stations/jars/sc_fetch.sh TAG [local_dir]                      # rsync the output back (default stations/jars/sc_out/TAG)
#
# The project is mirrored to $B/yam-arcade-jars (our own copy, so other stations' launchers never race with it). The
# robot MJCFs and the exported-robot index hard-code peefton's /home/ziga/projects/yam-arcade; they are rewritten for
# sc in the mirror. Env knobs: PARTITION [viscam-interactive], GPU [], CONSTRAINT [ada|hopper|blackwell], CPUS [2],
# MEM [16G], TIME [4:00:00].
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)                    # ~/projects/yam-arcade
B=/viscam/group/ziga/sim
REPO=$B/genesis-world PROJ=$B/yam-arcade-jars OUT_ROOT=$B/out/yam-jars
tag=${1:?usage: sc_run.sh TAG script.py [args...]}; shift
script=${1:?usage: sc_run.sh TAG script.py [args...]}; shift
out=$OUT_ROOT/$tag

rsync -a --delete --exclude out/ --exclude .venvs/ --exclude assets/sam3d/ --exclude assets/i2rt_upstream/ \
  --exclude site/ --exclude render/ --exclude __pycache__/ --exclude 'stations/*/scratch/' --exclude 'stations/*/sc_out/' \
  "$HERE/" sc:$PROJ/
ssh sc "sed -i 's#/home/ziga/projects/yam-arcade#$PROJ#g' $PROJ/assets/robots/*.xml $PROJ/assets/robots/export/*/index.json"

cmd=$(printf '%q ' "$REPO/.venv/bin/python" "$PROJ/stations/jars/$script" "$@" --out "$out")
job=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=jars-$tag
#SBATCH --account=viscam
#SBATCH --gres=gpu:${GPU:+$GPU:}1
#SBATCH --constraint=${CONSTRAINT:-ada|hopper|blackwell}
#SBATCH --cpus-per-task=${CPUS:-2}
#SBATCH --mem=${MEM:-16G}
#SBATCH --time=${TIME:-4:00:00}
#SBATCH --output=$B/logs/%j-jars-$tag.log
export PYOPENGL_PLATFORM=egl XDG_CACHE_HOME=$B/cache CUDA_CACHE_PATH=$B/cache/nv CUDA_CACHE_MAXSIZE=4294967296
echo "\$(hostname) \$SLURM_JOB_PARTITION \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
mkdir -p $(printf '%q' "$out") && cd $REPO && SECONDS=0
$cmd
rc=\$?; echo "exit \$rc after \${SECONDS} s wall"; exit \$rc
EOF
)
jid=$(ssh sc "sbatch --parsable -p ${PARTITION:-viscam-interactive}" <<<"$job")
[[ $jid =~ ^[0-9]+$ ]] || { echo "sbatch failed: $jid" >&2; exit 1; }
echo "job $jid  log: sc:$B/logs/$jid-jars-$tag.log  out: sc:$out"
