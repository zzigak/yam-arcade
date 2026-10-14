#!/usr/bin/env bash
# Run the water station's run.py as a 1-GPU batch job on the Stanford sc cluster (prefer the shared
# ~/projects/yam-arcade/tools/run_on_sc.sh for new runs; this one mirrors into $B/yam-arcade. Adapted from
# ~/projects/franka-deformables/sc/run_on_sc.sh; Genesis + venv already live under /viscam/group/ziga/sim).
#
#   stations/water/sc_run.sh robot            # -> sc:$B/out/yam-water/raw_robot
#   stations/water/sc_run.sh oracle
#   fetch: rsync -a sc:$B/out/yam-water/raw_<mode>/ ~/projects/yam-arcade/out/water/raw_<mode>/
#
# Env: PARTITION [viscam-interactive]  CONSTRAINT [ada|hopper|blackwell]  GPU []  CPUS [2]  MEM [24G]  TIME [6:00:00]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
B=/viscam/group/ziga/sim
REPO=$B/genesis-world PROJ=$B/yam-arcade OUT=$B/out/yam-water
mode=${1:?usage: sc_run.sh robot|oracle [extra run.py args]}; shift

# Mirror only what the water station needs (no outputs, other stations' data, SAM 3D work dirs or test outputs).
ssh sc "mkdir -p $PROJ $OUT $B/logs"
rsync -a \
  --include='/layout.py' --include='/export.py' \
  --include='/robots/' --include='/robots/*.py' \
  --include='/assets/' --include='/assets/robots/***' \
  --include='/assets/sam3d/' --include='/assets/sam3d/INDEX.md' \
  --include='/assets/sam3d/pitcher/***' --include='/assets/sam3d/mug/***' \
  --include='/assets/sam3d/tumbler/***' --include='/assets/sam3d/teacup/***' \
  --include='/stations/' --include='/stations/water/' --include='/stations/water/*.py' \
  --include='/stations/water/*.sh' --include='/stations/water/tests/' --include='/stations/water/tests/*.py' \
  --exclude='*' "$ROOT/" sc:$PROJ/
# The robot MJCFs hard-code peefton's meshdir: rewrite for sc (atomic swap, a running job never sees a half file).
for f in "$ROOT"/assets/robots/yam_cart_*.xml; do
  n=$(basename "$f")
  sed "s#$ROOT#$PROJ#g" "$f" | ssh sc "cat > $PROJ/assets/robots/.$n.tmp && mv $PROJ/assets/robots/.$n.tmp $PROJ/assets/robots/$n"
done

out=$OUT/raw_$mode
cmd=$(printf '%q ' "$REPO/.venv/bin/python" "$PROJ/stations/water/run.py" --mode "$mode" --out "$out" "$@")
job=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=water-$mode
#SBATCH --account=viscam
#SBATCH --gres=gpu:${GPU:+$GPU:}1
#SBATCH --constraint=${CONSTRAINT:-ada|hopper|blackwell}
#SBATCH --cpus-per-task=${CPUS:-2}
#SBATCH --mem=${MEM:-24G}
#SBATCH --time=${TIME:-6:00:00}
#SBATCH --output=$B/logs/%j-water-$mode.log
export PYOPENGL_PLATFORM=egl XDG_CACHE_HOME=$B/cache CUDA_CACHE_PATH=$B/cache/nv CUDA_CACHE_MAXSIZE=4294967296
echo "\$(hostname) \$SLURM_JOB_PARTITION \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
mkdir -p $(printf '%q' "$out") && cd $PROJ/stations/water && SECONDS=0
$cmd
rc=\$?; echo "exit \$rc after \${SECONDS} s wall"; exit \$rc
EOF
)
jid=$(ssh sc "sbatch --parsable -p ${PARTITION:-viscam-interactive}" <<<"$job")
[[ $jid =~ ^[0-9]+$ ]] || { echo "sbatch failed: $jid" >&2; exit 1; }
echo "job $jid on ${PARTITION:-viscam-interactive}"
echo "log: sc:$B/logs/$jid-water-$mode.log"
echo "out: sc:$out"
