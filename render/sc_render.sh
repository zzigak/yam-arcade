#!/usr/bin/env bash
# Render the fly-through, or one per-station clip (CLIP=<station>, render/clips.py), in frame chunks on the sc cluster: Blender 5.2.2 (same build as peefton's snap), Cycles
# OptiX, one GPU per job. Cluster conventions (account, logs, caches) follow tools/run_on_sc.sh / tools/SC_README.md.
#
#   render/sc_render.sh sync                                   # mirror layout.py, compose.py, flypath.py, render/snap
#   render/sc_render.sh submit <tag> <res> <samples> <chunks> [first:last]   # sbatch <chunks> jobs (<= 8)
#   render/sc_render.sh missing <tag> [n_frames]               # frame indices not rendered yet, as a:b ranges
#   render/sc_render.sh resubmit <tag> <res> <samples>         # one job per missing range (<= 8 ranges)
#   render/sc_render.sh fetch <tag>                            # -> render/frames/fly_<tag>/
#   CLIP=mahjong render/sc_render.sh submit mahjong_draft 640x360 16 3     # a per-station clip
#
# Jobs are requeueable (PARTITION default viscam,sc-loprio is preemptible): every frame is written to a temp name and
# renamed, and compose.py --skip-existing skips finished frames, so a requeued job resumes where it stopped.
# The mirror is this renderer's own ($B/yam-arcade-render), never synced with --delete.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
B=/viscam/group/ziga/sim
R=$B/yam-arcade-render
BL=$B/tools/blender-5.2.2-linux-x64/blender
OUT_ROOT=$B/out/yam-arcade/flythrough
if [[ -n ${CLIP:-} ]]; then
  N_FRAMES=$(cd "$HERE" && python3 -c "import sys; sys.path.insert(0,'render'); import clips; print(clips.Clip('$CLIP').N_FRAMES)")
  CAM_ARGS="--clip $CLIP --path $R/render/clips.py"
else
  N_FRAMES=$(cd "$HERE" && python3 -c "import sys; sys.path.insert(0,'render'); import flypath; print(flypath.N_FRAMES)")
  CAM_ARGS="--camera flythrough --path $R/render/flypath.py"
fi

cmd=${1:?usage: sc_render.sh sync|submit|missing|resubmit|fetch ...}; shift || true

sync() {
  ssh sc "mkdir -p $R/render $OUT_ROOT $B/logs"
  rsync -a "$HERE/layout.py" sc:$R/
  rsync -a "$HERE/render/compose.py" "$HERE/render/flypath.py" "$HERE/render/clips.py" "$HERE/render/snapshot.py" \
    sc:$R/render/
  rsync -a "$HERE/render/snap/" sc:$R/render/snap/
  echo "synced -> sc:$R"
}

submit_range() {  # tag res samples a b
  local tag=$1 res=$2 samples=$3 a=$4 b=$5
  local out=$OUT_ROOT/$tag
  local job
  job=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=ya-fly-$tag
#SBATCH --account=viscam
#SBATCH --gres=gpu:${GPU:-l40s}:1
#SBATCH --cpus-per-task=${CPUS:-4}
#SBATCH --mem=${MEM:-24G}
#SBATCH --time=${TIME:-4:00:00}
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=$B/logs/%j-ya-fly-$tag.log
export XDG_CACHE_HOME=$B/cache CUDA_CACHE_PATH=$B/cache/nv CUDA_CACHE_MAXSIZE=4294967296
echo "\$(date) \$(hostname) \$SLURM_JOB_PARTITION restart=\${SLURM_RESTART_COUNT:-0} \$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
mkdir -p $out
cd $R && SECONDS=0
$BL -b -P $R/render/compose.py -- $CAM_ARGS --snap $R/render/snap \\
  --hide water:water --frames $a:$b --res $res --samples $samples --out $out --skip-existing --no-wait \\
  --stats-name render_stats_${a}_${b}.json
rc=\$?; echo "exit \$rc after \${SECONDS} s wall"; exit \$rc
EOF
)
  ssh sc "sbatch --parsable -p ${PARTITION:-viscam,sc-loprio}" <<<"$job"
}

missing() {  # tag [n] -> "a:b" ranges of frames without a PNG
  local tag=$1 n=${2:-$N_FRAMES}
  ssh sc "ls $OUT_ROOT/$tag 2>/dev/null" | python3 -c "
import sys
have = {int(l[:5]) for l in sys.stdin if l[:5].isdigit() and l.strip().endswith('.png')}
n = $n; r = []; a = None
for f in range(n + 1):
    m = f < n and f not in have
    if m and a is None: a = f
    if not m and a is not None: r.append(f'{a}:{f}'); a = None
print(' '.join(r))"
}

case $cmd in
  sync) sync ;;
  submit)
    tag=$1 res=$2 samples=$3 chunks=$4 range=${5:-0:$N_FRAMES}
    (( chunks <= 8 )) || { echo "keep it to <= 8 jobs" >&2; exit 1; }
    a0=${range%:*} b0=${range#*:}
    n=$(( b0 - a0 ))
    for ((i = 0; i < chunks; i++)); do
      a=$(( a0 + n * i / chunks )) b=$(( a0 + n * (i + 1) / chunks ))
      echo "chunk $a:$b -> job $(submit_range "$tag" "$res" "$samples" $a $b)"
    done ;;
  missing) missing "$@" ;;
  resubmit)
    tag=$1 res=$2 samples=$3
    ranges=$(missing "$tag")
    [[ -n $ranges ]] || { echo "nothing missing"; exit 0; }
    set -- $ranges
    (( $# <= 8 )) || { echo "$# missing ranges; merge them by hand (<= 8 jobs)" >&2; exit 1; }
    for r in "$@"; do echo "range $r -> job $(submit_range "$tag" "$res" "$samples" "${r%:*}" "${r#*:}")"; done ;;
  fetch)
    tag=$1
    mkdir -p "$HERE/render/frames/fly_$tag"
    rsync -a --info=progress2 sc:$OUT_ROOT/$tag/ "$HERE/render/frames/fly_$tag/" --exclude='_tmp_*'
    echo "fetched -> render/frames/fly_$tag" ;;
  *) echo "unknown command $cmd" >&2; exit 1 ;;
esac
