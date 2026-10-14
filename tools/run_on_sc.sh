#!/usr/bin/env bash
# Run one yam-arcade script as a 1-GPU batch job on the Stanford sc cluster, with the shared Genesis + libuipc env.
#
#   tools/run_on_sc.sh stations/water/run.py --mode oracle --out out/water/sc_oracle
#   GPU=none tools/run_on_sc.sh stations/mahjong/run.py validate        # gs.cpu script: CPU-only job, Mesa EGL
#   PARTITION=viscam-interactive,sc-loprio tools/run_on_sc.sh tools/sc_smoke.py   # short job: take whichever starts
#   tools/run_on_sc.sh --sync-only                                      # only mirror code + assets to sc
#
# The job runs `$REPO/.venv/bin/python $PROJ/<script> args...` with cwd = $PROJ, so project-relative paths work as on
# peefton. $PROJ/out is a symlink to $OUT_ROOT: whatever a script writes under out/ lands in
# /viscam/group/ziga/sim/out/yam-arcade/ (fetch it with tools/fetch.sh). Full notes: tools/SC_README.md.
#
# Env knobs (defaults in brackets):
#   PARTITION [auto]  auto = viscam-hi while you hold < 8 jobs there (its per-user job cap), else viscam-interactive
#                     (< 2 jobs), else viscam (preemptible by viscam-hi: requeued, restarts from scratch). A comma list
#                     is passed to Slurm as is (the job starts in whichever partition fits first); sc-loprio reaches
#                     other groups' L40S/H200 but is preemptible (requeue), so only for short or restartable runs.
#   GPU []            gres type (rtxpro6000 / l40s / h200); empty = any GPU allowed by CONSTRAINT; none = CPU-only job
#   CONSTRAINT [ada|hopper|blackwell]   libuipc's CUDA backend ships sm_89 code only: Ampere/Turing cannot run it
#   CPUS [2]  MEM [24G]  TIME [4:00:00]  the rtxpro6000 nodes are usually CPU-packed; 2 CPUs fit more often
#   SYNC [1]          0 = submit without mirroring first
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
B=/viscam/group/ziga/sim                                  # big volume on sc (NFS, 16 TB); sc home is only 20 GB
# $B/yam-arcade is the water station's partial mirror (stations/water/sc_run.sh syncs it with --delete-excluded),
# so the full mirror lives beside it; jars keeps its own in $B/yam-arcade-jars.
REPO=$B/genesis-world PROJ=$B/yam-arcade-main OUT_ROOT=$B/out/yam-arcade
PEEF_PROJ=/home/ziga/projects/yam-arcade PEEF_REPO=/home/ziga/simquarry/genesis-world

# Code + assets only. Outputs, previews, venvs, scratch, logs and the SAM 3D work dir stay on peefton.
EXCL=(--exclude=/out --exclude=/site/ --exclude=/render/frames/ --exclude=/locks/ --exclude=.venvs/ --exclude=.git/
      --exclude=scratch/ --exclude=_scratch/ --exclude=__pycache__/ --exclude=/assets/sam3d/_work/
      --exclude='/stations/*/tests/out_*/' --exclude=/stations/mahjong/logs/ --exclude='*.log' --exclude='*.pid')

sync_project() {
  local tmp
  tmp=$(mktemp -d)
  # Generated files bake in peefton paths (robot MJCF meshdirs, water-prop MJCFs, plug JSONs, ...). They go to
  # .sc_orig/ verbatim and sc gets rewritten copies swapped in with mv, so a running job never reads a peefton path.
  (cd "$HERE" && rsync -a --list-only "${EXCL[@]}" ./ | awk '/^-/{print $5}' | tr '\n' '\0' |
    xargs -0 grep -lI -e "$PEEF_PROJ" -e "$PEEF_REPO" || true) > "$tmp/abs"
  sed 's#^#/#' "$tmp/abs" > "$tmp/abs_excl"
  ssh sc "mkdir -p $PROJ/.sc_orig $OUT_ROOT $B/logs && { [ -e $PROJ/out ] || ln -s $OUT_ROOT $PROJ/out; }"
  # exit 24 = a file vanished mid-transfer (another agent editing): harmless, the next sync picks it up
  rsync -a "${EXCL[@]}" --exclude-from="$tmp/abs_excl" "$HERE/" sc:$PROJ/ || [[ $? == 24 ]]
  # out/'s directory tree only (no files): scripts that open out/<station>/<sub>/x without makedirs work as on peefton
  rsync -a --include='*/' --exclude='*' "$HERE/out/" sc:$OUT_ROOT/
  rsync -a --files-from="$tmp/abs" "$HERE/" sc:$PROJ/.sc_orig/
  rsync -a "$tmp/abs" sc:$PROJ/.sc_orig/.list
  ssh sc bash -s <<EOF
set -e
cd $PROJ
while IFS= read -r f; do
  mkdir -p "\$(dirname "\$f")"
  sed -e 's#$PEEF_PROJ#$PROJ#g' -e 's#$PEEF_REPO#$REPO#g' ".sc_orig/\$f" > "\$f.sctmp" && mv -f "\$f.sctmp" "\$f"
done < .sc_orig/.list
EOF
  echo "synced $HERE -> sc:$PROJ ($(wc -l < "$tmp/abs") files path-rewritten)"
  rm -rf "$tmp"
}

script=${1:?usage: run_on_sc.sh <script.py relative to the project root> [args...] | --sync-only}; shift
[[ ${SYNC:-1} == 0 ]] || sync_project
[[ $script == --sync-only ]] && exit 0
stem=$(basename "$script" .py)

if [[ ${GPU:-} == none ]]; then
  gres="" egl=/usr/share/glvnd/egl_vendor.d/50_mesa.json
else
  gres="#SBATCH --gres=gpu:${GPU:+$GPU:}1
#SBATCH --constraint=${CONSTRAINT:-ada|hopper|blackwell}" egl=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi
cmd=$(printf '%q ' "$REPO/.venv/bin/python" "$PROJ/$script" "$@")
job=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=ya-$stem
#SBATCH --account=viscam
$gres
#SBATCH --cpus-per-task=${CPUS:-2}
#SBATCH --mem=${MEM:-24G}
#SBATCH --time=${TIME:-4:00:00}
#SBATCH --output=$B/logs/%j-ya-$stem.log
# Headless rasterizer through EGL: NVIDIA's on GPU jobs, Mesa's software one on CPU jobs (overrides a station's own
# setdefault, e.g. mahjong's Mesa pick for the shared peefton GPU). Kernel caches and the CUDA PTX-JIT cache live on the
# big volume, shared with the other Genesis projects on sc.
export PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=$egl
export XDG_CACHE_HOME=$B/cache CUDA_CACHE_PATH=$B/cache/nv CUDA_CACHE_MAXSIZE=4294967296
echo "\$(hostname) \$SLURM_JOB_PARTITION \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
cd $PROJ && SECONDS=0
$cmd
rc=\$?; echo "exit \$rc after \${SECONDS} s wall"; exit \$rc
EOF
)

# Partition is picked on sc (needs squeue); the job script arrives on sbatch's stdin.
jid=$(ssh sc "p=${PARTITION:-auto}
if [ \$p = auto ]; then
  if [ \$(squeue -h -u \$USER -p viscam-hi | wc -l) -lt 8 ]; then p=viscam-hi
  elif [ \$(squeue -h -u \$USER -p viscam-interactive | wc -l) -lt 2 ]; then p=viscam-interactive
  else p=viscam; fi
fi
echo \$(sbatch --parsable -p \$p) \$p" <<<"$job")
[[ ${jid% *} =~ ^[0-9]+$ ]] || { echo "sbatch failed: $jid" >&2; exit 1; }
echo "job ${jid% *} on ${jid#* }"
echo "log:   ssh sc tail -f $B/logs/${jid% *}-ya-$stem.log"
echo "out:   sc:$OUT_ROOT/...  (the script's out/ paths)"
echo "fetch: tools/fetch.sh <path under out/>      e.g. tools/fetch.sh water/sc_oracle"
