#!/usr/bin/env bash
# Copy a run's outputs back from sc into this project's out/.   tools/fetch.sh <path under out/> [local dir]
#   tools/fetch.sh water/sc_oracle                  # sc:/viscam/group/ziga/sim/out/yam-arcade/water/sc_oracle -> out/water/sc_oracle
#   tools/fetch.sh cables out/sc/cables             # whole station dir into a separate local dir
# rsync -a overwrites same-named local files, so give sc runs their own --out (e.g. out/<station>/sc_<tag>) when the
# station also runs on peefton. An absolute remote path is used as is.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
src=${1:?usage: fetch.sh <path under out/> [local dir]}
src=${src#out/}
dst=${2:-$HERE/out/${src#/viscam/group/ziga/sim/out/yam-arcade/}}
[[ $src == /* ]] || src=/viscam/group/ziga/sim/out/yam-arcade/$src
mkdir -p "$dst"
rsync -a --info=progress2 "sc:${src%/}/" "$dst/"
echo "fetched sc:$src -> $dst"
