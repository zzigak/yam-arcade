#!/usr/bin/env bash
# Fetch a jars sc job's output: sc_fetch.sh TAG [local_dir]   (default: stations/jars/sc_out/TAG)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
tag=${1:?usage: sc_fetch.sh TAG [local_dir]}
dst=${2:-$HERE/sc_out/$tag}
mkdir -p "$dst"
rsync -a "sc:/viscam/group/ziga/sim/out/yam-jars/$tag/" "$dst/"
echo "$dst"
