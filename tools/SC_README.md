# Running yam-arcade stations on the sc cluster

For when peefton's RTX 5090 is saturated. Ziga's standing permission (2026-10-13): move sim jobs to sc when the GPU
gets slow. Everything runs on the Genesis + libuipc env that franka-deformables built on sc; this project only adds a
code mirror, a launcher and a fetch script. Cluster basics (account, partitions, caps, GPU finder):
`~/.claude/projects/-home-ziga/memory/sc-cluster-gpu-access.md`.

## Where things live on sc

| what | path |
| --- | --- |
| big volume (`$B`) | `/viscam/group/ziga/sim` (NFS, 16 TB free; `/sailhome/ziga` has ~10 GB, keep everything off it) |
| Genesis repo + venv (shared, do not modify) | `$B/genesis-world`, `$B/genesis-world/.venv` (Genesis 1.3.1 `ipc_rod_and_mpm` @ c4e7dd9a, torch 2.11.0+cu128, pyuipc 0.0.26) |
| this project's mirror | `$B/yam-arcade-main` (code + assets, 0.5 GB) |
| outputs | `$B/out/yam-arcade/...`; `$B/yam-arcade-main/out` is a symlink there, so a script's `out/<station>/...` lands in it |
| job logs | `$B/logs/<jobid>-ya-<script>.log` |
| kernel / CUDA JIT caches (shared) | `$B/cache` (`XDG_CACHE_HOME`), `$B/cache/nv` (`CUDA_CACHE_PATH`) |

Other mirrors on the same volume, not used by these tools: `$B/yam-arcade` (water's `stations/water/sc_run.sh`, synced
with `--delete-excluded`, which is why this mirror sits beside it), `$B/yam-arcade-jars` (`stations/jars/sc_run.sh`).

## Run a station script

```bash
cd ~/projects/yam-arcade
tools/run_on_sc.sh stations/water/run.py --mode oracle --out out/water/sc_oracle   # sync + sbatch, prints job id + log
ssh sc tail -f /viscam/group/ziga/sim/logs/<jobid>-ya-run.log
tools/fetch.sh water/sc_oracle                    # -> ~/projects/yam-arcade/out/water/sc_oracle
```

- The launcher mirrors code + assets (no `out/`, `site/`, `render/frames/`, `.venvs/`, `scratch/`, `_scratch/`,
  `__pycache__/`, `locks/`, `.git/`, `assets/sam3d/_work/`, `stations/*/tests/out_*/`, `*.log`, `*.pid`), then submits
  `$B/genesis-world/.venv/bin/python $B/yam-arcade-main/<script> args...` with cwd = the mirror root.
- First sync takes ~4-5 min (the sc login node writes NFS slowly), later ones ~30 s. `tools/run_on_sc.sh --sync-only`
  just mirrors; `SYNC=0` submits without syncing. No `--delete`: files removed on peefton stay in the mirror.
- Give sc runs their own `--out` (e.g. `out/<station>/sc_<tag>`): `fetch.sh` rsyncs into the same path under the local
  `out/` and overwrites same-named files.
- Env knobs: `PARTITION` (default `auto`: viscam-hi if you hold < 8 jobs there, else viscam-interactive if < 2, else
  viscam), `GPU` (gres type, `none` = CPU-only job), `CONSTRAINT` (`ada|hopper|blackwell`), `CPUS` (2), `MEM` (24G),
  `TIME` (4:00:00). A comma list such as `PARTITION=viscam,sc-loprio` lets the job start wherever it fits first.
- A `gs.cpu` station (mahjong) can run with `GPU=none`: no GPU, the preview camera goes through Mesa's software EGL.

## Path rewriting

Generated files bake in peefton paths: robot MJCF `meshdir`s (`assets/robots/yam_cart_*.xml`), the exported-robot
`index.json`s, the water props' MJCFs (`stations/water/assets/*_free.xml`, `pitcher_oracle.xml`), the plug JSONs
(`stations/cables/assets/plugs/*.json`), `codeagent/layout_*.json`. The launcher finds every synced text file that
contains `/home/ziga/projects/yam-arcade` or `/home/ziga/simquarry/genesis-world`, ships it verbatim to
`$B/yam-arcade-main/.sc_orig/` and swaps in a rewritten copy (`$B/yam-arcade-main`, `$B/genesis-world`) with `mv`, so a
running job never reads a peefton path. New generated files are picked up automatically; any other absolute
`/home/ziga/...` path (e.g. `~/tools/sam-3d-objects`) does not exist on sc.

## Which GPUs

- libuipc's CUDA backend carries sm_89 code only: IPC stations (jars, cables, robot grasps) need Ada (l40s), Hopper
  (h200) or Blackwell (rtxpro6000); hence `--constraint=ada|hopper|blackwell` by default. Ampere/Turing (a6000, a40,
  a5000, 3090, titanrtx) cannot run it. SPH-only or rigid-only scripts could use them via `CONSTRAINT=` (untested).
- On 2026-10-13 viscam-hi was full with Ziga's 4dcb array (8-job cap), viscam-interactive held 2 franka jobs (2-job cap)
  and the rtxpro6000 nodes had 0 free CPUs, so new jobs pended for a while. Find free GPUs with
  `ssh sc ~/gpucheck.py rtxpro6000 l40s h200 -c 2 -m 24` (add `--scavenge` for other groups' nodes via `sc-loprio`,
  which are preemptible: a requeued job restarts from scratch).
- Batch jobs start on a backfill pass (every ~5 min), so a few minutes in `PENDING (Priority)` is normal.

## Smoke test

`tools/sc_smoke.py` checks the env end to end, each check in its own process; results in
`out/sc_smoke/summary.json` (+ one log per check):

```bash
PARTITION=viscam,sc-loprio CPUS=4 MEM=32G TIME=1:30:00 tools/run_on_sc.sh tools/sc_smoke.py
tools/fetch.sh sc_smoke
```

First run, 2026-10-13, job 17849833 on viscam13 (`viscam` partition), NVIDIA L40S 46 GB (Ada, driver 580.82), 4 CPUs;
8 min wall for all six, peak RSS 3.5 GB, ~0.9 core busy on average.

| check | sc (L40S) | peefton (RTX 5090, shared) |
| --- | --- | --- |
| `rigid` (cart loads, hold, gripper, IK reach; gs.cpu) | pass, 51 s total, 9.0 s after build; numbers identical | 4.7 s after build |
| `ipc_grasp` (yam cart + libuipc, 4 cm cube) | pass, lift 9.76 cm, slip 1.8 mm, grip 39.1 mm; 11.0 s loop, 104 s total | lift 9.76 cm, slip 1.9 mm; 37 s loop |
| `ipc_rod` (3 ropes + loop over a bar, 100 steps) | pass, 18 FPS, lengths within 0.1 mm of rest; 29 s total | - |
| `sph` (water oracle scene, 23 437 particles, 1 s settle + 2 cameras at 1280x720) | pass, build 106 s (cold cache), 143 s wall per simulated s | build 11 s (warm cache), 95 s per simulated s |
| `egl_nvidia` / `egl_mesa` | both render (identical image means); Mesa prints harmless `/dev/dri` permission warnings | - |

So an L40S is ~1.5x slower than the 5090 on the SPH scene and faster than the busy 5090 on the IPC grasp; the win is
running several stations at once. The rtxpro6000s (Blackwell, the 5090's architecture) were untested: their nodes had
no free CPUs.

## Caveats

- Rendering: `PYOPENGL_PLATFORM=egl` plus NVIDIA's EGL vendor file on GPU jobs (Mesa's on CPU jobs), set by the job
  script. This overrides mahjong's own `setdefault` of Mesa (chosen to keep previews off the shared peefton GPU).
  Nyx (`gs-nyx`) is not installed in the sc venv.
- The sc venv was resolved fresh (`uv.lock` is gitignored in genesis-world, so it was not shipped): the physics stack
  matches peefton (torch 2.11.0+cu128, numpy 2.3.5, quadrants 1.2.0, pyuipc 0.0.26, mujoco 3.10.0, scipy 1.17.1,
  tetgen 0.8.2), but some mesh/utility packages are newer (coacd 1.0.14 vs 1.0.11, trimesh 5.1.1 vs 5.0.0, libigl
  2.6.3 vs 2.6.1, numba 0.68 vs 0.66, pydantic 2.14 vs 2.13). A convex decomposition (coacd) can therefore differ
  slightly from peefton's. Newton / warp are not in it (cables_newton runs only on peefton).
- First run on a fresh cache spends extra minutes in Genesis kernel compiles, convex decomposition and libuipc's PTX
  JIT; later runs reuse `$B/cache`.
- `robots/test_*.py` write `robots/checks/*.json` inside the mirror; the next sync overwrites them with peefton's.
- The genesis-world copy on sc is a `git archive` of c4e7dd9a. If genesis-world moves on, update it the way
  `~/projects/franka-deformables/notes/sc_setup.md` describes (that note owns the env).
