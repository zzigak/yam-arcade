# YAM arcade: shared spec

Ziga's request (2026-10-13): a simulated room with several tables. Each table holds a different puzzle/game/task
for a robot. Wheeled bimanual I2RT YAM robots work the tables. Background white / light grey.

| station | task | arm |
|---|---|---|
| `mahjong` | mahjong tiles scattered on the table; put them somewhere (rack / tray), in order | YAM |
| `jars` | glass jars and loose lids; screw each lid onto its jar (real threads) | YAM Pro |
| `cables` | cables of different stiffness and non-straight rest shapes, dropped from the air into a tangled pile; sort them by colour into two boxes | YAM Ultra |
| `water` | a Brita-style pitcher (or urn) of water and three cups of different size and shape; pour into all three | Big YAM |

Cable reference: Ziga's earlier cable work (an internal page), source
`~/projects/franka-deformables/` (`genesis_cable.py`, `run_genesis/run_cable.py`, `site/cable.html`).

## Ground rules

- **Nothing non-physical in the objects.** No teleporting, no kinematic objects pretending to be simulated, no
  hidden pins. Every visual artefact (penetration, jitter, frozen bodies, exploding contacts) is found and fixed,
  not hidden by camera angles. Robot joint control (PD targets, IK) is fine. If a feasibility test drives an object
  directly (e.g. a torque on a lid), say so in NOTES.md and on the page.
- **Real-world scale and plausible materials.** Write the numbers you chose (masses, stiffness, friction, sizes) in
  your station's NOTES.md with a one-line justification each.
- **Verify with numbers, not just pictures.** Each station writes `out/<station>/checks.json`: per-check value, limit,
  pass/fail. Look at your own preview frames too (Read the PNGs).

## Paths

- Project: `~/projects/yam-arcade/` — `layout.py` (table, frames, robot mount, station list), `export.py` (Recorder),
  `assets/` (SAM 3D meshes in `assets/sam3d/`, robot assets in `assets/robots/`, I2RT upstream checkout in
  `assets/i2rt_upstream/`), `robots/` (robot module), `stations/<name>/` (station code + NOTES.md),
  `out/<name>/` (station outputs), `render/` (Blender compositor), `site/` (web page), `notes/`.
- Genesis: `~/simquarry/genesis-world`, run with `~/simquarry/genesis-world/.venv/bin/python` (Genesis 1.3.1, branch
  `ipc_rod_and_mpm`: `FEM.Rod` + `gs.morphs.Polyline` through the IPC coupler, MPM, SPH). Its `CLAUDE.md` describes
  repo conventions. Do NOT `pip install` into that venv (it breaks numpy/setuptools pins needed by torch and gs-nyx);
  make a separate venv under `~/projects/yam-arcade/.venvs/` for extra tools.
- SAM 3D Objects: `~/tools/sam-3d-objects` (`source env.sh`; `python $SAM3D_ROOT/sam3d_infer.py img mask out_prefix`).
- Blender 5.2 (snap): `blender -b -P script.py`.
- Hard-won Genesis/libuipc lessons, READ the relevant ones before building:
  `~/.claude/projects/-home-ziga/memory/genesis-ipc-constraints.md` (IPC coupler rules, rods, clamps, attachments),
  `franka-deformables-genesis-newton.md` (robot + IPC coupling settings that worked), `genesis-nyx-rendering.md`,
  `codimrecon-deformable-tips.md`, `sam3d-objects-install.md`, `claude-agent-genesis-pipeline.md`,
  `libuipc-impact-elasticity.md`, `ppf-to-libuipc-porting.md` (all in that memory directory).

## Frames and layout

`layout.py` is the single source of truth. Station-local frame: origin on the floor under the table-top centre,
+z up, metres; table top surface at z = 0.75, top 1.20 m (x) by 0.70 m (y); the robot cart stands on the -y side
and faces +y. Put task objects inside `WORK_ZONE`. Each station sim includes its own table top as a fixed body at
exactly that height and extent (legs optional in the sim; the compositor draws the furniture).

## GPU etiquette (one RTX 5090, 32 GB, shared by every agent)

- SAM 3D needs ~26 GB. While `~/projects/yam-arcade/locks/SAM3D_RUNNING` exists, do not START new GPU jobs
  (running ones may finish). Check `nvidia-smi --query-gpu=memory.used --format=csv` before big jobs.
- At most one GPU job per agent at a time unless it is tiny. Prefer `gs.cpu` for rigid-only tests.
- Never kill a process you did not start. Run long jobs with `nohup ... > log 2>&1 &` and poll the log.
- Put scratch files under your station/asset directory, not /tmp.

## Readiness markers

- `assets/sam3d/READY` — SAM 3D meshes processed; `assets/sam3d/INDEX.md` lists them with real-world sizes.
- `robots/READY` — robot module usable; `robots/README.md` documents the API.
- `out/<station>/DONE` — station export complete and checks pass (write the file last).

## Export format (what the compositor reads)

`export.py`'s `Recorder` writes `out/<station>/meta.json`, `traj.npz`, `meshes/` and `meshseq/`. Record at
`layout.RENDER_FPS` (30 fps), in the station-local frame. Kinds: `rigid` (mesh + per-frame pose, or a static pose),
`rod` (per-frame polyline + radius), `particles` (per-frame points + radius), `meshseq` (per-frame triangle mesh, e.g.
a reconstructed water surface). Robot links are rigids named `robot/<link>`; `robots/` provides a helper that
registers and records them. The station table top is NOT exported (the compositor draws furniture from layout.py).
Give every object a `material` (kind + sRGB colour) so the compositor can shade it.

## Robot

Wheeled bimanual cart (two arms of the station's YAM variant, I2RT `linear_4310` grippers unless noted) built by the
robot agent in `robots/`. Mount pose comes from `layout.ROBOT_MOUNT`. Until `robots/READY` exists, stations build
and test their objects without the robot and leave a hook for it.
