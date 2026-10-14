# Water station — notes

## Summary

A Brita-style pitcher holding 1.2 L of water and three cups of clearly different size and shape (a tall pint glass,
a mug, a wide low teacup) on the station table. The Big YAM cart's right arm grasps the pitcher by its handle and
pours into each cup in turn, then puts the pitcher back down and lets go. The water is Genesis SPH (WCSPH, 4 mm
particles, 23 437 of them), coupled to the rigid containers through Genesis's legacy coupler. The coupler has three
runtime patches made for this station; they are described below. The cups are the SAM 3D meshes (their own vertex
colours). Collision for them is fitted to their measured cavities. The pitcher is procedural.

## Results

Main run: `out/water/raw_robot`, the robot pour exported to `out/water/`. It runs 49.4 s of simulated time:
1 s settle, grasp, three pours, return, put-down, release, then 1.5 s of final settle.

| cup (brim capacity) | robot run | oracle run (same scene) |
|---|---|---|
| tumbler (536 ml) | **364.6 ml (68 %)** | 362.3 ml (68 %) |
| mug (414 ml) | **278.9 ml (67 %)** | 279.5 ml (68 %) |
| teacup (274 ml) | **186.3 ml (68 %)** | 184.0 ml (67 %) |
| spilled (table + floor) | **0.4 ml** (8 particles) | 0.2 ml |
| left in pitcher | 369.8 ml | 374 ml |

`success()` = every cup ≥ 50 % of its brim capacity and spill < 25 ml: **True** in both runs.

- 50 % means each cup is visibly half full; the controller aims at 72 % and overshoot stays below the rim.
- The 25 ml spill limit is ~2 % of the water, about what a careful person drips.

`out/water/checks.json` → `passed: true`. These checks count (main run + export):

- **Particle integrity:** no NaN; the volume accounting closes every frame.
- **Settle:** 0 particles outside the pitcher. Mean speed 0.0026 m/s, 99th percentile 0.013 m/s. The volume below the
  settled surface, minus the fluid-offset skin, matches the mass volume to 1.1 %.
- **Walls:** **0 particles inside any container wall in any frame**.
- **Cups stable:** cups moved < 0.001 mm and tilted < 0.07°.
- **Task:** the fills and spill above; 0 ml still in the air at the end.
- **Robot:**
  - The pitcher was carried by the robot for 38.2 s; the grasp held (drift 12.2 mm < 23 mm, half the pad
    contact).
  - The pitcher was put down upright (0.0°).
- **Export:**
  - Exported mesh × recorded pose matches Genesis's own visual vertices to 8×10⁻⁵ mm.
  - No reconstructed-surface vertex ended up inside a wall: 0 clamped, ~25k vertices per frame.

The grip creep (12.2 mm against a 5 mm target) is reported as an informational known issue (see below).

Stills (Genesis rasterizer, light-grey background, particles drawn as spheres) are in `out/water/`:

- `still_robot_{settled,pour_tumbler,pour_mug,pour_teacup,final}_{front,side}.png`
- preview videos `preview.mp4` (= `preview_robot.mp4`, from the far side) and `preview_robot_side.mp4`
- the oracle versions `still_oracle_*`, `preview_oracle*.mp4`

## Status

- **Done:**
  - Export (`out/water/`: rigids, 25 robot parts, `water` particles, `water_surface` meshseq, 1482 frames at
    30 fps), `checks.json` (passed), NOTES, `DONE`.
  - Every simulation ran on the sc cluster (`stations/water/sc_run.sh`); raw data under
    `sc:/viscam/group/ziga/sim/out/yam-water/`.
- **Robot:** working end to end with the Big YAM cart (`robots/yam_cart.py`, rigid mode). The legacy coupler
  handles robot ↔ water as well.
- **Open:**
  - The grip creeps (known issue 1 below).
  - The pitcher is procedural; the SAM 3D pitcher would need the same collision fit as the cups, plus a
    re-verified grasp.

## Scene

| object | what | size / capacity (brim) | mass | material (export) |
|---|---|---|---|---|
| pitcher | procedural Brita-style jug. Rounded-rectangle plan (superellipse n = 3.2), V-lip spout at local +x, lid over the back ¾ of the top, D-handle at local -x with a 22 × 16 mm grip bar and a 3.5 cm finger gap | 21 cm tall, 12.4 × 9.4 cm inner, **2.07 L** | 0.42 kg (SAN-like plastic, ρ 1050, 2.5 mm walls) | body+handle `glass` (clear plastic), lid `plastic` white |
| tumbler | SAM 3D `tumbler` (conical pint glass) | 14.0 cm tall, **536 ml** | 0.31 kg (glass, ρ 2500) | `glass` |
| mug | SAM 3D `mug` (dark bistro mug) | 9.5 cm tall, **414 ml** | 0.24 kg (stoneware, ρ 2300) | `ceramic`, own colours |
| teacup | SAM 3D `teacup` (wide, low, bamboo pattern) | 6.4 cm tall, **274 ml** | 0.14 kg (porcelain, ρ 2400) | `ceramic`, own colours |
| water | 23 437 SPH particles, 1.2 L by mass | — | 1.2 kg | `particles` + `water_surface` meshseq |

Layout (station frame, table top z = 0.75): pitcher at (0.25, 0.06), spout toward -x and handle toward +x (toward
the right arm). The cups stand in a column at x = -0.02, y = 0.18 (tumbler) / 0.04 (mug) / -0.10 (teacup), with
their handles pointing away from the pitcher. Everything is inside `layout.WORK_ZONE`.

Capacities are the visual cavity volume up to the rim, computed from the same polar wall table that is used to count
water. SAM 3D's own voxel numbers are 550, 421 and 278 ml; mine come out 2-3 % lower because I drop the ragged top
3 mm of the rim.

## Liquid model — why SPH (WCSPH) and not MPM / PBD / DFSPH

- **SPH (WCSPH) — chosen.** Particle–wall coupling is a per-particle velocity projection against each rigid
  geom's signed distance field. With thin walls this is the only one of Genesis's options that can be made
  leak-free: the walls are cut into small convex pieces, each piece has an exact fine SDF, and particles never have
  to cross a grid cell that straddles a 3 mm wall.
- MPM liquid: coupling happens on the MPM grid. A 3 mm wall needs dx ≤ 1.5 mm over a 1 × 0.6 × 0.5 m pour region,
  which is ~10⁸ cells; at 4-5 mm cells the grid leaks through the walls unless CPIC is used. It is also more viscous
  looking.
- PBD liquid: its rigid coupling projects positions, which is robust, but the incompressibility is iterative and soft,
  and the coupler has no friction or pressure exchange.
- DFSPH (tested): it allows ~10× larger steps, but at those steps particles were pushed into the walls (60-136 inside
  the pitcher wall after 0.6 s). WCSPH at a CFL step had 0.

## Parameters (justification)

| parameter | value | why |
|---|---|---|
| particle size d | 4 mm (rest spacing 3.71 mm, volume 0.8 d³ = 0.0512 ml) | the teacup's half fill is ~2 700 particles and a pour stream is 3-5 particles wide. 3 mm would be 2.4× the particles and cost |
| water | 1200 ml = 23 437 particles | a "filled ~1.2-1.5 L" pitcher. Robot payload: pitcher + water = 1.62 kg |
| density | 1000 kg/m³ | water |
| WCSPH stiffness, exponent | 50 kPa, 7 | Tait EOS. c = √(Bγ/ρ) = 18.7 m/s, ~10× the fastest flow (1.7 m/s), so ≤1 % compression |
| kinematic viscosity μ | 3×10⁻⁴ m²/s | artificial viscosity at the usual WCSPH level (α ≈ 0.02 for h = 8 mm, c = 18.7 m/s). Real water is 10⁻⁶; 4 mm particles cannot resolve that. Without the boundary-density fix the floor was 10⁻³ |
| cohesion γ | 0.005 | Genesis's SPH surface-tension term, half its default. Keeps the stream coherent without clumping |
| sim dt / substeps | 1/300 s / 40 → 8.3×10⁻⁵ s | WCSPH CFL: c·dt/d = 0.39. At 16 substeps (c·dt ≈ d) the water exploded |
| initial packing | cubic lattice at s = 0.928 d, filled bottom-up | Genesis's own sampler puts particles at spacing d, which is 20 % under-dense for its 0.8 d³ particle volume. The column then collapsed 2.5 cm and rebounded |
| wall coupling | coup_friction 1.0 (no-slip), softness 2 mm, restitution 0 | with friction 0 the wall-layer particles skated and boiled (mean 0.12 m/s at rest) |
| fluid offset δ | 2 mm on every water-facing collision face (inner face inward, outer face outward, rim upward) | a resting particle's centre sits about one radius off a surface. Particle spheres and the reconstructed surface then meet the visual walls instead of poking through them |
| container SDF | convex pieces, ≤ 1 mm SDF cells, ≥ 16 cells per axis | see "Containers". Genesis's default minimum of 32 cells per axis made 78 M cells (1.6 GB) for ~1 900 pieces |
| rigid contact | constraint_timeconst 2 ms (default 10), pyramidal cone + noslip_iterations 5 | at the defaults, the 80 N grasp sank ~4 mm into the handle and the pitcher rotated 10-60° in the grip during the first move. With these settings it holds, with a residual creep down the grip bar (known issue 1). noslip 20 was no better. The elliptic cone (impratio 100) cut the creep to 1 mm in a dry rehearsal, but its wet run slipped 7 mm in one go at a fast tilt (see "Robot runs") |
| table / cups friction | 0.5 / 0.5-0.6 | ceramic or glass on a wooden top |

## Containers: collision model

Hollow containers are not convex, and a grid SDF of a single 3 mm shell is fragile. Each container's collision is
therefore built in `geometry.py` from convex pieces:

- **Wall wedges:** a convex hull of each angular slice × height band of the wall.
- **Seam covers:** a narrow extra wedge straddling every seam. Without them, a particle that slips into a seam sees
  the pieces' internal side faces as nearest, gets a sideways normal and stays stuck in the wall. The first oracle
  run showed 4-5 such particles at the tumbler's seams.
- **Base, handle and lid** pieces.

For the SAM 3D cups (`sam3d_fit.py`), the inner wall radius R(φ, z) is measured by casting rays from the cavity axis
against horizontal sections of `collision_hollow.obj` (2° × ~1.5 mm grid, lightly smoothed). The height bands are
chosen so each band's straight chord stays within 0.4 mm of the measured wall. A uniform split put a straight chord
across the rounded inside bottom of the tumbler, which pushed the collision 8 mm into the cavity and made the water
explode out. The base piece is the hull of the shell below the floor; the handle piece is the hull of the shell
outside the body. Visual = SAM 3D `visual_lo.obj`, re-framed so the cavity axis is at the origin and the handle
points to -x (`assets/export/water_<cup>_sam3d.glb`, Y-up glTF with per-vertex colours).

I kept the procedural **pitcher** rather than the SAM 3D one:

- The robot grasp and pour were built and verified on its handle.
- The SAM 3D pitcher's handle has a frayed bump on its lower arm.
- Its visual has a full lid but its collision is open-topped.
- It holds 3.5 L, against the requested 1.5-2 L.

Fitting its collision would follow the same route as the cups.

## Coupling fixes (`coupler_patch.py`, applied at runtime by this station only; Genesis is not edited)

1. **Boundary pressure once per link.** Upstream applies the boundary-pressure push (and its reaction on the body)
   once per *geom*. With 2-5 overlapping convex pieces near a wall, it was over-counted 2-5×: water kicked to 2-6 m/s
   and boiled at rest. Now it is applied once per link, from the nearest piece. The velocity response (removing the
   inward relative velocity) still runs against every nearby piece, as upstream does. That is what keeps particles
   out of concave floor/wall corners, and it is not double counted.
2. **Boundary density.** Genesis SPH has no boundary particles. A wall particle's density is under-estimated, clamped
   to zero pressure, and the layer behind it is driven into the wall. I add the wall's analytic share of the kernel,
   ρ₀Φ(d), where Φ is the mass of the cubic spline beyond a plane at distance d (Φ(0) = ½). With this the resting
   water's mean speed dropped from 0.01-0.02 to 0.002-0.003 m/s, outliers went from 1-3 m/s to < 0.1 m/s, and the
   minimum distance to the wall rose to 2.2 mm.
3. **Performance only (results unchanged):**
   - SDF gradients are evaluated only inside the coupling band.
   - Per-link and per-piece bounding spheres skip far-away pieces.
   - The coupler's per-(particle × geom) normal history is not allocated or copied each substep, since nothing reads
     it (1 GB and two full copies per substep with ~1 900 pieces).

## Pouring (`pour.py`; robot: `robot_pour.py`)

- The pitcher's pose is parametrised by its **spout lip** (the pivot) and a **tilt β** about its local y axis.
- **Lip height** z_lip(β) is computed from the pitcher's own vertices. The pitcher stays ≥ 1.2 cm off the table and
  ≥ 1.2 cm above a cup's rim wherever it overhangs the cup. The lip sits 3 cm above the rim once the tilt allows.
- **Flow-onset tilt** is computed geometrically from the water left in the pitcher: 44° at 1.2 L, 51° at 0.96 L, 56°
  at 0.76 L. The controller tilts quickly to 8° short of it, then at 7°/s (3.5°/s past half the target).
- **Stopping is closed-loop on the simulated fill:** it stops when cup + in-flight + 0.25 s × the measured flow rate
  reaches 72 % of the brim capacity. Then it untilts with an ease-out (fast start, which cuts the stream). *Disclosed:*
  reading the fill from the simulation stands in for a camera or scale a real robot would use.
- **Order:** tumbler → mug → teacup. The fullest pitcher pours into the tallest glass, where its 44° onset leaves the
  lip low. The low teacup comes last, when the pitcher is emptier and its tilt is larger.

### Robot (Big YAM, right arm, `linear_4310` gripper)

- **Grasp:** the fingers close along station x across the vertical grip bar of the D-handle, with the inner finger
  through the loop, like a hand.
  - The approach is perpendicular to the loop but pitched 50° down. The vertical bar then lies along the pad
    diagonal, giving 46 mm of contact instead of the pad's 30 mm height.
  - The pitcher's weight moment about the pour axis (~1.7 N·m) is held by the normal-force couple across that
    contact.
  - **Grip force 80 N per finger.** I2RT's driver ships a 50 N software limiter
    (`get_robot(limit_gripper_force=50)`), but the DM4310 behind it (10 N·m, 14.6 mm effective drive radius) can
    push ~680 N. At 50 N (2.3 N·m capacity) the pitcher rotated in the grip during transfers in the dry rehearsal.
- **Control:**
  - IK every 2nd step from the commanded pitcher pose × the grasp transform, then the arm's PD.
  - The grasp transform is re-measured continuously from the pitcher's pose (object-pose feedback, standing in for
    visual tracking).
  - Slow integral terms remove the payload sag (≤ 2 cm, ≤ 0.1 rad).
  - Transfer moves run 1.5× slower than the oracle's.
- **Rigid-only rehearsal** (`tests/robot_dry_test.py`, 1.2 kg ballast at the water's centre of mass, 40 substeps,
  85° tilts): grasp at 22.0 mm pad gap on the 22 mm bar, three tilt cycles, pitcher put back upright (z 0.750,
  tilt < 1°), released. Mean tracking error 8 mm. At 100° tilts it was marginal (the pitcher pivoted out once), so
  the pour tilt is capped at 100°.
- The grip drift is measured as the TCP's position in the pitcher frame relative to its value at the grasp. An
  earlier version of the test compared against the driver's own continuously re-estimated grasp transform, which
  hid the creep. The rigid-contact row in the parameter table has the numbers.

### Robot runs (wet, on sc)

1. **Capped run** (`sc:.../raw_robot_capped40`): the same physics as the main run, stopped by the 40 s run cap after
   the third pour. All three cups reached 67-68 % and 0.15 ml was spilled. Its first 40 s are identical to the main
   run.
2. **Main run** (`raw_robot`): results above.
   - The flow started about 5-10° later than the geometric onset, because the held pitcher lags its command
     slightly.
   - Pitcher-pose tracking while lifted: position error median 2.1 mm, p95 16.7 mm, max 53 mm (fast transfer
     moves); orientation error median 0.3°, p95 2.3°.
3. **Elliptic-cone run** (`raw_robot_elliptic`, informational in `checks.json`). Its rigid contact used the elliptic
   friction cone to stop the grip creep.
   - The first two pours were fine (363 / 294 ml).
   - During the teacup pour the pitcher slipped in the grip at a fast tilt. Its underside came down onto the
     teacup's rim, tipped the teacup, which knocked over the mug: 345 ml spilled.
   - Two lessons: the lip schedule's 12 mm rim clearance is too thin against ~1-2 cm of tracking error plus grip
     slip, and the exported run keeps the pyramidal cone. If this is re-run, raise `clear_rim` in
     `pour.PourGeometry.lip_height` to ~25 mm.

## Oracle runs (disclosed: the pitcher is driven by a 6-DoF PD mount, 3 slides + 3 hinges, not by the robot)

- **v2** (`raw_oracle`, final geometry and SAM 3D cups): tumbler 362.3 ml, mug 279.5 ml, teacup 184.0 ml (67-68 %),
  spill 0.2 ml. Every check passes, including 0 particles in walls.
- **v1** (`raw_oracle_v1`, first feasibility run, superseded geometry: procedural cups of 355 / 303 / 150 ml, no
  seam covers, no outer offset):
  - tumbler 231 ml, mug 197 ml, teacup 97 ml (64-65 %)
  - spill 13.9 ml (droplets beside the tall glass)
  - 4-6 particles parked in the tumbler's wedge seams and 3 on the pitcher's outer wall. That is why the seam
    covers and the outer offset exist.

## Surface reconstruction

splashsurf (pysplashsurf 0.14, already in the Genesis venv) runs on the particle centres with these settings:

- particle radius r = 0.62 s = 2.30 mm (the equal-volume sphere, (4/3)πr³ = s³)
- smoothing length 2r, marching-cubes cell 0.75r (1.7 mm; ~27k vertices, ~0.46 MB per frame), iso-threshold 0.6
- 25 weighted-Laplacian smoothing iterations

About 0.12 s per frame. The settled pitcher's surface is watertight, 1166 ml against 1200 ml by mass.

A safety pass (`clamp_to_cavities`) moves any vertex the kernel smoothing pushes inside a container wall back onto
the wall. With the final collision geometry it moved **none** in the main export. (The v1 geometry, without the outer
offset, needed up to 185 vertices per frame moved by ≤ 1.8 mm.)

The export is about 0.6 GB of surface meshes plus 0.36 GB of particle trajectories.

## Known issues

1. **Grip creep.** The pitcher slides down the grip bar at ~0.3 mm/s while held: 12.2 mm over 38 s, roughly 1 cm
   of fingertip travel along the handle in the video. A real 80 N rubber grip on a 16 N load would not slip.
   - The cause is Genesis's soft rigid contact under sustained shear, made worse by the water's fluctuating
     forces. The grasp never fails: drift stays below half the pad contact, and the pitcher follows its commanded
     tilt.
   - Tried: noslip 20 (no better); the elliptic cone (much less creep, but it slipped in one go and knocked the
     cups over in its wet run).
   - Not yet tried: a "hook" grip just under the top bar of the handle, so the bar rests on the inner finger and
     bounds the slide geometrically. It needs the approach re-checked against the top bar.
2. Viscosity is ~300× water's (an artificial-viscosity floor at 4 mm resolution). The stream is a little more
  coherent and the sloshing calmer than real water.
3. The stream's impact sprays a few single particles (droplets) that land on the table; this is the spill (0.4 ml).
4. The fill controller reads the simulated cup volume (privileged information).
5. The collision skin (δ = 2 mm) means the water looks 2 mm thinner than the visual cavity; volumes are reported by
  mass.
6. The SAM 3D cups' collision follows the measured cavity to ~0.5 mm (voxel noise of the source shell, chord sagitta).
7. The pitcher is procedural (see above).
8. Run time: ~100-230 s wall per simulated second (single-threaded kernel launches plus the rigid solver's broadphase
   over ~1 900 collision pieces at 40 substeps). The pours ran on the sc cluster (`sc_run.sh`, RTX PRO 6000 and TITAN
   RTX at about the same speed, 2-3 h for the robot pour), because the shared 5090 was at 4 jobs.
9. An earlier version of `export_water.py` indexed `raw["particles"][i]` on the NpzFile every frame. That decompresses
   the whole T × N × 3 array per access; together with the Recorder keeping views (since fixed in `export.py`), it
   grew to 117 GB and the OOM killer took the session. All my scripts now load raw.npz once (`station.load_raw`), and
   heavy jobs run under `systemd-run --user --scope -p MemoryMax=24G`.

## Files

- `station.py`: build(), env (volumes, success, frame_stats)
- `geometry.py`: container model
- `containers.py`: specs (procedural + SAM 3D switch)
- `sam3d_fit.py`
- `coupler_patch.py`
- `pour.py`: controller + oracle driver
- `robot_pour.py`: robot driver
- `run.py`: simulation → `out/water/raw_<mode>/`
- `export_water.py`: splashsurf + Recorder → `out/water/`
- `checks.py`: `out/water/checks.json`, previews, stills
- `finalize.sh`: fetch from sc → export → checks → previews
- `sc_run.sh`: one run on the sc cluster (`robot` | `oracle`)
- `tests/`:
  - `settle_test.py`, `cup_settle_test.py` (leak tests)
  - `robot_dry_test.py`, `reach_scan.py`, `grasp_debug.py`
  - `frame_check.py` (exported mesh × pose vs Genesis's own vertices: < 10⁻⁴ mm)
