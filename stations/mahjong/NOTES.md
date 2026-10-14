# Mahjong station

Fourteen riichi tiles lie scattered on the table between the robot and a slotted wooden tile rack. Some are face
down, all are turned at random, and two pairs are stacked. The wheeled bimanual YAM (`yam`, `linear_4310` grippers)
has to stand every tile in the rack, facing itself, sorted into the hand.

Code: `spec.py` (geometry, task constants), `make_assets.py` (textures, meshes, URDFs), `station.py` (`build()`,
`MahjongEnv`, `success()`), `demo.py` (scripted IK/PD controller), `run.py` (runs, checks, export, previews).
Outputs: `out/mahjong/` (`meta.json`, `traj.npz`, `meshes/`, `checks.json`, `preview.mp4`, `stills/`).

```bash
cd ~/projects/yam-arcade/stations/mahjong
~/projects/yam-arcade/.venvs/mahjong/bin/python make_assets.py      # regenerate assets (PIL, trimesh, manifold3d)
~/simquarry/genesis-world/.venv/bin/python run.py all               # validate + solved + robot demo + export (~5 min)
```

## Task

- **Tiles**: the complete hand 1-2-3 characters, 4-5-6 dots, 7-8-9 bamboo, East ×3, Red dragon ×2 (`spec.HAND`).
- **Start** (`station.scattered_layout`, seed 7): 10 single tiles and 2 stacks of two, 5 face down (4 on the table, 1
  on top of a stack), yaw uniform in ±180°. The tiles sit in x ∈ [-0.40, 0.40], y ∈ [-0.215, -0.02], so every pick
  is a top-down grasp: the robot module reaches top-down only for y < ~0.05.
- **Goal**: every one of the 14 slots holds exactly one tile, and every tile is:
  - seated: back face on the backboard, bottom face on the shelf;
  - upright, with the engraved face towards the robot;
  - at rest.

  Read left to right from the robot (-x to +x), the tiles spell `spec.HAND_SORTED`: characters, dots, bamboo, then
  honours, numbers ascending. Identical tiles (the three Easts, the two Red dragons) are interchangeable.
- **`env.success()`** returns `(ok, detail)`. It works in the rack's current frame, so a bumped rack still counts,
  and requires the rack to stand upright on the table. Per tile it reports slot, `dx_mm` (along the rack),
  `dbu_mm` (profile-plane error from the seated pose), `up_err_deg`, `face_err_deg` and speed. Tolerances:
  - |dx| ≤ 5 mm;
  - profile error ≤ 4 mm;
  - up and face errors ≤ 10°;
  - speed ≤ 5 mm/s.

  Per slot it reports wanted kind vs present kinds. `detail["n_correct"]` counts correct slots (partial progress).

## Objects and parameters

| item | value | why |
|---|---|---|
| tile size | 26 × 35 × 21 mm (w × h × thickness) | riichi tile |
| tile edge radius | 2 mm (visual mesh only) | moulded tile edge |
| tile layers | 13 mm ivory face layer + 8 mm jade-green back | classic two-tone riichi tile |
| tile density / mass | 1450 kg/m³ → 27.7 g | melamine/urea resin is 1.45-1.5 g/cm³ |
| tile collision | exact 26 × 35 × 21 box, `box_box_detection` | flat faces stack and rest exactly; see known issues for the rounded edges |
| tile faces | 34 kinds drawn with PIL, Noto Serif CJK JP Black for 一..九 萬 東南西北 中 發 | real glyphs, standard riichi dot/bamboo layouts and colours, "engraved" groove shading |
| tile UVs | front face, back and two-tone sides in one 1024² atlas per kind | `assets/tiles/contact_sheet.png` shows all 34 faces |
| rack | 524 × 50 × 35 mm, 282 g, density 650 kg/m³ | beech-like hardwood |
| rack geometry | 19 boxes; the visual mesh is the manifold3d union of the same boxes | collision = visual exactly |
| rack tilt | shelf and backboard tilted 15° back | a standing tile leans on the backboard by gravity, face slightly up towards the robot |
| slots | 14 × 36 mm pitch, 3 × 6 mm dividers, 27 mm deep (21 mm + 6 mm slack), backboard 22 mm high | the 10 mm gap between neighbouring tiles leaves room for the 8 mm pads; the 6 mm slack lets a tile be lowered in along the backboard; the pads stay above the 6 mm dividers |
| front lip | none | a lip hid the lower 12 mm of every face from low views; tiles are held by gravity in the shelf/backboard corner |
| rack placement | plinth centre at (0, 0.085), free body resting on the table | not pinned; it may be pushed |
| friction | tile 0.35, table 0.35, rack 0.40, gripper pads 1.0 (robot module) | melamine on melamine/lacquer, oiled wood, rubber pads; Genesis takes the max of the pair |
| grip | 6 N per finger, closing at 5 cm/s | holds 28 g with a margin of ~40 |
| table | fixed box, top at z = 0.75, 1.20 × 0.70 × 0.03 m | layout.TABLE |

Because 15° is below the tile/rack friction angle (atan 0.40 = 21.8°), a tile does not slide into the corner by
itself; it has to be put there.

### Solver settings (`station.py`)

- Rigid solver on `gs.cpu`, float64, dt 2 ms, 1 substep.
- Newton constraint solver, 100 iterations.
- **Elliptic friction cone, impratio 1000** (Signorini contact resolution).
- `constraint_timeconst` 5 ms (≥ 2 dt).
- `box_box_detection`, torsional friction on, no no-slip pass.

How these were chosen (`scratch/sweep.py`, `scratch/test_slip.py`; 6 s runs, settle 1 s, then measure):

| cone | impratio / no-slip | case | drift 1→6 s | max speed, last 1 s | max penetration |
|---|---|---|---|---|---|
| pyramidal | 1 / 0 | scattered | 2e-5 mm | 5e-6 mm/s | 0.0007 mm |
| pyramidal | 1 / 0 | solved (14 tiles in the rack) | 3e-7 mm | 3e-11 mm/s | 0.0010 mm |
| pyramidal | 1 / 0 | tile held at 6 N / pad | **creeps down the pads at 0.65 mm/s** (also at 15 N) | | |
| pyramidal | 1 / 5 sweeps | solved | 1.8e-3 mm | **0.26 mm/s spikes** (end-slot tiles buzz) | 0.0013 mm |
| elliptic | 100 / 0 | solved | **0.011 mm** (creep) | 2e-3 mm/s | 0.017 mm |
| elliptic | 300 / 0 | solved | 0.0036 mm | 7e-4 mm/s | 0.017 mm |
| **elliptic** | **1000 / 0** | solved | 1.1e-3 mm | 2e-4 mm/s | 0.017 mm |
| **elliptic** | **1000 / 0** | scattered | 4e-7 mm | 9e-8 mm/s | 0.015 mm |
| **elliptic** | **1000 / 0** | tile held at 6 N / pad | settles 0.5 mm as gravity turns relative to the grip, then **0 creep** | | |

Why this combination:

- **The pyramidal cone lets a gripped tile slide.** A tile held at 6 N per pad with μ 1.0 crept down the pads at
  0.65 mm/s, about 3 mm per placement, even with the arm still. That is the compliance of MuJoCo-style soft
  friction: 12 N of friction capacity against 0.27 N of weight should not slip, and a stronger grip did not help.
- **No-slip fixes the grip but makes resting tiles buzz.** Genesis's no-slip post-solve stops the creep, but it
  gives the end-slot tiles velocity spikes of 0.26 mm/s (µm amplitude).
- **impratio 1000 removes the elliptic cone's creep.** Creep under the elliptic cone falls as 1/impratio; at 1000 it
  is 0.2 µm/s in the rack (1 µm in 5 s) and zero in the grip. That costs 0.015 mm of resting penetration instead of
  0.001 mm, which is acceptable.
- **The 5 ms time constant** keeps resting penetration in the µm range; 10 ms quadruples it.
- **float32 hides creep.** At z ≈ 0.76 m a step of v·dt below half an ulp rounds away, so anything slower than
  ~30 µm/s is invisible. The float32 runs reported exactly 0.000 drift; float64 shows the real numbers above.

## Results (`out/mahjong/checks.json`)

| check | value | limit | ok |
|---|---|---|---|
| `settle_drift_1s_to_6s_mm` | 4.3e-07 | <= 0.05 | yes |
| `settle_rotation_1s_to_6s_deg` | 1.7e-06 | <= 0.05 | yes |
| `settle_max_speed_last_1s_mm_s` | 8.6e-08 | <= 0.1 | yes |
| `settle_total_motion_mm` | 0.2359 | <= 0.3 (0.1 mm spawn gap, 0.2 for stacks) | yes |
| `settle_solver_penetration_first_1s_mm` | 0.02624 | <= 0.1 (landing from the spawn gap) | yes |
| `settle_rest_height_error_mm` | 0.0345 | <= 0.1 (on table / on the tile below) | yes |
| `settle_solver_penetration_mm` | 0.0148 | <= 0.05 (t >= 1 s) | yes |
| `settle_box_overlap_tile-tile_mm` | 0.0081 | <= 0.05 (independent SAT test on the final poses; < 0 = gap) | yes |
| `settle_box_overlap_tile-table_mm` | 0.0296 | <= 0.05 (independent SAT test on the final poses; < 0 = gap) | yes |
| `settle_box_overlap_tile-rack_mm` | -61.82 | <= 0.05 (independent SAT test on the final poses; < 0 = gap) | yes |
| `settle_box_overlap_rack-table_mm` | 0.0141 | <= 0.05 (independent SAT test on the final poses; < 0 = gap) | yes |
| `scattered_success_is_false` | False | False (nothing in the rack yet) | yes |
| `solved_success` | True | True (all 14 tiles seated in order) | yes |
| `solved_n_correct` | 14 | == 14 | yes |
| `solved_drift_1s_to_6s_mm` | 0.001083 | <= 0.05 (tiles leaning in the rack do not creep) | yes |
| `solved_rack_drift_mm` | 5.0e-10 | <= 0.05 | yes |
| `solved_max_speed_last_1s_mm_s` | 2.2e-04 | <= 0.1 | yes |
| `solved_solver_penetration_mm` | 0.01718 | <= 0.05 (t >= 1 s) | yes |
| `solved_solver_penetration_first_1s_mm` | 0.05163 | <= 0.1 (landing from the spawn gap) | yes |
| `solved_box_overlap_tile-tile_mm` | -10 | <= 0.05 | yes |
| `solved_box_overlap_tile-table_mm` | -12.96 | <= 0.05 | yes |
| `solved_box_overlap_tile-rack_mm` | 0.0206 | <= 0.05 | yes |
| `solved_box_overlap_rack-table_mm` | 0.0344 | <= 0.05 | yes |
| `demo_tile_07_2m_in_slot_1` | True | seated (<= 5 mm along, <= 4 mm in profile), upright & facing robot (<= 10 deg), at rest, right slot | yes |
| `demo_tile_11_6p_in_slot_5` | True | seated (<= 5 mm along, <= 4 mm in profile), upright & facing robot (<= 10 deg), at rest, right slot | yes |
| `demo_tile_01_7z_in_slot_12` | True | seated (<= 5 mm along, <= 4 mm in profile), upright & facing robot (<= 10 deg), at rest, right slot | yes |
| `demo_n_correct_slots` | 3 | == 3 | yes |
| `demo_in_hand_slip_mm` | 0.5828 | <= 5 (tile pose in the gripper, lift -> 8 mm above the seat) | yes |
| `demo_untouched_tiles_moved_mm` | 0.01371 | <= 0.3 (spawn drop only) | yes |
| `demo_rack_moved_mm` | 0.004028 | <= 1.0 | yes |
| `demo_object_penetration_mm` | 0.08926 | <= 0.1 (tile/rack/table contacts, every step) | yes |
| `demo_finger_penetration_mm` | 0.3046 | <= 1.0 (pads squeezing at 6 N, every step) | yes |
| `demo_robot_touches_only_its_tiles` | {} | {} (no robot contact with table, rack, other tiles) | yes |
| `demo_end_box_overlap_tile-tile_mm` | 0.0076 | <= 0.05 (< 0 = gap) | yes |
| `demo_end_box_overlap_tile-table_mm` | 0.0287 | <= 0.05 (< 0 = gap) | yes |
| `demo_end_box_overlap_tile-rack_mm` | 0.0206 | <= 0.05 (< 0 = gap) | yes |
| `demo_end_box_overlap_rack-table_mm` | 0.0187 | <= 0.05 (< 0 = gap) | yes |
| `export_mesh_frame_error_mm` | 1.1e-13 | <= 0.01 (Genesis visual vertices mapped by the recorded pose vs the exported OBJ bounds, rack + 14 tiles) | yes |

**38 / 38 checks pass.** Demo run: 1709 frames, 57.0 s simulated in 3.0 min wall (CPU physics plus 4 software-GL cameras per frame).

Per demo tile (rack frame, after release):

| tile | start | arms | slot | dx | profile error | up / face error | in-hand slip |
|---|---|---|---|---|---|---|---|
| tile_07_2m | face up | left | 1 | 0.00 mm | 0.11 mm | 0.04° / 0.04° | 0.58 mm |
| tile_11_6p | stacked, face up | right | 5 | 0.01 mm | 0.12 mm | 0.04° / 0.04° | 0.55 mm |
| tile_01_7z | face down | left (stand up) + right (rack) | 12 | 0.01 mm | 0.16 mm | 0.04° / 0.04° | 0.24 mm |

Penetration maxima over every step of the demo: rack-table 0.022 mm, rack-tile 0.089 mm, robot-tile 0.305 mm, table-tile 0.043 mm, tile-tile 0.026 mm.
Robot contact points (summed over all steps) by partner: tile_07_2m 29483, tile_11_6p 29474, tile_01_7z 71875 (no table, rack or other-tile contact).

## Robot demo (`demo.py`, `run.py demo`)

All motion comes from IK targets on `yam_cart` joint PD, with `cart.update()` before every `scene.step()`. Tiles
move only through friction grasps. Nothing is teleported, welded or pushed by an applied force. The controller
uses the tile poses from the simulator as its perception.

### Face-up tile into its slot (`stand_in_rack`)

1. Grip from above, across the tile's width. The fingertip ends sit 4 mm above the table and the pads are centred
   6 mm towards the tile's top edge.
2. Lift 10 cm.
3. Measure the tile's pose in the gripper.
4. Move along a Cartesian line (24 IK waypoints, min-jerk timing) to 5 cm above the slot. The gripper pitches 75°
   about the finger axis, which turns the face from up to towards the robot.
5. Lower along the backboard to 8 mm above the seat, staying 3 mm off the backboard.
6. Measure the tile and correct the residual error, which comes from PD tracking lag of ~1-3 mm.
7. Seat the tile, open the fingers slowly and retreat.

A first version that lowered the tile in one move landed it on the backboard's top edge: the PD lag put it 3 mm too
far back. The two-stage descent with feedback fixed that.

### Face-down tile (`flip_and_rack`)

A face-down tile cannot be racked with one grasp: the gripper would have to approach from behind the backboard. It
is regrasped instead:

1. **Arm A** grips it across its width from above. The approach goes into its back, tilted 20° towards its bottom
   edge, so the palm clears the table later.
2. A lifts it, turns it upright and sets it down standing on the table with its face towards arm B. The fingers
   use the reversed finger-axis assignment: the wrist roll is limited to ±120°, and only that sign solves at both
   the pick and the stand pose.

   The set-down stops 8 mm above the table, corrects the measured tile error, then releases from 0.3 mm. Releasing
   from 1 mm let the tile land on an edge with a 0.1 mm contact transient.
3. A releases the tile and goes home.
4. **Arm B** grips the standing tile from its face side, 30° below horizontal, and racks it at a 45° pitch with
   the same feedback placement as above.

### Demo plan

| order | arm | tile | start | slot |
|---|---|---|---|---|
| 1 | left | 2 characters (`tile_07_2m`) | face up, heading -43° | 1 |
| 2 | right | 6 dots (`tile_11_6p`) | face up, on top of a stack | 5 |
| 3 | left stands it up, right racks it | Red dragon 中 (`tile_01_7z`) | face down, far left | 12 |

Results are in the table above and `checks.json["info"]["demo"]`: per-tile seat errors, in-hand slip, robot
contacts, and how far every other tile moved.

## Export and previews

- `out/mahjong/meta.json` + `traj.npz`, written by `export.Recorder` at 30 fps from t = 0:
  - 14 tiles, each `kind: plastic` with its own OBJ+MTL+PNG (`meshes/tile_<kind>.*`; identical kinds share a file);
  - the rack (`kind: wood`, `rack.obj` + `rack_wood.png`);
  - every robot link via `cart.register`.

  `meta.json` also lists the task, the sorted hand, each tile's kind and target slots, the demo plan and the event
  log.
- `export_mesh_frame_error_mm` checks that the exported mesh frame is the frame whose pose is recorded. Genesis
  re-anchors free bodies at their COM internally; `get_pos/get_quat` report the URDF frame, and the check proves it.
- `preview.mp4` is a 2×2 grid at 1280×720: overview from the far right corner, top-down, scattered-tiles close-up,
  rack close-up.
- Stills (`out/mahjong/stills/`, 1600×900):
  - `start_*`, `end_*`;
  - `solved_rack_closeup.png`: all 14 tiles seated, from the validation run;
  - `settled_tiles_closeup.png`;
  - `*_in_hand_overview.png`;
  - `tile_01_7z_{turned_upright,standing,regrasp}_tiles.png`: the flip.
- Rendering uses Genesis's rasterizer through Mesa's software EGL, so previews never touch the shared GPU.

## Known issues / limits

- **Rounded edges vs collision box.** Collision is the exact 26 × 35 × 21 box; the visual tile has 2 mm rounded
  edges. Contacts on an edge or corner (a tile tipping over, an edge touching the rack) therefore happen up to
  0.6 mm outside the visible surface. Face contacts, which is all that happens in the settle, rack and demo runs,
  are exact.
- **Only 3 of 14 tiles are racked.** The demo is a feasibility demo; a full solve is not scripted. The pieces
  exist:
  - face-up tiles whose glyph points within about ±110° of "upright for the robot" go in directly (wrist roll is
    ±120°);
  - the others, and all face-down tiles, go through the stand-up regrasp;
  - stacked tiles: take the top one first.
- **Adjacent slots are tight.** With 36 mm pitch and 8 mm pads there is about 1 mm per side beside an already
  seated neighbour; the demo uses non-adjacent slots. A full solve would place every other slot first, or grip
  higher on the tile.
- **Release workaround.** `demo.Runner.release` resets `cart._grip[side]["target"]` to the current finger position
  before `set_gripper`. Otherwise the opening ramp starts from the grasp squeeze target, 2 cm past closed, and the
  fingers do not move for ~1.5 s. This touches the robot module's private state, and the robot agent may want to
  fix it in `yam_cart.set_gripper`.
- **The start is spawned, not dropped.** Tiles start 0.1 mm above their supports and drop at t = 0. A free drop
  from the air would need a longer settle and give a different layout every time; the layout is a fixed, seeded
  arrangement instead.
- **SAM 3D reference unused.** `assets/sam3d/mahjong_tile` (one 4-dots tile, visual only) was not used: the
  procedural tiles have exact riichi dimensions and all 34 faces. There is no SAM 3D rack (no photo found).
- **Preview furniture.** The Genesis previews draw the table as a plain box and the floor as a plane; the
  compositor draws the real furniture from `layout.py`.
