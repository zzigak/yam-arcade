# Cables station, Newton version: notes

Same task as `stations/cables/` (Genesis + libuipc): eight cables of two colours, each with its own stiffness and a curved
rest shape, are dropped from the air into a tangled pile between two open bins. The wheeled bimanual YAM Ultra then
sorts cables into the bin of their colour. Ziga asked for a fast Newton version. This one runs the whole drop, settle
and two sorts in about 2.5 minutes of wall time.

Run (GPU job, memory-capped as the orchestrator asked):

```bash
cd ~/sims/newton && nohup systemd-run --user --scope -q -p MemoryMax=24G .venv/bin/python \
    ~/projects/yam-arcade/stations/cables_newton/run.py > ~/projects/yam-arcade/stations/cables_newton/logs/main_run.log 2>&1 &
python3 ~/projects/yam-arcade/stations/cables_newton/check.py      # -> out/cables/newton/checks.json
```

| file | what it does |
|---|---|
| `snapshot_spec.py` → `spec_genesis.json` | Read-only snapshot of the Genesis station's spec: cables, rest polylines, drop layout, bins, colours. Source file hashes are recorded. Re-run it if the Genesis spec changes. |
| `ncable.py` | Cable rods, plugs, table, bins, read-back helpers, metrics (retention and tangle use the Genesis definitions). |
| `nrobot.py` | Loads the YAM Ultra cart MJCF into Newton (kinematic), does MuJoCo IK/FK on the same MJCF, holds pad geometry and export parts. |
| `run.py` | Station runner: drop, settle, grasp planning, sorting, force-servo gripper, per-frame physics log, export, previews. |
| `check.py` | Writes `checks.json`. |
| `calib/cantilever.py` → `calib_bend.json` | Per-cable bend calibration at each substep × iteration budget. |
| `tests/rest_shape_test.py` | Checks that each cable keeps its rest shape alone, plus a straight-rest control. |

The outputs are in `out/cables/newton/`:
- the export: `meta.json`, `traj.npz`, `meshes/`;
- `log.json`, `checks.json`, `timing.json`;
- `preview.mp4`;
- stills: `still_{start,settled,end,top,close,front}.png`, plus grasp close-ups in `stills/`;
- `tests/` holds the rest-shape test and the pile budget runs.

## 1. Cable model

- **Rod.** Each cable is a `ModelBuilder.add_rod` chain: one capsule per segment, with VBD cable joints between them. A cable joint has one stretch slot and one isotropic bend/twist slot.
  - Segment length, radius and line density match the Genesis spec: 9-14 mm segments, 475 nodes in all.
  - Capsule density is set so each segment weighs `rho_line × seg`.
  - The whole station has 477 bodies and 469 cable joints without the robot.
- **Rest shape is honest (verified).** `SolverVBD` measures a cable joint's bend/twist relative to the joint's rest relative rotation. In `rigid_vbd_kernels.compute_kappa`, `q_align = q_rel * inverse(q_rel_rest)`. The rest pose is `model.body_q` when the solver is created.
  - A rod built along the curved rest polyline is therefore stress-free in that shape. No extra rest-angle attributes are needed.
  - Body frames come from parallel transport along the rest curve (`newton.utils.create_parallel_transport_cable_quaternions`), so the rest twist is zero.
- **Rest-shape test** (`tests/rest_shape_test.py`, at 40×3). Each cable is dropped alone, 3 cm onto the floor, for 4 s:
  - retention ranges from 0.980 (aux loop) to 0.998 (HDMI);
  - summed-curvature ratio is 1.000-1.008;
  - creep over the last second is at most 0.12 mm.

  Control: the HDMI U-bend starts in its U but is given a straight rest pose. It springs open: retention 0.12, curvature 0.23, RMSD 144 mm. The Genesis plain-`FEM.Rod` control gave the same 144 mm. So the kept shapes come from the rest curvature, not from table friction.
- **Bending.** The per-joint stiffness is `k = s · EI / l`. The scale `s` is calibrated per cable and per budget:
  - The test is a clamped cantilever against the heavy elastica (`calib/cantilever.py`).
  - Overhang is about 2 bending lengths. Gravity is ramped in over 2 s and held 3 s. A secant iteration on `s` finds `EI_eff/EI = 1`.
  - `s` > 1 is needed because block Gauss-Seidel with few iterations is not converged, and its static stiffness comes out low. Thin, light leads need the most.

  Results at the run budget (40×3):

  | cable | EI (N m²) | Lb (cm) | overhang (cm) | elastica tip drop (cm) | sim drop (cm) | scale s | EI_eff/EI |
  |---|---|---|---|---|---|---|---|
  | power_coil | 4.0e-3 | 16.9 | 33.6 | 21.41 | 21.42 | 1.119 | 1.000 |
  | hdmi_ubend | 2.5e-3 | 15.4 | 31.2 | 20.36 | 20.35 | 1.135 | 1.000 |
  | ext_coil | 1.2e-3 | 13.5 | 26.4 | 16.44 | 16.44 | 1.178 | 1.000 |
  | cat6_scurve | 5.0e-4 | 10.8 | 22.0 | 14.36 | 14.36 | 1.241 | 1.000 |
  | barrel_arc | 1.0e-4 | 7.7 | 15.3 | 9.65 | 9.65 | 1.520 | 1.000 |
  | usbc_wave | 4.0e-5 | 6.1 | 12.6 | 8.42 | 8.42 | 1.697 | 1.000 |
  | micro_scurve | 3.0e-5 | 5.8 | 11.7 | 7.65 | 7.65 | 1.794 | 1.000 |
  | aux_loop | 2.0e-5 | 5.1 | 9.9 | 6.01 | 6.02 | 1.871 | 0.999 |

  Other budgets are in `calib_bend.json`. Two examples:
  - 20×20 needs 1.08-1.42;
  - 10×10 needs 1.26-4.9, and the aux lead could not be fitted at all.
- **Torsion.** The Newton cable joint uses one isotropic bend/twist stiffness, so GJ = EI. Genesis uses GJ = EI/(1+ν) = 0.69 EI.
- **Stretch.** The stretch slots are hard (augmented Lagrangian). The 2e3 N/m penalty only sets conditioning (franka lesson: a soft, stiff stretch leaves Gauss-Seidel unconverged). Measured stretch over every 1/60 s frame of the final run is [0.986, 1.036]. The largest gap between capsule ends at a joint is 0.52 mm.
- **Damping.** Bend/twist damping is 0.02 s × k (franka value). There is no air drag.

## 2. Plugs (Newton only; the Genesis cables have none)

Each plug is a set of rigid shapes on an extra rod body at one end of the cable. A hidden, non-colliding capsule carries the 60 % of the overmould that the cable runs into. The plug's strain-relief joint is 3 × EI.

The plug goes on whichever end has more clearance from the other cables, the plugs already placed, the bins and the table at t = 0. The smallest clearance is micro-USB at 9.8 mm; every other plug is at least 24 mm clear.

| cable | plug | mass | parts (overmould in the cable colour; metal tip) |
|---|---|---|---|
| power_coil | IEC C13 socket | 30 g | boot r 6.5 mm + 38×33×22 mm body |
| hdmi_ubend | HDMI type A | 10 g | boot + 28×21×11 mm body + 11×14×4.5 mm shell |
| ext_coil | Europlug CEE 7/16 | 20 g | boot + 36×35×15 mm body + two 4 mm pins, 19 mm long |
| cat6_scurve | RJ45 + boot | 6 g | boot + clear 22×11.8×13.6 mm plug |
| barrel_arc | DC barrel 5.5×2.1 | 4 g | 9 mm round overmould, 25 mm long + barrel |
| usbc_wave | USB-C | 4 g | 24×12×6.5 mm overmould + 7×8.4×2.5 mm shell |
| micro_scurve | micro-USB | 3 g | 20×11×6 mm overmould + shell |
| aux_loop | 3.5 mm TRS | 4 g | 8 mm round overmould, 23 mm long + 3.5 mm pin |

## 3. Scene, contact, materials

| item | value | why |
|---|---|---|
| table top | static box, 1.20 × 0.70 × 0.03 m, top at z = 0.75; plus a floor plane | layout.py |
| bins | free bodies, 0.30 kg each; five boxes, 0.22 × 0.28 × 0.07 m, 5 mm walls and floor | same solid and placement as the Genesis bins; polypropylene, μ 0.5 |
| contact | soft penalty (`rigid_contact_hard=False`); force = mean ke × overlap; μ mixed by geometric mean | the grip sensor reads the same law |
| ke: cable / plug / pad / table, bin, link | 1e4 / 2e4 / 2e4 / 1e5 N/m | a 1 g segment sinks 1 µm; 6 N on a pad compresses ~0.3 mm |
| μ: cable / plug / pad / table, bin | 0.5 / 0.8 / 1.0 / 0.5 | as in Genesis (cables 0.5); rubber pads |
| contact gap | 4 mm | speculative contacts: a 2 m/s cable moves 0.8 mm per 1/2400 s substep |
| friction regularisation | ε_v = 1e-3 m/s (Newton default 1e-2) | less creep under a held load; 1e-2 made the pile *less* stable |
| per-body contact buffer | 512 | 128 overflowed (141) when a cable lay in a bin |

## 4. Robot

- **Model.** `assets/robots/yam_cart_yam_ultra.xml` loads through `add_mjcf`: 21 bodies, 19 coordinates.
  - The cart is parked by setting `base_y = ROBOT_MOUNT["cart_center_y"]`, the same in Newton and MuJoCo.
  - Newton's forward kinematics agree with MuJoCo's to 5e-7 m.
- **Kinematic arms.** Every robot body has zero mass and the KINEMATIC flag (Franka-Newton pattern).
  - Each 1/60 s frame has a joint target from MuJoCo damped-least-squares IK on the same MJCF.
  - Every substep interpolates the joints, runs `eval_fk` on a robot-only model and writes the poses into both VBD states.
  - Cartesian keyframes use min-jerk timing, stretched so the TCP stays under 0.25 m/s. The fastest arm joint in the final run moved at 1.18 rad/s.
- **Collision shapes.**
  - The fingertip pad boxes (8 × 30 × 39.5 mm, from the MJCF) and the palm boxes collide.
  - Convex hulls of the arm-link meshes collide.
  - The robot never collides with itself: filter pairs, since adjacent hulls overlap. Without them, contact generation between those hulls made the run 3.3× slower.
  - The cart's boxes have collision off.
- **Gripper force servo.** The squeeze per finger is read every frame with VBD's own contact law (mean ke × overlap) on the pad boxes.
  - The I2RT linear_4310 has one motor and a rack driving both fingers (joint7 = joint8). So the servo regulates the mean of the two finger forces, toward 6 N per finger, the same cap as the Genesis station.
  - It uses the spring-model update from the Franka cable: close at 5 cm/s until the first touch, then aim at q0 − G/k with an online estimate of k.
  - The settled hold force is 6.0 ± 0.4 N (median per finger); the worst transient is 8.3 N.
- **Grasp planning.**
  - The plug's grip point and lateral axis come from its pose. Approach is top-down where the YAM Ultra reaches. Otherwise the approach is pitched about the closing axis, so the pad faces stay vertical and a flat plug is still squeezed on its sides (`nrobot.grasp_R`).
  - Before a grasp is used, both pads, at the pre-close width, are swept back up along the approach by 9 cm. The sweep must contain no other cable node (inflated by its radius) and no other plug point. The lowest pad corner must stay at least 1.5 mm above the table.
  - Candidates are ranked by:
    - how trapped the cable is (top-view crossings where it lies under another cable);
    - cable length;
    - distance to the bin.
- **Sorting** (`sort_cable`).
  1. Plug transfer: grasp, lift with a 0.3 rad pitch (top-down reach ends ~0.15 m above the table near the pile), carry over the far half of the bin, lower to 12 cm, release.
  2. Feeds: the part still outside the bin is grasped by the cable body and carried in, until ≥ 75 % of the nodes are inside the cavity (at most 5 feeds). A feed is dragged low along the table to just outside the bin wall, then lifted over the rim.
     - A free outside tail is lifted in by its end.
     - A tail trapped in the pile is pulled out hand over hand: grasped next to the bin, so it slides out lengthwise.
     - A loop hanging over the rim is fed from its middle.
  3. Failure handling: a missed grasp (fingers closed on nothing) and a cable pulled out of the fingers during the carry are detected. The arm backs off and re-plans from the new state. Both are logged.

## 5. Budget and timings

The run budget is 60 fps × **40 substeps × 3 iterations** (dt = 0.42 ms).

Budget sweep: the 8-cable pile, 3 s each, no robot, no video, GPU shared. "Spikes" counts frames after 0.8 s with any segment above 0.3 m/s. KE and v are the solver's instantaneous body values.

| budget | wall for 3 s | spikes | max joint gap | max overlap | KE at 3 s | max v, last 1 s |
|---|---|---|---|---|---|---|
| 20×10 | 5.1 s | 59 | 8.9 mm | 6.1 mm | 7.7e-6 J | 0.48 m/s |
| 20×10, ε_v 1e-2 | 5.2 s | 88 | 16.9 mm | 3.4 mm | 2.8e-3 J | 3.15 m/s |
| 20×10, hard contacts | exploded | | | | | |
| 30×10 | 7.3 s | 69 | 7.1 mm | 6.5 mm | 2.2e-2 J | 1.57 m/s |
| 40×2 | 4.0 s | 98 | 23 mm | 3.7 mm | 9.5e-6 J | 2.47 m/s |
| 20×20 | 9.3 s | 0 | 1.0 mm | 0.48 mm | 1.1e-5 J | 0.29 m/s |
| 32×3 | 4.7 s | 0 | 0.24 mm | 0.53 mm | 5.3e-6 J | 0.26 m/s |
| 36×3 | 5.0 s | 0 | 0.15 mm | 0.15 mm | 5.3e-7 J | 0.07 m/s |
| **40×3** | **5.0 s** | **0** | **0.13 mm** | **0.10 mm** | **2.7e-6 J** | **0.08 m/s** |
| 40×5 | 6.1 s | 2 | 0.11 mm | 0.14 mm | 8.4e-8 J | 0.10 m/s |
| 60×3 | 6.3 s | 0 | 0.08 mm | 0.08 mm | 1.1e-7 J | 0.03 m/s |
| 80×2 | 6.8 s | 0 | 0.13 mm | 0.08 mm | 1.8e-7 J | 0.06 m/s |

For these thin, light cables (0.13-1.2 g segments), the substep length matters more than the iteration count. Below ~36 substeps per frame, contacts between cables spike and interpenetrate by millimetres, and the pile never settles. 40×3 is the cheapest budget with margin.

Final run timings (`timing.json`). The GPU was shared with 2 other processes at 99 % utilisation, 13 GB used at the start.

| phase | simulated | wall | wall / simulated s |
|---|---|---|---|
| drop + settle (8 cables, robot in scene) | 4.0 s | 8.7 s | **2.19** |
| approach | 18.3 s | 41.8 s | 2.29 |
| grasp (servo closing) | 3.5 s | 8.3 s | 2.37 |
| carry | 28.1 s | 63.7 s | 2.27 |
| release | 7.5 s | 16.9 s | 2.25 |
| retreat | 4.0 s | 9.3 s | 2.33 |
| **whole run** | **65.3 s** | **148.7 s** (+ 4.8 s build) | **2.28** |

These wall times include per-frame read-back (contacts, centrelines, grip forces), the physics log and 1280 × 720 rendering at 30 fps. Stepping alone, measured once when the GPU was idle (0 % utilisation, 40 MiB used), costs ~1.0 s per simulated second with the robot.

## 6. Results (final run; `checks.json`, all pass)

- **Pile.**
  - Settled at 4 s: KE 9.2e-8 J; fastest centreline node between the last two frames 3.1 mm/s (limits 2e-5 J and 0.02 m/s, from Genesis).
  - Top-view crossings 116 and 5 interlocked pairs (Genesis: 111 and 5).
  - Retention in the pile is 0.70 (micro-USB) to 0.99 (Cat6). Curvature ratio is 0.95-1.35.
- **Physics.**
  - Stretch [0.986, 1.036].
  - Lowest centreline point 0.05 mm above the table top.
  - Pads at least 2.0 mm above the table.
  - No NaN.
  - Bins shifted at most 1.3 mm and tilted at most 0.002°.
- **Largest overlap per contact pair, over all frames** (limit 1 mm):

  | pair | mm |
  |---|---|
  | cable-pad | 0.76 |
  | plug-pad | 0.63 |
  | cable-cable | 0.42 |
  | cable-self | 0.24 |
  | table-plug | 0.07 |
  | others | ≤ 0.05 |

- **Sorting.**
  - **barrel_arc → blue bin** with the right arm: plug transfer plus 1 feed, 100 % inside. Plug slip at the grip point 0.20 mm. Hold force median 6.03 / 5.97 N.
  - **cat6_scurve → orange bin** with the left arm: plug transfer plus 2 feeds, 100 % inside. Plug slip 0.72 mm. Hold force 5.66 / 6.18 N.
  - The planner first chose aux_loop for the left arm, but rejected it as unreachable before moving.
- **Repeat runs** with the same code (`tests/repeat1`, `tests/repeat2`, no video). All checks pass in both:

  | run | barrel_arc in blue bin | cat6 in orange bin |
  |---|---|---|
  | repeat1 | 85 % | 93 % |
  | repeat2 | 85 % | 88 % |

## 7. Comparison with the Genesis version

| | Genesis + libuipc (stations/cables) | Newton VBD (this) |
|---|---|---|
| cable model | `FEM.Rod` with ghost-point material frames and SoftVertexStitch network; isotropic EI, GJ = EI/1.45 | `add_rod` capsules and cable joints; rest curvature from `model.body_q`; GJ = EI |
| rest shape alone (retention) | ≥ 0.9 check (framed rod) | 0.980-0.998; straight-rest control 0.12 |
| bending | analytic spring fit (EI within 0.6 %) | calibrated per cable against the elastica (EI_eff/EI 0.999-1.000 at 40×3) |
| contact | IPC barrier, d_hat 1 mm, no interpenetration by construction | soft penalty, overlap = F/ke: ≤ 0.76 mm under grip, ≤ 0.42 mm between cables |
| time step | dt = 1/90 s | 1/60 s frames × 40 substeps × 3 iterations |
| plugs | none | 8 rigid plugs (IEC, HDMI, Europlug, RJ45, barrel, USB-C, micro-USB, TRS) |
| drop + settle cost (local shared 5090) | ~0.6 s per 1/90 s step settling (~55 s/s), 2.6 s/step during the drop (~230 s/s) (`out/cables/tests/profile_base.json`) | **2.2 s/s** (0.024 s per 1/90 s of simulated time), robot in scene, rendering included |
| same, unshared GPU on sc | 0.024 s/step settled (2.1 s/s), 0.24 s/step drop (21 s/s) (`profile_sc_base.json`) | not run on sc (no Newton env there) |
| robot phases | 600-830 s wall per simulated s (`out/cables/run.log`: t = 7→8 s took 634 s, 8→9 s took 826 s) | **2.3 s/s** |
| whole demo | not finished at the time of writing (the run log stops during the first lift) | drop, settle and 2 cables sorted: 65 s simulated in 2.5 min wall |
| pile | 111 crossings, 5 interlocked pairs | 116 crossings, 5 interlocked pairs (same drop layout) |

## 8. Known issues

- **Not bit-reproducible.** GPU float atomics make Newton runs differ slightly. The settled pile and which cable or grasp the planner picks vary from run to run; for example, barrel_arc retention in the pile ranged 0.70-0.92.
  - The planner works from the current state and handles misses and drops. All 3 full runs with the final code passed every check: the exported run, `tests/repeat1` and `tests/repeat2`.
  - Earlier versions of the feeding logic did fail. Their losses are documented in §4.
- **The left arm's sorting job is hard.** Its only top-down orange cable, cat6, was dropped first and lies at the bottom of the pile, with ~20 cables crossing over it.
  - Lifting it sideways at 6 N pulled it out of the fingers: 2 × μ 0.89 × 6 N ≈ 10.7 N of friction.
  - The hand-over-hand, drag-low feeding fixed this in the final runs. A deliberate untangling strategy (remove the cables on top first) is not implemented.
  - Far-side orange plugs need a pitched grasp. A round TRS plug lying on the table cannot be gripped pitched: the pad's lowest corner would hit the table before the pad covers the plug's lower half. Those candidates are rejected.
- **Bend calibration is static.** Dynamics at 40×3 (wave speed, damping) were not compared with Genesis. Thin leads need s ≈ 1.5-1.9 to reach the static EI.
- **Torsion.** GJ = EI rather than 0.69 EI: Newton's cable joint has one stiffness for bend and twist.
- **Penalty contact.** Contacts are penalty-based, not barrier-based. Overlaps stay under 1 mm, but they are not zero.
  - The solver's instantaneous substep velocities show micrometre contact noise on a few 0.15 g segments (up to ~3 cm/s, i.e. < 0.1 mm per frame). It is invisible.
  - The settle check uses frame-to-frame centreline velocity, as the Genesis check does.
- **Kinematic robot.** The arm has no dynamics or torque limits. A 30 g plug cannot push a YAM off its path, but a stalled arm would not be detected. The gripper force comes from the contact law, not from finger dynamics.
- **Preview video.** It is Newton's GL viewer, which draws capsule chains, so faint seams show at bends. The export carries smooth rods (radius and colour) for the compositor.
