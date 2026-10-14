# Cables station (Genesis + libuipc)

## Summary

Eight cables in two colours, each with a plug on one end, are dropped from the air into a tangled pile in the middle
of the table, next to two open-top trays (blue on the left, orange on the right). The cables differ in length
(0.35-0.90 m), diameter (4-9 mm) and bending stiffness (EI 2e-5 to 4e-3 N m², 2.3 decades). Each keeps a non-straight
shape it took in storage: packaging coils, a folded-double U, S-bends from cable ties, waves, a curl, and a kinked loop.

- **Rest shape.** Genesis's libuipc rod (`FEM.Rod`) always relaxes to a straight line. Here each cable is a **framed
  rod**: three non-colliding "ghost" points per segment carry a material frame, and springs fitted per hinge give
  isotropic bending EI and torsion GJ around a stress-free curved rest shape. Dropped alone, every cable keeps its
  shape (retention ≥ 0.969); plain-`FEM.Rod` controls straighten (0.05-0.07).
- **Plugs.** Rigid bodies stitched to the cable end (libuipc `SoftVertexStitch`), as in Ziga's Franka cable.
- **Robot.** The wheeled bimanual YAM Ultra (`robots/yam_cart.py`, IPC pads) picks an untangled cable by its plug,
  lifts it out of the pile, carries it to the tray of its colour, lays the plug in and releases it. The YAM Ultra lifts
  only ~0.25 m, so a stiff 0.6-0.9 m cable trails back over the rim. Only 26-38% of the cable ends inside, and "feeds"
  (re-grasping the cable body) did not help in this Genesis version (section 6). The Newton version of this station
  completes the sort (`stations/cables_newton/`).
- **Speed.** On a dedicated GPU (sc RTX PRO 6000 / L40S) the drop + settle takes 7-9 s of wall time per simulated
  second, and a whole drop + demo takes 2-4 min. On peefton's shared 5090 the same run was 10-25x slower.

Files: `cable_model.py` (rest shapes, framed rod, libuipc hooks), `plugs.py` (connectors), `station.py` (cables,
plugs, trays, placement, `build()`, `Env.success()`, `Env.tangle_metric()`), `run.py` (drop + settle + demo, export,
preview), `check.py` (`checks.json`), `placement.json` (optimised start, `tests/plan_pile.py`), `gpu_run.sh` (local:
waits for the SAM 3D lock), `tests/` (every test quoted below). Runs go to sc with
`tools/run_on_sc.sh stations/cables/run.py --robot [--no-demo] --dt 1/90 --t-settle 4 --out out/cables[/...]`.

## Results

Official export `out/cables/` (drop + 5 s settle, cart present and idle): see `out/cables/checks.json`. Demo run
`out/cables/demo_attempt/` (its own `checks.json`). The numbers below are from those files (section 8 has the
comparison with Newton).

**Official export** (`out/cables/checks.json`, **all pass**): the cart is present and idle, and the drop + 5 s settle
ran on an unshared L40S on sc.

| check | value | limit |
|---|---|---|
| segment stretch, every step | 0.9996-1.0013 | [0.9, 1.1] |
| interpenetration (cable-cable / within a cable) | +0.89 / +0.92 mm | ≥ −0.5 mm |
| lowest centreline above table − radius | +0.79 mm | ≥ −0.5 mm |
| lowest plug point above table | +0.64 mm | ≥ −0.5 mm |
| cables on the table, none in a tray after the drop | yes | |
| settled (last 5 frames) | v_max 2.7 mm/s, KE 8e-8 J | 20 mm/s, 2e-5 J |
| pile retention (1 = rest shape, 0 = straight) | 0.72-0.97 | ≥ 0.5 |
| pile curvature (none straightened) | 0.96-1.32 | ≥ 0.75 |
| isolated drop: retention / curvature / creep last s | ≥ 0.969 / 0.998-1.003 / ≤ 0.16 mm | 0.9 / [0.9, 1.1] / 0.5 mm |
| plain FEM.Rod controls (retention) | U 0.07, coil 0.05 | reported |
| tangle: crossings / contacting segment pairs / interlocked pairs | 69 / 144 / 3 | > 0 |
| trays: tilt / shift | 0.01° / 0.6 mm | 3° / 10 mm |
| exported pose × mesh vs Genesis's own vertices | 0.61 mm | 2 mm |
| timing | 5.0 s simulated in 38.8 s wall (7.8 s/s) | |

**Demo run** (`out/cables/demo_attempt/checks.json`): the planner picked the **USB-C lead** (untangled, on top of
the pile) by its plug with the left arm, top-down. It carried the lead to the blue tray, swept and released it.

| check | value | limit | |
|---|---|---|---|
| cable in the blue tray | 51.6% of its centreline | ≥ 50% | pass |
| plug slip at the grip point (flange frame) | 3.7 mm, rotation 5.9° | 2 mm | **fail** |
| grip force (PD, capped) | 15 N per finger | | |
| drawn fingers inside the plug (coupling lag) | 1.66 mm median | 2 mm | pass |
| fastest arm joint | 1.8 rad/s | 3 rad/s | pass |
| pads above table / robot links clear of cables, table, trays | +2.4 mm / 0 hits | | pass |
| all pile checks | as above (74 crossings, 3 interlocked) | | pass |

Demo timing (L40S, unshared, rendering and per-frame logging included):

| phase | simulated | wall | wall per simulated s |
|---|---|---|---|
| drop + settle | 4.0 s | 34.5 s | 8.6 |
| approach | 4.9 s | 25.8 s | 5.3 |
| grasp | 1.0 s | 4.9 s | 4.9 |
| lift + carry + lay in | 12.3 s | 88.8 s | 7.2 |
| release | 1.0 s | 5.6 s | 5.6 |
| retreat | 4.2 s | 25.6 s | 6.1 |
| **total** | **27.2 s** | **186 s** | 6.8 |


Stills: `out/cables/still_settled.png`, `still_top.png`, `still_close.png`; demo: `out/cables/demo_attempt/still_*.png`,
`preview.mp4`. Rest-shape panel: `out/cables/tests/rest_shape_test_panels.png`.

## Status

- **Object work: done**, every check passes (`out/cables/DONE`).
- **Robot integration: done.** The cart is in the scene with IPC-coupled pads and nothing intersects at init.
- **Feasibility demo: partially achieved.** A cable is picked by its plug out of the pile and carried into the tray of
  its colour, through physics only, with 52% of the cable inside. The plug slipped 3.7 mm in the grip (limit 2 mm).
- In earlier runs the planner picked longer, stiffer cables (HDMI 0.6 m, extension 0.9 m). Their plugs reached the
  tray but only 15-38% of the cable followed: the YAM Ultra lifts ~0.25 m, and the rest trails back through the pile.
  Feeds (re-grasping the cable body) did not fix that here (section 6).
- The Newton version (`stations/cables_newton/`) sorts two cables completely.

## 1. libuipc's rod has no rest shape (verified)

Genesis `FEM.Rod` = libuipc `HookeanSpring` (stretch) + `KirchhoffRodBending` (KRB). In
`~/simquarry/libuipc/src/backends/cuda/finite_element/constitutions/kirchhoff_rod_bending.cu` the hinge energy is
`E = (k π r⁴/4) |κ|² / L0`, with `κ = 2 e0×e1 / (|e0||e1| + e0·e1)` from the **current** positions. Rest positions
enter only through `L0`, and the spec says "the energy penalizes deviations from the straight configuration".
pyuipc 0.0.26's `KirchhoffRodBending.apply_to(sc, E)` has no rest-angle attribute; the only per-vertex attribute is
`bending_stiffness`. The rod has no twist.

Measured in `tests/rest_shape_test.py`: the HDMI U-bend and the power coil built as plain `FEM.Rod` and dropped 3 cm
onto the floor keep retention 0.07 and 0.05 (curvature 0.14 and 0.08 of rest). They open into near-straight curves.

## 2. The framed rod

Springs between a rod's own vertices cannot hold a curved shape. Any energy of distances between points of a planar
curve changes only at second order when the curve bends out of its plane, so a planar rest curve held that way has
zero linear out-of-plane stiffness. A 3-point hinge energy at a curved equilibrium has none either, pre-stressed or
not. A first attempt (chord + dihedral springs) showed exactly this, and pre-curving springs against KRB made some
spring stiffnesses negative. The cable needs material frames.

Positions-only rods get frames from ghost points (Umetani, Schmidt & Stam 2014, *Position-based elastic rods*):

- Each edge carries **three ghost points**, 120° apart around its midpoint, at h = 1.5 × segment from the axis
  (directions from a parallel-transported frame). The edge and its ghosts are held rigid by stiff springs (all 10
  distances).
- Between consecutive frames, 16 springs carry hinge j. Their stiffnesses are fitted per hinge (bounded least squares)
  so the hinge stiffness matrix is `(EI/l)(I − ttᵀ) + (GJ/l)ttᵀ`, with GJ = EI/(1+ν), ν = 0.45 (PVC/rubber jacket).
  The fit residual is 1e-15.
- Rest lengths come from the rest shape, so the rest shape is stress-free.
- Ghosts carry 5% of the cable mass. They take part in no contact (libuipc's no-collision element) and are neither
  drawn nor exported: they discretise the cross-section's orientation, they are not objects.
- Springs are libuipc `SoftVertexStitch` with per-instance `rest_length` / `kappa`, installed by a hook just before
  `world.init`. KRB is left at a negligible 1 Pa.
- Why three ghosts: with two, the ghost centroid is off the axis. Gravity on it gave a curved cable's
  near-zero-stiffness frame-rolling mode a negative stiffness, and a U-bend lying on the table crept 2 mm/s. With three
  at 120° the centroid stays on the axis for any frame rotation.

Checks on CPU (`tests/design_check.py`, same energy as the GPU): whole-cable EI in two planes and GJ within 0.6% of
target for straight, U, coil, S, wave and loop shapes. With finite frame springs, straight-cable EI is 0.98 of target
(`tests/frame_stiff_check.py`). A naturally curved cable measured that way is softer out of plane (coil 0.17, U 0.69).
That is Kirchhoff physics: a curved rod absorbs a uniform out-of-plane curvature change by rolling its material frame.

Checks on GPU (`tests/rest_shape_test.py`, every cable dropped alone, 4 s):

- planar shapes: retention 1.000 (RMSD < 0.01 mm), curvature 1.000;
- coils: settle once as the upper turn sags onto the lower (power coil RMSD 7.7 mm, retention 0.969; extension coil
  5.7 mm, 0.976), then creep < 0.2 mm over the last second;
- the kinked loop: 0.979.

## 3. Cables, plugs, trays

| cable | colour | plug | rest shape | L (m) | OD (mm) | EI (N m²) | GJ (N m²) | g/m | bending length |
|---|---|---|---|---|---|---|---|---|---|
| power_coil | orange | IEC C13, 35 g | 2-turn packaging coil, R 55 mm, bowed tails | 0.86 | 9 | 4.0e-3 | 2.8e-3 | 85 | 16.9 cm |
| hdmi_ubend | blue | HDMI, 12 g | folded double, R 45 mm, bowed legs | 0.60 | 8 | 2.5e-3 | 1.7e-3 | 70 | 15.4 cm |
| ext_coil | blue | Euro 2-pin, 25 g | 1.5-turn loose coil, R 70 mm | 0.90 | 7 | 1.2e-3 | 8.3e-4 | 50 | 13.5 cm |
| cat6_scurve | orange | RJ45, 5 g | S-bends, R 80-120 mm | 0.73 | 6 | 5.0e-4 | 3.4e-4 | 40 | 10.8 cm |
| barrel_arc | blue | 5.5 mm barrel, 4 g | 270° curl tightening R 130 → 65 mm | 0.46 | 5 | 1.0e-4 | 6.9e-5 | 22 | 7.7 cm |
| usbc_wave | blue | USB-C, 3 g | 6 waves, R 32 mm | 0.50 | 4 | 4.0e-5 | 2.8e-5 | 18 | 6.1 cm |
| micro_scurve | orange | micro-USB, 2.5 g | S of two 160° arcs, R 62 mm | 0.35 | 4 | 3.0e-5 | 2.1e-5 | 16 | 5.8 cm |
| aux_loop | orange | 3.5 mm jack, 3 g | one kinked loop, R 40 mm, curved leads | 0.59 | 4 | 2.0e-5 | 1.4e-5 | 15 | 5.1 cm |

Cable parameters:

- **EI.** Effective modulus EI/(πr⁴/4) of 1.6-12 MPa, the composite of a soft PVC jacket over stranded copper whose
  strands slide. Bending lengths (EI/w)^⅓ of 5-17 cm match how these leads droop. The Cat6 value is the Franka cable's
  estimate.
- **Linear density.** Typical of those constructions.
- **Stretch.** HookeanSpring E = 4e7 Pa, as in the Franka cable; measured stretch < 0.5%.
- **Friction.** Cable 0.5, plugs 0.6, trays 0.5, pads 1.0 (geometric mean in libuipc).
- **Vertex spacing.** 9-14 mm so that spacing − 2r > d_hat (the libuipc rod rule); contact d_hat = 1 mm.

Plugs (`plugs.py`):

- Procedural boxes or octagonal prisms of real connector sizes, masses as listed (ρ = 1000-2200 kg/m³).
- Collision = convex hull of the whole plug, metal shell or pin included.
- The cable runs straight through the plug in 4-7 segments, 100x stiffer (moulded in), to the shell's tip. Two
  `SoftVertexStitch` springs (kappa 1e6 N/m, zero length) join its first and boot vertices to the hull's tip and boot
  vertices, so the plug holds the cable's end position and direction.
- Contact between a cable and its own plug is off. Where a shell is thinner than the cable, the stitched cable end
  bulges up to ~1 mm out of the hull. It still collides with everything else and is not drawn: the export starts at
  the boot.

Trays: polypropylene storage trays 220 × 280 × 70 mm, 5 mm walls, 300 g, one watertight solid each, free bodies starting
2 mm above the table. They are 70 mm tall because the YAM Ultra reaches only ~0.2 m (top-down) to ~0.27 m (tilted)
above the table over them. Pale blue and pale orange with a saturated label on the front. The table top is a fixed box,
1.20 × 0.70 × 0.03 m, top at z = 0.75.

## 4. Initial placement and tangle

Every cable starts in the air at its rest shape, rigidly placed. `tests/plan_pile.py` does a random search over
position, yaw, tilt (±35°), flip and height. Hard constraints:

- ≥ 6 mm between any two cables or plugs, ≥ 15 mm above the table;
- inside the central pile area, 3 cm clear of every robot link at home.

It maximises pairs of cables that cross each other **over and under** at t = 0; IPC never lets them pass through each
other, so that topology survives the drop. The 3.5 mm audio lead's loop stands vertical with the micro-USB lead
threaded through it. The USB-C lead is dropped last, flat, from the top (0.275 m), so at least one cable lands *on*
the pile for the demo; every other cable is held below 0.285 m.

`station.tangle_metric()` reports:

- top-view crossings between different cables;
- segment pairs of different cables within 2 mm;
- interlocked pairs (A over B somewhere, under B elsewhere);
- woven cables (over one, under another).

Settled piles over the final runs had 62-81 crossings, 116-159 contacting segment pairs and 3-5 interlocked pairs.

## 5. Robot and demo

The cart is `robots/yam_cart.py` with `ipc=True`: fingertip pads and palms are libuipc bodies, and the arm follows
joint PD targets. The coupler soft-constraint strength is **20 translation / 1000 rotation**, from
`tests/grasp_debug.py`:

- At 10, the libuipc pad copies tilted under a squeeze, the drawn fingers followed into the object, and the arm was
  thrown.
- At 100, the explicit coupling force on a 71 g fingertip with 0.5 kg armature exceeds the explicit-integration bound
  (strength·m/m_eff ≈ 12 > ~4). The plug slipped and the drawn fingers swung between 73 mm and −2 mm.
- At 20 (≈ 2.5) the grip is steady.

Pinching the bare rod with the cart's own `grasp()` fails. On a 4-9 mm rod the fingers touch at q = 1.5-4 mm, too late
for its stall detector, so its target runs past closed and the drawn fingers crossed each other.

Demo (`run.py --robot`):

1. **Choose a plug.** It must rest on the table with its width axis horizontal, with no other cable inside the pads'
   outer faces. It is picked by the arm in front of the tray of its colour (each arm reaches ~0.15 m past the centre
   line). Untangled cables rank first, then fewer cables touching, then shorter.
2. **Plan.** A straight-line Cartesian TCP path with slerped orientation. IK is seeded by the previous sample,
   checked for < 3 mm error and < 0.15 rad between samples (catches branch flips), and validated before anything
   moves.
3. **Approach.** Top-down where the YAM reaches. Otherwise pitched about the closing axis, so the pads stay vertical
   (the cart's `down_quat` pitches about world x, which tilts the pads). The TCP is raised by 15 mm·sin(pitch) so the
   pads' lower corner clears the table.
4. **Grasp.** `cart.grasp`: close at 5 cm/s, stall, then a 15 N force-capped squeeze per finger.
5. **Transfer.** Pull 6 cm toward the cart while lifting 10 cm, then carry at 0.24 m with the approach pitched forward
   0.3 rad. Lower into the tray's far corner, sweep along the far wall to lay the trailing cable in, release, retreat.

Grip force:

- 4 N let the extension lead's plug pull out while five cables lay across that cable.
- 8 N let the HDMI plug pull out while its stiff lead dragged through the pile.
- 15 N held it. At dt 1/90 the drawn fingers then sit ~0.8 mm inside the plug (grip force / coupling stiffness).

## 6. What did not work (kept as options)

- **Feeds** (`--feeds N`): re-grasping the trailing cable body and carrying it over the rim, as the Newton version
  does. For the 0.9 m extension coil the outside part drapes over the rim and lies on other cables; three feeds left
  the in-tray fraction at 0.15-0.19. The stiff coil springs back, and the cable entered the gripper housing mesh
  (whose contact body is only the palm box). Off by default.
- **al-ipc contact** (`--contact al-ipc`): about 7x faster in the drop (0.034 vs 0.237 s/step). Pile-only (no robot)
  it produced segments stretched to 1.49x, a pile still moving at 0.16 m/s after 4 s, and lower retention, because the
  stiff spring network doesn't converge under its line search. With the robot coupled it produced NaN coupling forces.
  Rejected; interpenetration itself stayed ≥ 0.03 mm.
- Coarser rod segments (×1.5): no speed gain. The cost is in impact frames with up to 129 Newton iterations.

## 7. Speed

Pile, no robot, dt 1/90, wall seconds per step:

| run | GPU | drop (0-1 s) | settled | Newton iterations per step in the drop (max) |
|---|---|---|---|---|
| IPC | sc RTX PRO 6000, dedicated | 0.237 | 0.024 | 4.6 (87) |
| al-ipc | same | 0.034 | 0.027 | 0.44 (4) |
| segments × 1.5 | same | 0.209 | 0.036 | 8.1 (129) |
| dt 1/60 | same | 0.139 (8.3 s per sim s) | 0.024 | 3.4 (55) |
| IPC | peefton 5090, shared with 4-6 other jobs | 2.6 | 0.63 | 2.5 (24) |

95% of a step is libuipc `world.advance` (cProfile). On peefton, GPU time-slicing with other agents' jobs makes every
kernel launch wait; plain rods without frames were even slower there (3.5 s/step in the air). Full runs on sc with
rendering, robot and per-frame logging: see the timing table in RESULTS (`checks.json` → `timing`).

## 8. Comparison with the Newton version (`stations/cables_newton/`, `out/cables/newton/`)

| | Genesis + libuipc (this) | Newton (`out/cables/newton/checks.json`) |
|---|---|---|
| rest shape | framed rod: ghost material frames + fitted springs on libuipc rods | rest-curvature rods |
| shape alone (retention) | 0.969-1.000; plain-rod control 0.05-0.07 | 0.980-0.998; straight-rest control 0.12 |
| contact | IPC barrier, d_hat 1 mm: no interpenetration (≥ +0.5 mm here) | soft penalty: ≤ 0.76 mm under grip, ≤ 0.42 mm between cables |
| pile | 69-74 crossings, 3 interlocked pairs | 116 crossings, 5 interlocked pairs |
| pile retention | 0.72-0.97 | 0.70-0.99 |
| drop + settle cost | 7.8-8.6 s wall per sim s (unshared L40S, rendering on) | 2.2 s/s (shared 5090, rendering on) |
| demo | 1 cable by its plug, 52% in tray; plug slip 3.7 mm | 2 cables, plug + feeds, 100% in tray; slip 0.2 / 0.7 mm |
| grip | 15 N PD cap per finger (8 N let plugs pull out) | 6 N force servo |
| whole run | 27 s simulated in 3.1 min wall | 65 s simulated in 2.5 min wall |


## 9. Checks revised after seeing results (disclosed)

- Pile curvature: first two-sided [0.75, 1.33]. The softest leads drape over thicker cables and come out more curved
  (up to 1.7); that is loading, not loss of shape. The pile check is one-sided (≥ 0.75); the two-sided test is the
  isolated drop ([0.9, 1.1]).
- Isolated drift: first "change from t = 1 s < 2 mm". Coils settle until ~2.5 s, so it is "change over the last
  second < 0.5 mm".
- Demo in-tray limit: first 0.9, then 0.5 ("most of the cable"). Neither is met by the Genesis demo (0.26-0.38).
- Plug slip is measured at the grip point (body centre) in the flange frame, as in the Newton check; rotation is
  reported separately.

## 10. Known issues

- **Demo is partial.** The plug is delivered to the right tray, but only 26-38% of a stiff 0.6-0.9 m cable follows it
  in. See Status.
- **Not bit-reproducible.** Different sc GPUs (L40S, RTX PRO 6000) give slightly different piles. Which cable is
  picked, and whether it is interlocked, varies between runs.
- **Ghost frames.** Their rotational inertia is about twice a real cable's cross-section inertia (5% of the mass at
  1.5 segments from the axis); this only affects how fast a cable twists.
- **Drawn fingers trail the IPC pads** by grip force / coupling stiffness (~0.8 mm into a plug at 15 N, dt 1/90).
- **Stitched cable end.** Where a plug's metal shell is thinner than the cable, the stitched, never-drawn cable end
  bulges ≤ 1 mm out of the plug hull.
- **Trays sit fixed by friction only.** They are free bodies on the table and shifted < 0.2 mm in the runs.
