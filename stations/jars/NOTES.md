# Jars station: notes

Ziga's brief: "just glass jars with covers that have to be screwed on." Three glass jars with real, unequal screw
threads. Their three lids lie loose on the table, and each lid fits only its own jar.

## Summary

Three glass jars (spice, jam, mason) with real procedural helical threads, and three loose screw lids that fit only
their own jar. All parts are exact-mesh libuipc affine bodies.

- **Feasibility test.** A disclosed torque about the jar axis turns each lid. The lid descends **one pitch per turn**:
  3.192 / 3.600 / 4.229 mm per turn against 3.2 / 3.6 / 4.23 mm, within 0.26%. It **stops on the rim**
  (0.196 mm, the contact gap) and **unscrews in reverse** at the same rate. Left untwisted, a lid just rests on the
  thread.
- **Robot demo** (`out/jars/`, the main export). The wheeled bimanual YAM Pro screws **all three lids on**. For each
  jar it picks the lid, drives over, holds the jar with one hand, and makes two 218° wrist strokes with a regrasp.
  Every lid ends 0.186-0.187 mm above its rim, coaxial within 0.015 mm, after 1.09-1.12 turns, and `success()` is
  true for all three.

## Geometry (procedural, `geometry.py`)

Every part is a lathe (a closed r-z profile revolved about +z) unioned with a swept thread ridge. The ridge is an
axial trapezoid profile swept along a right-handed helix, and its ends taper over 45 degrees into the wall like a
real thread run-out. The union is done with manifold3d. Every mesh is checked to be one closed, consistently
oriented 2-manifold of genus 0 (Euler characteristic 2), and every lathe profile is checked to be a simple polygon.

| | spice jar | jam jar | mason jar (regular-mouth pint) |
|---|---|---|---|
| body | Ø 48 mm, rim at 100 mm | Ø 70 mm, rim at 98 mm | Ø 84 mm, rim at 128 mm, bead under the thread |
| glass wall / base | 2.5 / 5.0 mm (neck 2.4) | 3.0 / 6.0 mm (neck 2.8) | 3.5 / 6.0 mm (neck 3.0) |
| finish (thread crest Ø) | 43 mm | 63 mm | 70.4 mm |
| pitch | 3.2 mm (8 TPI) | 3.6 mm | 4.23 mm (6 TPI) |
| thread depth | 1.0 mm | 1.2 mm | 1.4 mm |
| jar / lid thread turns | 1.75 / 1.5 | 1.5 / 1.5 | 1.75 / 1.5 |
| lid | red polypropylene, ribbed rim (90 ribs), Ø 47.2 mm | gold-lacquered tinplate, smooth, Ø 66.4 mm | zinc-plated steel, knurled (150), Ø 74 mm |
| mass jar / lid | 104 g / 6 g | 197 g / 13 g | 336 g / 20 g |

The diameters follow common closure finishes: 43 mm spice caps, 63 mm jam lids, and the 70 mm regular-mouth mason
finish at 6 threads per inch. Jar masses are not tuned. They come from soda-lime glass (2500 kg/m³) and the real wall
volume of the mesh, and they land on real jars (a 4 oz spice jar is about 110 g, a 340 g jam jar about 200 g, a Ball
pint about 340-400 g).

The thread profile has 30-degree flanks (60 degrees included). Clearances are designed against the contact barrier:
- **Radial clearance:** 0.5 mm crest-to-root on both parts.
- **Axial clearance:** 0.4 mm per flank, with the lid centred in the groove (0.8 mm backlash), which gives a
  **0.346 mm normal flank gap**.
- **Lid profile:** the lid crest width is P - w_root + 2·c_r·tan30 - 2·g, so the lid's flanks are parallel to the
  jar's.

The helices are phased so that a lid riding on its lower flank lands with its seat plane exactly on the rim. The
lid frame origin is that seat plane (the underside of the top panel), so a seated lid has lid origin = (0, 0, rim_z)
in the jar frame.

Meshes:
- `assets/<jar>_{jar,lid}_col.obj`: collision meshes, 120 segments per turn, trapezoid thread, 3.5-8.3 k vertices.
  Facet sag is at most 0.015 mm, against a 0.2 mm d_hat.
- `*_vis.obj`: render meshes, 240+ segments, chamfered thread crests, ribbed/knurled lid rims. They stay within
  0.1 mm of the collision surfaces. The measured maximum is 0.07-0.09 mm outside, all of it at fillets that the
  collision mesh samples more coarsely; the ribs and knurls are cut inward.
- All meshes are in metres, z-up, in the same frame as the collision mesh.

The lathe columns sit half a step off the thread samples, because coincident axial-plane edges gave the boolean
zero-length edges. The neck has no rings inside the thread band, because ring vertices near the intersection curve
made slivers. The minimum edge on a collision mesh is 7-60 µm.

Static fit check (`check_fit.py` → `assets/fit_check.json`, exact vertex/triangle distances, no physics):
- **Right-hand screw path:** sweeping the lid along it (z = rim + g + n·P, yaw = 2π·n, n = 0 … 1.5 turns) keeps the
  clearance at **0.343-0.347 mm** for every n, matching the design 0.346.
- **Wrong-handed path:** collides by 0.5-0.9 mm, so the threads really interlock.
- **Mismatched lids:** each wrong lid is too small for the neck (its crest falls inside the jar root) or too big
  (its thread passes over the jar thread and drops over the neck). Only the matching lid can engage.

The SAM 3D jar was not needed. Proportions come from the closure standards above.

## Physics parameters (`station.py`)

| parameter | value | why |
|---|---|---|
| solver | libuipc through Genesis `IPCCouplerOptions`, every part an affine body | exact mesh-mesh contact, needed for threads |
| jars, lids | `coup_type="ipc_only"`, free | no pins, no kinematics |
| table top | fixed `ipc_only` box, 1.20 × 0.70 × 0.03 m, top at z = 0.750 | layout.py |
| dt | 1/300 s (10 IPC steps per 30 fps frame) | a lid turning 0.5 turn/s moves 0.4 mm/step at its crest |
| contact_d_hat | 0.2 mm | below every designed gap (0.346 flank, 0.5 radial), well above facet sag (0.015) |
| newton velocity tol | 2e-4 m/s | measured: at 5e-3 a lid resting cocked on its thread start crept 95 µm/s (slow-motion tipping); at 2e-4 the creep is 0.06 µm/s |
| contact_eps_velocity | 1e-4 m/s | IPC friction smoothing. At the 1e-2 default, sub-threshold loads creep at mm/s (a lid would "screw itself down" under its own weight) |
| friction (per body, pairs = geometric mean) | glass 0.45, metal lid 0.45, plastic lid 0.30, table 0.50, robot pads 1.0 (robot default) | dry glass-on-steel/lacquer about 0.4-0.5; PP on glass about 0.3; glass on a laminate top about 0.5 |
| lid density | effective, mass / mesh volume (1117 / 1784 / 1958 kg/m³) | a solid 1.2-1.6 mm shell stands in for 0.2 mm tinplate plus liner (or a 1.6 mm PP cap); the mass is right |
| initial clearance | everything lifted 0.6 mm | libuipc aborts init on zero-distance contact |

## Twist feasibility test (disclosed drive)

`run.py twist`. Each lid starts coaxial over its own jar, rotated so its thread is one turn out, and 1.5 mm above
that. It falls onto the thread and rests. Then a **torque about the jar axis** is applied to the lid. It goes in as a
libuipc `AffineBodyExternalBodyForce`, the 12-D generalized force F_A = ½ [τ]× A, which is a pure torque and does no
work on strain modes. An animator injected before world init updates it every step from a torque-limited angle servo
on the lid's yaw relative to its jar: 0.5 turn/s, with a slipping clutch at τ_max = 4 / 10 / 20 mN·m for
spice / jam / mason, about 0.4 of the torque at which each jar would slip on the table. The opening clutch is 1.5×
that. No force, position or velocity is ever set on a lid or a jar, so gravity and the thread flanks alone decide
how far the lid descends.

Why one turn out: a free lid sits level only if more than about half a turn of thread supports it. Started 1.25 turns
out, the lids sat cocked on the thread start (4.4 degrees, a real-world behaviour) and the pure twist then wedged
them. Lid threads were lengthened from 1.25 to 1.5 turns so that one turn out gives a level rest. A levelling couple
was tried and dropped: an explicit torque servo cannot be stiff enough at this dt for a 6 g lid (it needs ω_n of
about 300 rad/s).

## Robot integration notes (YAM Pro cart, `robot_demo.py`)

**Coupling fix (station-side monkeypatch, `station.patch_coupling_at_geometry_centroid`).** Genesis computes the
two-way soft-constraint reaction on a coupled robot link from the displacement of the link ORIGIN and applies it
there. libuipc's SoftTransformConstraint, however, is mass-weighted over the link's IPC geometry. On the YAM
fingertips the origin sits 70-90 mm from the 8 × 30 × 40 mm pad.
- **Failure:** a 12-14 mm lid loads only the bottom of the pad, so the pad copy tilts by about 0.7°. The origin then
  moves by about 1 mm, and the explicit reaction pulls the arm down while friction holds the copy back.
- **Measurement** (`scratch/gmin_*.log`): it diverged in about 10 steps, with the fingers crossing to -16 mm and the
  hand driven 6-9 cm through the table. This also happened with the robot module's own validated settings
  (d_hat 1 mm, resistance 1e7, coupling 10), which passed only because its test cube loads the whole pad.
- **Fix:** measure the spring at the geometry centroid, apply it there (force plus its moment about the origin), and
  keep Genesis's rotational spring. With that, the same grasp holds: 8 N per finger, 71.9 mm pad gap on the 74 mm
  mason lid, lifted 4.6 cm.
- This affects every station whose robot touches short objects with the pad tips. Worth adopting in
  `robots/yam_cart.py`.

**Poses come from libuipc, not Genesis.** Genesis's `get_pos`/`get_quat` of an `ipc_only` body lag libuipc's state by
exactly g·dt² straight down: 0.109 mm at dt 1/300, 0.436 mm at 1/150 (`scratch/t_frames.py`). Genesis integrates
gravity once more after the coupler writes the pose (the "double time integration" FIXME in
`IPCCoupler._post_advance_ipc_only`). Genesis also moves an imported mesh's link frame to its centre of mass and
principal axes; for the lids that is a 2-2.5 mm offset and a 42° rotation. `JarsEnv.pose` therefore returns
libuipc's body transform composed with the geom's `init_pos`/`init_quat`, which is the pose of the mesh-file frame.
That is what is recorded and what every check uses. The export check: recorded pose × mesh file equals libuipc's own
world vertices to ≤ 0.1 µm (`meta.json: frame_check_max_vertex_error_m`).

**Distance-checker bug, fixed.** trimesh's point-triangle closest point uses absolute tolerances. With meshes in
metres, sub-millimetre triangles gave wrong distances (0.218 mm reported where the truth was 0.0039 mm).
`meshdist.py` and `check_fit.py` now work in millimetres.

**Procedure** (`robot_demo.py`; only joint PD targets via IK, the base PD target and gripper commands; nothing on the
jars or lids is ever set). Per jar, in the order mason, jam, spice:

1. **Pick.** The wheeled base drives so the lid lies 0.15 m right of the cart centre. The right hand picks it
   top-down (8 N per finger) and lifts it to 14.5 cm, above every jar.
2. **Carry and approach.** The base drives so the jar sits 0.05 m right of the cart centre. Two simultaneous
   Cartesian moves follow: the right hand carries the lid above the mouth, and the left hand goes from home to in front
   of the jar, turning to a horizontal grasp. A joint-space swing here hit the jar in an earlier run.
3. **Hold.** The left hand holds the jar body with a horizontal side grasp, 6 cm up, at 15 N per finger.
4. **Land.** The right hand lowers the lid until its thread lands one turn out. The robot chooses the hand yaw at
   pick-up so the lid arrives at the right phase, and corrects the landing height for the measured lid yaw. A real
   robot would find the thread start by turning back until the click; that search is not modelled.
5. **Strokes.** Joint 6 turns from -1.9 to +1.9 rad, 218° clockwise from above, with velocity feed-forward and a
   0.25 N·m motor cap, while the hand follows the thread down one pitch per turn. Then the hand opens, waits until the
   pads have really opened, turns back and regrasps.
6. **Seat and release.** Strokes end at the seat (the wrist stalls, or the lid stops following the wrist by
   > 0.12 rad), or once the lid is within 0.5 mm of the rim. Then both hands open, again waiting for the pads to open,
   back off along Cartesian paths, and go home.

**Findings along the way:**
- The YAM wrist's joint damping (0.1 N·m·s/rad in the I2RT MJCF) eats 0.16 N·m at 1.6 rad/s, so the torque cap must
  cover it. At 0.10 N·m the stroke only reached half its travel.
- After a grasp, the robot module's finger target sits past closed (the squeeze target). `set_gripper` ramps from
  there, so a fixed 0.3 s wait left the pads still gripping. The hand then dragged the lid back 0.108 turn during a
  regrasp and dragged the jar 22 mm during release. Hence `open_wait`.
- A top-down twist at the jar row needs y ≤ -0.05. Jars therefore stand at y = -0.06 with the lids in front, placed
  between the jars so the lane in front of each jar stays free for the holding hand.

**Solver settings for the robot run** (`robot_demo.ROBOT_COUPLER`, dt 1/150).
- **Newton tolerance and friction smoothing: 1e-3 m/s each.** With the twist-test values (2e-4 and 1e-4) and a
  gripper squeezing a part, steps took 5-36 s. With these, about 0.35 s (`scratch/gmin_*.log`). Seated and held lids
  do not creep at these settings; a cocked free lid would.
- **Coupling strength: translation 10, rotation 40.** Translation 10 is the robot module's validated value; 100
  diverges. Rotation 40 is stable with the centroid patch and cuts the libuipc fingertip copies' lag behind Genesis's
  fingertips from 11.2 to 3.6 mm (0.9 mm p95 while twisting). Translation 20 changed nothing.

## Results

All numbers are from `out/jars/checks.json`, which `checks.py` measures from the recorded libuipc poses and the
collision meshes.

**Geometry** (`check_fit.py`). Screw-path clearance is 0.343-0.346 mm for every n in 0 … 1.5 turns, against 0.346
designed. The wrong-handed path collides by 0.50 / 0.69 / 0.88 mm. All 12 meshes are closed 2-manifolds with Euler
characteristic 2.

**Settle** (`out/jars/settle/`, the puzzle layout with no robot):
- **Clearance:** initial 0.49 mm everywhere, above d_hat 0.2 mm. Minimum distance over the run is 0.188 mm, so
  nothing ever penetrates.
- **Stillness over the last second:** drift 2.4e-12 m and rotation 1.7e-6°, so no jitter and no creep.
- **Resting gaps:** 0.188-0.197 mm to the table, inside d_hat.

**Twist test** (`out/jars/twist_test/`, disclosed external torque, dt 1/300; plot in `descent.png`):

| | spice | jam | mason |
|---|---|---|---|
| rest height above seat (1 turn out) | 3.428 mm, tilt 0.004° | 3.828 mm | 4.460 mm |
| rest drift, last 0.3 s | 0 | 0.06 µm | 0 |
| **descent per turn** (pitch) | **3.1918 mm** (3.2, -0.26%) | **3.5997 mm** (3.6, -0.01%) | **4.2289 mm** (4.23, -0.03%) |
| seated height above rim | 0.196 mm | 0.196 mm | 0.195 mm |
| seated coaxial offset / tilt | 2 µm / 0.0002° | 28 µm / 0.002° | 1 µm / 0.001° |
| turns from rest to seat | 1.117 | 1.102 | 1.089 |
| rise per turn when unscrewing | 3.190 mm (-0.32%) | 3.596 mm (-0.11%) | 4.225 mm (-0.11%) |
| jar rotation / slide during the test | 0.05° / 0.05 µm | 0.10° / 0.8 µm | 0.04° / 0.5 µm |
| max drive torque | 4.5 mN·m | 15 mN·m | 30 mN·m |

Minimum distance over all frames is 0.179 mm, so nothing penetrates. Unscrewed past the thread start, each lid rides
off the thread end and drops back onto the mouth, cocked: the real "clunk".

**Robot demo** (`out/jars/`, 62.3 s at dt 1/150, about 40 min wall on the shared 5090):

| | mason | jam | spice |
|---|---|---|---|
| lid landed (turns out, measured) | 0.999 | 1.000 | 1.000 |
| stroke 1: wrist / lid turns | 0.605 / 0.606 | 0.605 / 0.607 | 0.605 / 0.608 |
| regrasp: lid dragged | 0.000 turn | 0.000 turn | 0.001 turn |
| stroke 2 ends at the seat (lid stops following the wrist) | 0.507 wrist turn | 0.522 | 0.534 |
| final seat gap above rim | 0.186 mm | 0.188 mm | 0.186 mm |
| coaxial offset / tilt | 12 µm / 0.0003° | 6 µm / 0.0009° | 6 µm / 0.0004° |
| engaged turns | 1.093 | 1.109 | 1.124 |

- `success()`: all three lids on.
- Over every 3rd frame, the minimum distance between jars, lids and the table is 0.179 mm, so nothing penetrates.
- The drawn fingertip collision pads stay ≥ 0.094 mm outside every part.
- No drawn robot part is inside the table top.
- Fingertip Genesis-vs-libuipc offset: 3.6 mm max, 2.9 mm p95.

Files: `robot_result.json` (events), the stills `robot_t0.png`, `robot_after_{mason,jam,spice}.png` and
`robot_end_{wide,close}.png`, and the video `preview.mp4`.

## Status

- Geometry, physics, twist test, settle and robot demo are done, and the checks pass (summary at the bottom of
  checks.json).
- Exports: `out/jars/` (robot demo, the main one for the compositor), `out/jars/settle/`, `out/jars/twist_test/`.
  Each export's `meta.json` records `frame_check_max_vertex_error_m`, pose × mesh against libuipc's own vertices:
  0.25 µm at most.
- Running on sc: `stations/jars/sc_run.sh TAG run.py <part>` mirrors the project to
  `/viscam/group/ziga/sim/yam-arcade-jars` and rewrites the robot MJCF paths. It is untested end to end: the jobs sat
  in the sc-loprio queue and were cancelled once the local GPU freed up.

## Known issues

- **Drive disclosure.** The feasibility test drives lids with an external torque (disclosed above). The robot demo
  uses no external forces on the parts.
- **Lid phase.** The robot chooses the lid's yaw at pick-up so the thread lands one turn out, using its model of the
  thread phase. It does not search for the thread start by back-turning.
- **Fingertips are drawn at their libuipc (contact) poses**, which are what touch the parts; everything else is
  drawn at Genesis poses. Genesis's own fingertip links sit away from them by the two-way soft-coupling offset. Drawn
  from Genesis, they sank up to 3.3 mm into a lid. Drawn from libuipc, the drawn pads stay ≥ 0.09 mm outside every
  part, and the offset shows as finger play in the mount. Logged as `fingertip_genesis_vs_libuipc_offset_m` in
  checks.json: 3.6 mm max (2.9 mm p95) with rotation coupling 40, against 11 mm at the default 10.
- **Station-side Genesis patches** (both monkeypatches in our code, nothing edited in Genesis or `robots/`):
  - the coupling reaction measured at the geometry centroid;
  - poses read from libuipc rather than from Genesis's lagged ones.
- **Solver settings differ between runs.** The robot run uses Newton 1e-3 m/s and friction smoothing 1e-3 m/s for
  speed; the twist test uses 2e-4 and 1e-4. With the robot's settings, a cocked free lid would creep; in the demo
  every lid is held or seated, and the seated lids hold to 0.1 µm.
- **Tightening torque is small:** the clutch is 4-20 mN·m in the twist test and 0.25 N·m at the robot wrist. Real
  hand-tight is about 1-2 N·m. Plastisol liners are not modelled; the lid's top panel seats straight on the glass rim.
- **The SAM 3D jar reference was not used**, and no upside-down lid is in the layout (no flipping needed).
