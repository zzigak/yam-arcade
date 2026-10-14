# robots/ — wheeled bimanual I2RT YAM cart

One articulated Genesis entity per station: a wheeled base (procedural replica of I2RT's own **Flow Base**,
the four-caster swerve base in `assets/i2rt_upstream/i2rt/flow_base/`, with its lift column), an arm deck at
table-top height, an overhead D405 mast, and **two arms of the station's YAM variant with I2RT `linear_4310`
grippers**. Arm and gripper MJCFs come straight from the I2RT checkout and are composed the way
`i2rt.robots.utils.combine_arm_and_gripper_xml` does it.

| file | what |
|---|---|
| `yam_cart.py` | the API stations import (`add_cart`, `Cart`) |
| `build_models.py` | regenerates `assets/robots/yam_cart_<variant>.xml` (+ `assets/robots/meshes/<variant>/`) |
| `measure_gripper.py` | measures fingertip pads / palm from the meshes → `assets/robots/gripper_collision.json` |
| `test_rigid.py` | load, joints, hold, table clearance, gripper travel, IK reach maps → `checks/rigid_<variant>.json` |
| `test_grasp.py` | 4 cm cube grasp + 10 cm lift, rigid (`gs.cpu`) or `--ipc` (`gs.gpu`) → `checks/grasp_*.json` |
| `test_export.py` | Recorder round trip; Genesis link frames == exported mesh frames → `checks/export_<variant>.json` |
| `preview.py` | rasterizer previews `preview_<variant>.png` |

## Quick start

```python
import os, sys
sys.path.insert(0, os.path.expanduser("~/projects/yam-arcade"))
sys.path.insert(0, os.path.expanduser("~/projects/yam-arcade/robots"))
import genesis as gs, layout
from yam_cart import add_cart

cart = add_cart(scene, gs, "yam_pro", ipc=False)   # BEFORE scene.build(); parks per layout.ROBOT_MOUNT
scene.build()
cart.after_build()                                 # gains, torque limits, home pose (call once)

q, tilt = cart.ik_grasp("left", (0.10, 0.0, layout.TABLE_TOP_Z + 0.03), yaw=0.0)   # TCP target, top-down if possible
for qk in cart.joint_path("left", q, duration=1.5, dt=scene.dt):                    # min-jerk joint interpolation
    cart.set_arm_target("left", qk)
    cart.update()                                  # advances gripper targets; call EVERY step before scene.step()
    scene.step()
cart.grasp("left", force=10.0)                     # close at 5 cm/s, then force-capped squeeze (N per finger)
...
cart.open("left")                                  # or cart.set_gripper("left", width=0.06)
```

## API (`yam_cart.py`)

- `add_cart(scene, gs, variant, gripper="linear_4310", ipc=False, pos=None, yaw_deg=0.0, coup_links=None,
  coup_friction=1.0, name="robot", gravity_compensation=1.0)` → `Cart`. `variant` ∈ `yam | yam_pro | yam_ultra | big_yam`. Default placement
  `pos=(0, ROBOT_MOUNT["cart_center_y"], 0)` in the station frame. `ipc=True` gives the entity
  `gs.materials.Rigid(coup_type="two_way_soft_constraint", coup_links=cart.coup_links(), gravity_compensation=1.0)`.
  Rigid mode uses `gravity_compensation=1.0` too (real YAMs run gravity-comp feedforward).
- `cart.after_build(home=True)` — must be called once after `scene.build()`.
- Attributes: `entity`, `base_dofs` (3: `base_x, base_y, base_yaw`), `arm_dofs[side]` (6), `finger_dofs[side]` (2),
  `ee_link[side]` (`<side>_gripper`, the joint6 flange), `tcp_offset` (TCP in the flange frame: 144.65 mm along
  flange +z, I2RT's `grasp_site` = midpoint of the fingertip end facets), `home_qpos` (full-robot qpos),
  `q_target` (current PD targets). Sides: `"left"` is at **-x**, `"right"` at **+x** (robot faces +y).
- Kinematics:
  - `cart.ik(side, pos, quat=None, yaw=0.0, tilt=0.0, init_qpos=None, return_error=False)` → 6 arm angles. `pos`
    is the TCP (world = station frame). `quat` is the flange orientation (w,x,y,z); default `down_quat(yaw, tilt)`.
    Only that arm's dofs move.
  - `cart.ik_grasp(side, pos, yaw=0.0, tilts=GRASP_TILTS)` → `(q, tilt)` or `(None, None)`: tries top-down first,
    then approach axes pitched 0.25 … 0.9 rad towards +y. **Use this for anything beyond y ≈ 0.05** with the
    YAM / Pro / Ultra (see reach below).
  - `down_quat(yaw, tilt)` — flange quat: approach along −z, fingers close along `(cos yaw, sin yaw, 0)`.
  - `cart.tcp_pose(side)` → (TCP world pos, flange quat); `cart.fk_tcp(side, q_arm)`.
- Control (all PD targets; nothing is teleported during a run):
  - `cart.set_arm_target(side, q6)`, `cart.set_base_target(x, y, yaw)` (offset from the parked pose, stiff PD),
    `cart.set_qpos(q)` (teleport — only before the run starts).
  - `cart.joint_path(side, q_goal, duration, dt)` → list of per-step waypoints (minimum-jerk);
    `cart.both_paths({side: q_goal}, duration, dt)` for simultaneous moves.
  - Gripper: `cart.set_gripper(side, width, force=20, speed=0.05)` (pad gap in m, 0 … `MAX_WIDTH` = 0.096),
    `cart.open(side)`, `cart.grasp(side, force=10.0, speed=0.05)` (speed-limited close; once the fingers stall
    3 mm behind their target the target jumps past closed and `force` (N per finger) caps the squeeze),
    `cart.grasped(side)`, `cart.gripper_width(side)`, `cart.finger_force(side)`.
  - **`cart.update()` once per step before `scene.step()`** — it advances the gripper state machine.
- IPC: `cart.coup_links()` → `("left_tip_left", "left_tip_right", "left_gripper", "right_…")`: the four
  fingertip **box pads** and the two **palm boxes** (the mesh visuals are not collision geometry).
- Export (Blender compositor, `export.py` Recorder):
  - `cart.register(rec)` adds one rigid per (link, material): `robot/<link>` (single-material links) or
    `robot/<link>/<material>` (the `cart` link has seven: white_shell, top_plate, rubber, aluminium, graphite,
    estop_red, camera). Meshes are OBJ in the **link frame, z-up, metres** (no axis swap), under
    `assets/robots/export/<variant>/`.
  - `cart.poses()` → `{name: (pos, quat_wxyz)}`; merge into your frame call:
    `rec.frame(t, rigid={**cart.poses(), **mine}, rods=...)`, or `cart.record(rec, t, rigid=mine, rods=...)`.
  - Materials: arms anodised black (`metal`, (0.045, 0.045, 0.05)), YAM Pro graphite links, dark-grey bases,
    silver wrist (link5) and fingertips, white Flow Base shell, dark top plate, black rubber wheels.
    (I2RT's MJCF rgba values are CAD debug colours, so they are replaced.)

## Model facts

- Joints: `base_x`, `base_y` (slide), `base_yaw` (hinge) — virtual planar base; `<side>_joint1..6` (arm),
  `<side>_joint7/8` (fingers, slide 0 = closed … 0.0475 m = open, along the flange x axis). 19 dofs, 21 links.
- Links: `cart_x`, `cart_y` (massless-ish carriers), `cart` (base + column + deck + mast + camera, 45 kg),
  `<side>_base`, `<side>_link1..5`, `<side>_gripper` (flange + gripper housing), `<side>_tip_left/right`.
- Torque limits = I2RT motor peaks: DM4340 28 N·m (joints 1-3), DM4310 10 N·m (joints 4-6); Big YAM DM6248
  120 N·m (1-2), DM4340 (3-4), DM4310 (5-6). PD gains kp (600, 600, 500, 150, 120, 60), kv (40, 40, 30, 8, 6, 3);
  joint armature 0.05 / 0.01 (reflected rotor inertia). Fingers: kp 2000, kv 20, armature 0.5 kg, damping
  50 N·s/m, and `joint8 = joint7` (one motor + rack drives both, as in I2RT's MJCF), so they move symmetrically.
- Collision: arm links = convex hulls of the I2RT meshes; gripper = palm box + one 8 × 30 × 40 mm box pad per
  fingertip (distal 4 cm). Pad gap = `2 q + 1 mm`: the fingertip meshes overlap 1 mm at q = 0, each pad is
  recessed 1 mm so the pads never touch — **a grasped object shows ≤ 1 mm of fingertip-mesh overlap per side**
  (plus ~1.3 mm rigid-solver contact penetration at 10 N in rigid scenes).
- Contact masks (rigid solver): an arm never collides with itself or the cart; left and right arms collide with
  each other; everything collides with your props (Genesis default masks 0xFFFF). Wheels are visual only (the
  base is held by its virtual joints), so the cart never pushes on the floor.
- The cart footprint is 0.52 m square centred at y = −0.62 (front face y = −0.36); the deck front and the arm base
  plates end at y = −0.36, 1 cm behind the table edge; nothing of the cart is under the table top.
- Overhead camera: the D405 mesh sits on the mast at the I2RT station pose (0.954 m above the arm bases,
  60° down, on the midline); site `top_camera` in the MJCF.

## Mount and reach (`layout.ROBOT_MOUNT`)

`arm_spacing 0.61`, **`arm_base_y −0.395`** (was −0.43; moved forward until the 7 cm-deep base plates end 1 cm
behind the table edge), `arm_base_z 0.75` (level with the table top), `arm_yaw_deg 90`, `cart_center_y −0.62`.

IK reach over `WORK_ZONE` (11 × 7 grid, TCP at table + 2 cm and + 10 cm, fingers along x or y; both arms):

| variant | top-down, either arm | top-down, one arm | with `ik_grasp` tilts ≤ 0.9 rad, either arm |
|---|---|---|---|
| yam | 68 % (all of y ≤ 0.05; y = 0.13 only at the sides; nothing at y = 0.22) | 40 % | **100 %** |
| yam_pro | 68 % | 40 % | **100 %** |
| yam_ultra | 68 % | 40 % | **100 %** |
| big_yam | **100 %** | 81 % | 100 % |

Raising the arm deck 10 cm or moving it 8 cm further forward only took top-down reach from 61 % to 77–83 %
(sweep in `scratch/sweep_yam2.log`); the YAM's 0.51 m upper+fore-arm is the limit. So for YAM / Pro / Ultra:
**put objects that need a vertical grasp at y ≤ 0.05; beyond that use `ik_grasp` (forward-tilted approach)**.
Each arm covers its own half plus ~15 cm past the centre line. Maps: `checks/rigid_<variant>.json`.

## Verified (see `checks/`)

- All four variants load in Genesis 1.3.1 (`gs.cpu`), 19 dofs; home pose held 3 s with PD + gravity
  compensation: TCP drift 0.0 mm, joint error 0.0 rad (PD alone, `gravity_compensation=0`: 3.6 mm yam, 8.0 mm
  big_yam; holding the 45 g cube in IPC the TCP sits 0.6 mm under its target); 0 robot–table contacts at home;
  gripper opens to 96 mm, closes on nothing to the stop.
- Rigid grasp (`test_grasp.py --variant yam`): 4 cm, 45 g cube lifted 9.7 cm, 3.2 mm slip, pad gap 37.7 mm at
  10 N/finger (also yam_ultra, big_yam).
- Export: Genesis link frames match the MuJoCo body frames the OBJs are written in to 1e-7 m.
- IPC grasp + lift works (below).

## IPC notes (what stations must do)

- Pass `ipc=True`. This also monkeypatches a Genesis 1.3.1 coupler bug (`yam_cart.patch_ipc_only_coupling_forces`,
  from the Franka cable work): without it the two-way coupling reaction is applied to free `ipc_only` props too, and
  a cube next to the robot flew to 1e30 m within 2 s.
- Coupler settings that work (`test_grasp.py --ipc`), dt = 0.01:
  ```python
  gs.options.IPCCouplerOptions(contact_d_hat=0.001, contact_resistance=1e7,
      constraint_strength_translation=10.0, constraint_strength_rotation=10.0,   # NOT 100 (see below)
      newton_semi_implicit_enable=False, enable_rigid_rigid_contact=True, enable_rigid_ground_contact=True)
  ```
  The soft-constraint reaction is an explicit spring of stiffness `strength * m_link / dt^2` on the Genesis link.
  At strength 100 (the Franka value) a 71 g YAM fingertip against a rigid cube oscillated and blew up (or, with
  less armature, NaN'd libuipc); at 10 it is stable and the drawn pads sit within 0.5 mm of their libuipc copies.
- Robot-vs-rigid-prop contact needs `enable_rigid_rigid_contact=True`; deformables touch the pads either way.
- Start with fingers OPEN and nothing within `contact_d_hat` of a pad or palm (libuipc aborts init on touching
  geometry). Fingers closing on nothing stop at q = 0 with the pads 1 mm apart.
- Only pads + palms exist in libuipc; arm links pass through IPC objects — keep props out of the arm's path.

## IPC status

**Works.** `test_grasp.py --variant yam --ipc --coupling 10` (gs.gpu, 37 s wall): 45 g, 4 cm `ipc_only` cube on an
`ipc_only` fixed table top, grasped top-down at 10 N/finger and lifted **9.76 cm**, slip 1.9 mm, pad gap 39.1 mm,
cube centred between the pads, no chatter (`checks/grasp_ipc_yam.json`, video `checks/grasp_ipc_yam.mp4`).
Same grasp rigid-only (gs.cpu): lift 9.67 cm, slip 3.2 mm, gap 37.7 mm (rigid-solver penetration ~1.2 mm/side);
also passes for yam_ultra and big_yam (`checks/grasp_rigid_*.json`).
What made it work: finger reflected mass (armature 0.5 kg; a geared rack drive is ~10 kg in reality), 50 N·s/m finger
damping (the force-capped PD gives no damping once saturated), the I2RT joint7 = joint8 equality (fingers were
drifting asymmetrically, 1.4 cm), coupling strength 10, and the ipc_only patch.
IPC was run on `yam` only; the other variants use the identical gripper (rerun:
`test_grasp.py --variant big_yam --ipc --coupling 10` once the SAM3D lock is gone).
