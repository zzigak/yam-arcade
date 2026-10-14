"""Scripted feasibility demo for the mahjong station: the YAM cart picks tiles off the table and stands them in their
rack slots, with IK + joint PD control and physical friction grasps (no teleporting, no welding, no object forces).

Grasp: fingers close across the tile's width (26 mm); the pads sit 6 mm towards the tile's top edge so that, once the
tile stands in the rack, the pads stay above the 6 mm slot dividers. A face-up tile is gripped from above (approach
into the engraved face); turning it to stand in the rack is a 75 deg pitch of the gripper about the finger axis.
The place target for the flange is computed from the tile's pose in the gripper measured after the lift (state
feedback), so in-hand slip does not mis-seat the tile. The tile is lowered into its slot along the backboard.

A face-down tile cannot be stood in the rack with one grasp (the gripper would have to come from behind the backboard),
so it is regrasped: arm A grips it across its width from above (approach into its back), turns it upright and sets it
down standing on the table with its face towards arm B; B grips it from the face side and stands it in the rack.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spec  # noqa: E402
from station import ROOT, TABLE_Z, qmat  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "robots"))
from yam_cart import down_quat, mat_from_quat, min_jerk, quat_from_mat  # noqa: E402

GRASP_Y = 0.006       # pad centre offset towards the tile's top edge (tile +y), m
GRASP_DEPTH = 0.004   # fingertip ends this far above the tile's lower (table-side) face, m
GRIP_FORCE = 6.0      # N per finger; holds 28 g with a margin of ~40 at pad friction 1.0
OPEN_W = 0.042        # pad gap while approaching a tile (26 mm tile + 8 mm each side)
RELEASE_W = 0.034     # pad gap after release in the rack (stays clear of the slot dividers' tops)
FLIP_TILT = -0.35     # face-down regrasp: approach tilted 20 deg (towards the tile's bottom edge) so that, with the
                      # tile standing, the palm stays above the table
FLIP_GRASP_Y = 0.003  # pads 3 mm towards the tile top: when it stands, the pads stay >= 5 mm above the table
FLIP_GRASP_DEPTH = 0.0065  # fingertip ends 6.5 mm above the table (the tilted pads' lower corners clear it)
STANDING_BETA = 0.52  # second grasp: 30 deg below horizontal into the engraved face -> rack placement at 45 deg pitch
U = np.array([0.0, spec.SIN_A, spec.COS_A])      # rack 'up the backboard' (rack frame)
B = np.array([0.0, spec.COS_A, -spec.SIN_A])     # rack 'into the backboard'


def slerp(q0, q1, s):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)


def heading(R):
    """Yaw of the tile's +y (glyph top) about world z, measured from +y (0 = glyph upright for the robot)."""
    return math.atan2(-R[0, 1], R[1, 1])


class IKError(RuntimeError):
    pass


class Runner:
    """Steps the scene with the cart's gripper state machine; on_step(runner) is called after every physics step."""

    def __init__(self, env, on_step=None):
        self.env, self.cart, self.scene = env, env.robot, env.scene
        self.dt = env.dt
        self.t = 0.0
        self.on_step = on_step
        self.events = []
        self.hand_rel = {}
        self.slip_mm = {}

    def log(self, msg, **kw):
        self.events.append(dict(t=round(self.t, 3), msg=msg, **kw))
        print(f"[{self.t:7.3f}] {msg} {kw if kw else ''}", flush=True)

    def step(self, n=1):
        for _ in range(n):
            self.cart.update()
            self.scene.step()
            self.t += self.dt
            if self.on_step is not None:
                self.on_step(self)

    def wait(self, sec):
        self.step(max(1, int(round(sec / self.dt))))

    # ------------------------------------------------------------------------------------------- motion
    def ik(self, side, tcp, quat, seed):
        full = self.cart.q_target.copy()
        full[self.cart.arm_dofs[side]] = seed
        q, err = self.cart.ik(side, tcp, quat=quat, init_qpos=full, return_error=True, max_samples=10)
        if np.linalg.norm(err[:3]) > 0.002 or np.linalg.norm(err[3:]) > 0.02:
            raise IKError(f"IK {side} tcp={np.round(tcp, 4)} err={np.round(err, 4)}")
        return q

    def move_joint(self, side, q_goal, duration):
        for q in self.cart.joint_path(side, q_goal, duration, self.dt):
            self.cart.set_arm_target(side, q)
            self.step()

    def move_line(self, side, tcp0, quat0, tcp1, quat1, duration, n_way=12, max_jump=0.35):
        """Straight TCP line with slerped orientation: IK at n_way waypoints, each seeded by the previous one, then
        joint-space interpolation between waypoints on one min-jerk time scale."""
        seed = self.cart.q_target[self.cart.arm_dofs[side]].copy()
        qs = [seed]
        for k in range(1, n_way + 1):
            s = k / n_way
            q = self.ik(side, (1 - s) * np.asarray(tcp0) + s * np.asarray(tcp1), slerp(quat0, quat1, s), qs[-1])
            if np.abs(q - qs[-1]).max() > max_jump:
                raise IKError(f"IK branch jump {np.abs(q - qs[-1]).max():.2f} rad on line segment {k}")
            qs.append(q)
        qs = np.stack(qs)
        n = max(1, int(round(duration / self.dt)))
        for s in min_jerk(n):
            x = s * n_way
            i = min(int(x), n_way - 1)
            q = qs[i] + (x - i) * (qs[i + 1] - qs[i])
            self.cart.set_arm_target(side, q)
            self.step()

    def move_to(self, side, tcp, quat, duration):
        q = self.ik(side, tcp, quat, self.cart.q_target[self.cart.arm_dofs[side]])
        self.move_joint(side, q, duration)

    def home(self, side, duration=1.6):
        self.move_joint(side, self.cart.home_qpos[self.cart.arm_dofs[side]], duration)

    def release(self, side, width, speed=0.02):
        """Open slowly from where the fingers are now. (yam_cart's set_gripper ramps from the current *target*, which
        after a grasp sits 2 cm past closed, so without this the fingers would not move for ~1.5 s.)"""
        self.cart._grip[side]["target"] = float(self.cart.finger_q(side).mean())
        self.cart.set_gripper(side, width, speed=speed)

    def grasp(self, side, force=GRIP_FORCE, timeout=2.0):
        self.cart.grasp(side, force=force)
        n = 0
        while not self.cart.grasped(side) and n * self.dt < timeout:
            self.step()
            n += 1
        self.wait(0.35)
        return self.cart.grasped(side)

    # -------------------------------------------------------------------------------------------- poses
    def tile_pose(self, tile):
        return np.asarray(tile.entity.get_pos().cpu().numpy(), float), qmat(tile.entity.get_quat().cpu().numpy())

    def flange_pose(self, side):
        tcp, q = self.cart.tcp_pose(side)
        return np.asarray(tcp, float), mat_from_quat(q)

    def seated_pose_world(self, slot):
        rp, rq = self.env.rack_pose()
        Rr = qmat(rq)
        (x, y, z), q = spec.seated_tile_pose_rack(slot)
        return rp + Rr @ np.array([x, y, z]), Rr @ qmat(q), Rr

    # --------------------------------------------------------------------------------------------- task
    def stand_in_rack(self, side, tile, slot):
        """Pick a face-up tile and stand it in rack slot `slot`, facing the robot."""
        cart = self.cart
        p_t, R_t = self.tile_pose(tile)
        assert R_t[2, 2] > 0.9, f"{tile.name} is not face up"
        psi = heading(R_t)
        g = np.array([0.0, GRASP_Y, -0.5 * spec.TILE_T + GRASP_DEPTH])
        pick = p_t + R_t @ g
        q_pick = down_quat(psi, 0.0)
        up = np.array([0, 0, 1.0])
        self.log(f"{side}: pick {tile.name} (heading {math.degrees(psi):.0f} deg) for slot {slot}",
                 pick=np.round(pick, 4).tolist())
        cart.set_gripper(side, OPEN_W)
        self.move_to(side, pick + 0.07 * up, q_pick, 1.6)
        self.move_line(side, pick + 0.07 * up, q_pick, pick, q_pick, 1.0)
        ok = self.grasp(side)
        self.log(f"{side}: grasp {'ok' if ok else 'FAILED'}", width_mm=round(1000 * cart.gripper_width(side), 1))
        self.move_line(side, pick, q_pick, pick + 0.10 * up, q_pick, 1.0)
        self.wait(0.2)
        self.place_in_slot(side, tile, slot)

    def place_in_slot(self, side, tile, slot, transport_time=2.4):
        """The tile is in the `side` gripper, lifted clear of the table: stand it in rack slot `slot`."""
        cart = self.cart
        up = np.array([0, 0, 1.0])
        tcp0, R_f0 = self.flange_pose(side)
        q0 = quat_from_mat(R_f0)
        # tile pose in the TCP frame, measured (state feedback)
        p_t, R_t = self.tile_pose(tile)
        R_rel = R_t.T @ R_f0
        p_rel = R_t.T @ (tcp0 - p_t)
        self.hand_rel[tile.name] = (p_rel.copy(), R_rel.copy())
        # target: seated pose lifted 0.5 mm off the shelf and 0.3 mm off the backboard
        p_s, R_s, Rr = self.seated_pose_world(slot)
        p_s = p_s + Rr @ (0.0005 * U - 0.0003 * B)
        R_goal = R_s @ R_rel
        q_goal = quat_from_mat(R_goal)
        tcp_goal = p_s + R_s @ p_rel
        # lower the tile into the slot along the backboard: 5 cm above, then 8 mm above (3 mm off the backboard),
        # correct the residual tile-pose error measured there (state feedback), then seat it
        off_hi = Rr @ (0.05 * U - 0.003 * B)
        off_lo = Rr @ (0.008 * U - 0.003 * B)
        self.log(f"{side}: transport {tile.name} to slot {slot}", tcp_goal=np.round(tcp_goal, 4).tolist())
        self.move_line(side, tcp0, q0, tcp_goal + off_hi, q_goal, transport_time, n_way=24)
        self.wait(0.25)
        self.move_line(side, tcp_goal + off_hi, q_goal, tcp_goal + off_lo, q_goal, 1.6)
        self.wait(0.3)
        p_t, R_t = self.tile_pose(tile)
        # in-hand slip since the lift (tile pose relative to the TCP)
        tcp_now_meas, R_f_now = self.flange_pose(side)
        slip = np.linalg.norm(R_t.T @ (tcp_now_meas - p_t) - p_rel)
        self.slip_mm[tile.name] = 1000 * float(slip)
        err = (p_s + off_lo) - p_t
        self.log(f"{side}: tile error 8 mm above the seat", err_mm=np.round(1000 * err, 2).tolist(),
                 in_hand_slip_mm=round(1000 * float(slip), 2))
        tcp_now = tcp_goal + off_lo
        self.move_line(side, tcp_now, q_goal, tcp_goal + err, q_goal, 1.0)
        self.wait(0.2)
        self.release(side, RELEASE_W)
        self.wait(0.7)
        tcp_goal = tcp_goal + err
        self.log(f"{side}: released {tile.name}")
        approach = R_goal[:, 2]
        self.move_line(side, tcp_goal, q_goal, tcp_goal - 0.06 * approach + 0.01 * up, q_goal, 1.0)
        cart.set_gripper(side, OPEN_W)
        self.home(side)

    def flip_to_standing(self, side, tile, spot, face_dir, tilt=FLIP_TILT):
        """Regrasp step for a face-down tile: grip it across its width from above with the approach tilted `tilt` rad
        (about the tile's x axis, towards its bottom edge) into its back, and set it down standing on its bottom edge
        at `spot` (x, y) with the engraved face towards `face_dir` (horizontal unit vector)."""
        cart = self.cart
        up = np.array([0, 0, 1.0])
        p_t, R_t = self.tile_pose(tile)
        assert R_t[2, 2] < -0.9, f"{tile.name} is not face down"
        # gripper in the tile frame: approach a into the back, fingers along -tile x (the wrist-roll-feasible sign)
        a = np.array([0.0, math.sin(tilt), math.cos(tilt)])
        x = np.array([-1.0, 0.0, 0.0])
        G = np.stack([x, np.cross(a, x), a], 1)
        g = np.array([0.0, FLIP_GRASP_Y, 0.5 * spec.TILE_T - FLIP_GRASP_DEPTH])
        pick = p_t + R_t @ g
        R_pick = R_t @ G
        q_pick = quat_from_mat(R_pick)
        a_w = R_pick[:, 2]
        self.log(f"{side}: pick face-down {tile.name} (heading {math.degrees(heading(R_t)):.0f} deg) to stand it up",
                 pick=np.round(pick, 4).tolist())
        cart.set_gripper(side, OPEN_W)
        self.move_to(side, pick - 0.07 * a_w, q_pick, 1.8)
        self.move_line(side, pick - 0.07 * a_w, q_pick, pick, q_pick, 1.0)
        ok = self.grasp(side)
        self.log(f"{side}: grasp {'ok' if ok else 'FAILED'}", width_mm=round(1000 * cart.gripper_width(side), 1))
        self.move_line(side, pick, q_pick, pick + 0.10 * up, q_pick, 1.0)
        self.wait(0.2)
        tcp0, R_f0 = self.flange_pose(side)
        p_t, R_t = self.tile_pose(tile)
        R_rel, p_rel = R_t.T @ R_f0, R_t.T @ (tcp0 - p_t)
        # standing pose: tile +z (face) = face_dir, tile +y (top) = up, bottom face 1 mm above the table
        d = np.asarray(face_dir, float)
        R_m = np.stack([np.cross(up, d), up, d], 1)
        p_m = np.array([spot[0], spot[1], TABLE_Z + 0.5 * spec.TILE_H + 0.0003])
        R_goal = R_m @ R_rel
        q_goal = quat_from_mat(R_goal)
        tcp_goal = p_m + R_m @ p_rel
        self.log(f"{side}: turn {tile.name} upright", tcp_goal=np.round(tcp_goal, 4).tolist())
        self.move_line(side, tcp0, quat_from_mat(R_f0), tcp_goal + 0.06 * up, q_goal, 2.6, n_way=26)
        self.wait(0.2)
        # lower to 8 mm above the table, correct the measured tile error, then set it down 0.3 mm above the table
        self.move_line(side, tcp_goal + 0.06 * up, q_goal, tcp_goal + 0.008 * up, q_goal, 1.2)
        self.wait(0.3)
        p_t, _ = self.tile_pose(tile)
        err = (p_m + 0.008 * up) - p_t
        self.log(f"{side}: tile error 8 mm above the table", err_mm=np.round(1000 * err, 2).tolist())
        self.move_line(side, tcp_goal + 0.008 * up, q_goal, tcp_goal + err, q_goal, 0.8)
        tcp_goal = tcp_goal + err
        self.wait(0.3)
        self.release(side, 0.036)
        self.wait(0.7)
        self.log(f"{side}: released {tile.name} standing")
        aw = R_goal[:, 2]
        self.move_line(side, tcp_goal, q_goal, tcp_goal - 0.06 * aw + 0.02 * up, q_goal, 1.0)
        cart.set_gripper(side, OPEN_W)
        self.home(side)

    def pick_standing(self, side, tile, beta=STANDING_BETA):
        """Grip a tile that stands on its bottom edge across its width, approaching its engraved face from above at
        `beta` rad below horizontal, and lift it 8 cm."""
        cart = self.cart
        up = np.array([0, 0, 1.0])
        p_t, R_t = self.tile_pose(tile)
        a = np.array([0.0, math.sin(-beta), -math.cos(-beta)])
        x = np.array([1.0, 0.0, 0.0])
        G = np.stack([x, np.cross(a, x), a], 1)
        g = np.array([0.0, GRASP_Y, -0.5 * spec.TILE_T + GRASP_DEPTH])
        grasp = p_t + R_t @ g
        R_g = R_t @ G
        q_g = quat_from_mat(R_g)
        a_w = R_g[:, 2]
        self.log(f"{side}: grip standing {tile.name} from its face side", grasp=np.round(grasp, 4).tolist(),
                 approach=np.round(a_w, 3).tolist())
        cart.set_gripper(side, OPEN_W)
        self.move_to(side, grasp - 0.07 * a_w, q_g, 1.8)
        self.move_line(side, grasp - 0.07 * a_w, q_g, grasp, q_g, 1.2)
        ok = self.grasp(side)
        self.log(f"{side}: grasp {'ok' if ok else 'FAILED'}", width_mm=round(1000 * cart.gripper_width(side), 1))
        self.move_line(side, grasp, q_g, grasp + 0.08 * up, q_g, 1.0)
        self.wait(0.2)

    def flip_and_rack(self, side_a, side_b, tile, slot, spot, face_dir):
        """Face-down tile into the rack: arm A stands it up on the table facing arm B, B regrasps it from its face
        side and stands it in the rack."""
        self.flip_to_standing(side_a, tile, spot, face_dir)
        self.wait(0.5)
        self.pick_standing(side_b, tile)
        self.place_in_slot(side_b, tile, slot, transport_time=2.8)
