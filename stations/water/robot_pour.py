"""Big YAM cart for the water station: the right arm grasps the pitcher's handle and pours.

Grasp: horizontal approach along station +y (from the robot), fingers closing along station x across the vertical
grip bar of the D-handle - the inner finger goes through the handle loop, like a hand.  The pour tilt is then a roll
about the approach axis, i.e. mostly the wrist roll (joint6), and the bar is held between the pads by normal force
(not by friction torque about the pinch axis, which would be the weak direction).

RobotDriver implements the PourController driver interface:
  prepare()  generator: reach, approach, close; records the TCP pose in the pitcher frame
  command()  pitcher target pose -> TCP target (rigid grasp transform) -> IK -> arm joint PD targets
  release()  generator: open, retreat, go home
  update()   per-step housekeeping (gripper state machine)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from station import ROOT, quat_to_R  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "robots"))
import yam_cart  # noqa: E402
from yam_cart import add_cart, quat_from_mat, mat_from_quat  # noqa: E402

SIDE = "right"
GRIP_LOCAL = np.array([-0.110, 0.0, 0.140])   # grip-bar centreline point in the pitcher frame
TCP_PAST_BAR = 0.018                          # TCP (fingertip end) this far past the bar centre along the approach
PRE_BACK = 0.10                               # pre-grasp stand-off along -approach
OPEN_WIDTH = 0.060                            # pad gap while approaching (bar is 22 mm wide, 43 mm loop gap)
# Clamp force per finger.  I2RT's driver ships a 50 N software force limiter (get_robot(limit_gripper_force=50));
# the DM4310 behind it (10 N m peak, 6.57 rad of motor per 96 mm stroke = 14.6 mm effective radius) can push
# ~680 N, so 80 N is within the hardware.  Needed because the full pitcher's weight moment about the pour axis
# (~1.7 N m) must be held by the pads' normal-force couple: 80 N x 46 mm contact = 3.7 N m (50 N gave 2.3 N m and
# the pitcher rotated in the grip under the travel accelerations in the dry rehearsal).
GRIP_FORCE = 80.0
# Approach pitched down 50 deg from horizontal (still perpendicular to the handle loop): a vertical bar then lies
# along the pad diagonal, 47 mm of contact instead of the pad's 30 mm height, and the weight moment of the full
# pitcher about the pour axis (~1.8 N m) stays below what a 50 N clamp resists (50 N x 47 mm = 2.3 N m).
APPROACH_PITCH = np.radians(50.0)


class RobotHandle:
    def __init__(self, cart):
        self.cart = cart
        self.log = []
        self._parts = None

    def poses(self):
        """{robot/<link>[/<material>]: (pos, quat)} for every exported robot part (same names as Cart.register)."""
        from genesis.utils.misc import tensor_to_array
        if self._parts is None:
            self._parts = self.cart.export_parts()
        out, cache = {}, {}
        for obj, link, _, _ in self._parts:
            if link not in cache:
                lk = self.cart.entity.get_link(link)
                cache[link] = (tensor_to_array(lk.get_pos()), tensor_to_array(lk.get_quat()))
            out[obj] = cache[link]
        return out

    def summary(self):
        return dict(variant="big_yam", side=SIDE, grip_force_N=GRIP_FORCE, log=self.log[-20:])


def builder(scene):
    import genesis as gs
    cart = add_cart(scene, gs, "big_yam", ipc=False)
    return RobotHandle(cart)


def T_from(pos, quat):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(quat)
    T[:3, 3] = pos
    return T


def _rotvec(R):
    """Rotation matrix -> axis * angle."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(c)
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(ang))
    return v * ang


def _expmap(w):
    ang = np.linalg.norm(w)
    if ang < 1e-12:
        return np.eye(3)
    k = w / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def _blend(Ta, Tb, w):
    """Pose interpolation (1-w) Ta + w Tb with the rotation re-orthonormalised (small steps)."""
    T = (1.0 - w) * Ta + w * Tb
    U, _, Vt = np.linalg.svd(T[:3, :3])
    T[:3, :3] = U @ Vt
    T[3] = (0.0, 0.0, 0.0, 1.0)
    return T


class RobotDriver:
    def __init__(self, env, ik_every=2):
        self.env = env
        self.cart = env.robot.cart
        self.scene = env.scene
        self.dt = env.scene.sim_options.dt
        self.ik_every = ik_every
        self.k = 0
        self.T_tcp_in_pitcher = None
        self.err_log = []
        self.cart.after_build()

    # ---------------------------------------------------------------- helpers
    def grasp_frame(self, approach=np.array([0.0, 1.0, 0.0]), close_x_sign=1.0):
        """Flange rotation: z = approach, x = closing axis (station +-x)."""
        z = approach / np.linalg.norm(approach)
        x = np.array([close_x_sign, 0.0, 0.0])
        x = x - z * (x @ z)
        x /= np.linalg.norm(x)
        return np.stack([x, np.cross(z, x), z], axis=1)

    def ik(self, pos, R, seed=None):
        q, err = self.cart.ik(SIDE, pos, quat=quat_from_mat(R), init_qpos=seed, return_error=True)
        return q, err

    def _goto_joint(self, q_goal, duration):
        for qk in self.cart.joint_path(SIDE, q_goal, duration, self.dt):
            self.cart.set_arm_target(SIDE, qk)
            yield

    def _cartesian(self, p0, p1, R, duration):
        n = max(1, int(round(duration / self.dt)))
        for k in range(1, n + 1):
            s = k / n
            s = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
            if k % self.ik_every == 0 or k == n:
                q, _ = self.ik(p0 + s * (p1 - p0), R)
                self.cart.set_arm_target(SIDE, q)
            yield

    # ---------------------------------------------------------------- driver interface
    def prepare(self):
        pos, quat = self.env.pose("pitcher")
        Rp = quat_to_R(quat)
        grip = pos + Rp @ GRIP_LOCAL
        approach = np.array([0.0, np.cos(APPROACH_PITCH), -np.sin(APPROACH_PITCH)])
        tcp = grip + TCP_PAST_BAR * approach
        pre = tcp - PRE_BACK * approach
        # choose the closing-axis sign whose IK converges (position AND orientation) at the grasp and over the pour,
        # which rotates the grasp about station -y (the pitcher's local y) by up to ~110 deg
        best = None
        for sign in (1.0, -1.0):
            R = self.grasp_frame(approach, sign)
            q_pre, e_pre = self.ik(pre, R)
            q_tcp, e_tcp = self.ik(tcp, R, seed=None)
            score = lambda e: np.linalg.norm(e[:3]) + 0.05 * np.linalg.norm(e[3:])   # 5 cm per rad
            worst = max(score(e_pre), score(e_tcp))
            for beta in np.radians([40.0, 80.0, 110.0]):
                c, s_ = np.cos(beta), np.sin(beta)
                Ry = np.array([[c, 0, -s_], [0, 1, 0], [s_, 0, c]])    # rotation about -y by beta
                _, e = self.ik(tcp + np.array([-0.05, 0.0, 0.15]), Ry @ R)
                worst = max(worst, score(e))
            self.env.robot.log.append(f"grasp option sign={sign:+.0f} j6={q_tcp[5]:.2f} worst_ik_score={worst * 1e3:.1f}mm")
            if best is None or worst < best[0]:
                best = (worst, sign, R, q_pre, q_tcp)
        _, sign, R, q_pre, q_tcp = best
        self.R_grasp0 = R
        self.cart.set_gripper(SIDE, OPEN_WIDTH, speed=0.08)
        above = pre + np.array([0.0, -0.03, 0.08])
        q_above, _ = self.ik(above, R)
        yield from self._goto_joint(q_above, 1.6)
        yield from self._cartesian(above, pre, R, 0.6)
        yield from self._cartesian(pre, tcp, R, 1.0)
        for _ in range(int(0.2 / self.dt)):
            yield
        self.cart.grasp(SIDE, force=GRIP_FORCE, speed=0.04, squeeze_q=-0.04)
        n = 0
        while not self.cart.grasped(SIDE) and n < int(2.0 / self.dt):
            n += 1
            yield
        for _ in range(int(0.4 / self.dt)):
            yield
        tcp_pos, fq = self.cart.tcp_pose(SIDE)
        pos, quat = self.env.pose("pitcher")
        self.T_tcp_in_pitcher = np.linalg.inv(T_from(pos, quat)) @ T_from(tcp_pos, fq)
        self.env.robot.log.append(f"grasped={self.cart.grasped(SIDE)} width={self.cart.gripper_width(SIDE) * 1e3:.1f}mm "
                                  f"after {n * self.dt:.2f}s")
        self.offset = np.zeros(3)
        self.rot_offset = np.zeros(3)

    def command(self, base_pos, base_quat, dt):
        self.k += 1
        if self.k % self.ik_every:
            return
        # Object-pose feedback (a stand-in for visual tracking of the pitcher): re-measure where the pitcher sits in
        # the hand, so millimetre-scale slip / rotation in the grasp does not accumulate into the commanded pose.
        pos, quat = self.env.pose("pitcher")
        tcp_pos, fq = self.cart.tcp_pose(SIDE)
        T_meas = np.linalg.inv(T_from(pos, quat)) @ T_from(tcp_pos, fq)
        self.T_tcp_in_pitcher = _blend(self.T_tcp_in_pitcher, T_meas, 0.05)
        T = T_from(base_pos, base_quat) @ self.T_tcp_in_pitcher
        # slow integral correction of the steady-state sag (payload on PD joints) using the pitcher's actual pose,
        # in position and orientation (the wrist joints sag ~3 deg under the full pitcher's moment)
        err = base_pos - pos
        self.offset += 2.0 * err * dt * self.ik_every
        self.offset = np.clip(self.offset, -0.02, 0.02)
        e_rot = _rotvec(quat_to_R(base_quat) @ quat_to_R(quat).T)
        self.rot_offset += 2.0 * e_rot * dt * self.ik_every
        n = np.linalg.norm(self.rot_offset)
        if n > 0.1:
            self.rot_offset *= 0.1 / n
        q, e = self.ik(T[:3, 3] + self.offset, _expmap(self.rot_offset) @ T[:3, :3])
        self.cart.set_arm_target(SIDE, q)
        self.err_log.append((np.linalg.norm(err), np.linalg.norm(e[:3])))

    def release(self):
        # Open from where the fingers ARE: Cart.set_gripper ramps its target from the previous target, which after a
        # force-capped grasp sits 4 cm past closed, so the fingers would keep pinching for ~0.7 s.
        self.cart._grip[SIDE]["target"] = float(self.cart.finger_q(SIDE).mean())
        self.cart.open(SIDE, speed=0.06)
        n = 0
        while self.cart.gripper_width(SIDE) < 0.045 and n < int(1.5 / self.dt):
            n += 1
            yield
        for _ in range(int(0.2 / self.dt)):
            yield
        tcp, fq = self.cart.tcp_pose(SIDE)
        R = mat_from_quat(fq)
        yield from self._cartesian(tcp, tcp - PRE_BACK * R[:, 2], R, 1.0)
        yield from self._goto_joint(self.cart.home_qpos[self.cart.arm_dofs[SIDE]], 1.5)

    def update(self):
        self.cart.update()
