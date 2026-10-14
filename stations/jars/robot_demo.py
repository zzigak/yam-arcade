"""Robot demo: the wheeled bimanual YAM Pro screws each lid onto the jar it fits (called by `run.py robot`).

Per jar (order mason, jam, spice):
  1. drive the base so the lid lies at x = +0.15 m from the cart centre; right arm picks the lid top-down
  2. lift it above every jar, drive the base so the jar sits at x = +0.05 m from the cart centre
  3. left arm holds the jar body with a horizontal grasp from the robot side (low, out of the way of the twisting hand)
  4. right arm lowers the lid onto the mouth (lid yaw chosen so its thread lands one turn out, see NOTES.md)
  5. wrist strokes: joint 6 turns -1.9 -> +1.9 rad (218 deg clockwise from above) with a torque-capped wrist (the cap
     is the tightening torque), the hand following the thread down at one pitch per turn; open, turn back, regrasp;
     repeat until the wrist stalls (lid seated)
  6. release both hands, back off, go home.
Only joint PD targets (IK + min-jerk joint paths), the base PD target and gripper commands are used. Nothing on the
jars or lids is ever set; contact, friction and the threads do the rest.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "robots"))

ORDER = ("mason", "jam", "spice")
X_PICK = 0.15  # lid x relative to the cart centre when picking
X_WORK = 0.05  # jar x relative to the cart centre when screwing
Q6_A, Q6_B = -1.9, 1.9  # wrist stroke (joint 6 range is +-2.094)
STROKE_W = 1.6  # rad/s wrist speed during a stroke
# N m, joint-6 motor torque cap during strokes. It must cover the YAM wrist's joint damping (0.1 N m s/rad in the I2RT
# MJCF: 0.16 N m at 1.6 rad/s; a 0.10 cap halved the stroke). The stroke ends when the wrist stalls or, if the pads
# would rather slip round the seated lid, as soon as the lid stops following the wrist (>0.12 rad behind).
WRIST_CAP = dict(spice=0.25, jam=0.25, mason=0.25)
LID_GRIP = 8.0  # N per finger on the lid
JAR_GRIP = 15.0  # N per finger on the jar
CARRY_Z = 0.145  # TCP height above the table when carrying a lid (clears the tallest jar + lid)
HOLD_Z = 0.06  # TCP height above the table for the jar hold
N_UP = 1.0  # turns out from seated at which the lid is landed on the thread
# Solver settings for the robot run (scratch/gmin_*.log): with a gripper squeezing a part, the twist-test settings
# (friction smoothing 1e-4 m/s, Newton 2e-4 m/s) cost 5-36 s/step; these cost ~0.35 s/step. Translation coupling 10 as
# validated by robots/ (100 diverges). Rotation coupling 40: with the centroid patch it is stable and cuts the libuipc
# fingertip copies' lag behind Genesis's fingertips from 11.2 to 3.6 mm max (0.9 mm p95 while twisting;
# scratch/rtw_jam_r10.log vs rtw_jam_r40.log); translation 20 changed nothing.
ROBOT_COUPLER = dict(constraint_strength_translation=10.0, constraint_strength_rotation=40.0,
                     newton_tolerance=1e-3, contact_eps_velocity=1e-3)


def min_jerk(n):
    s = np.linspace(0.0, 1.0, n + 1)[1:]
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class Demo:
    def __init__(self, env, ses, S):
        self.env, self.ses, self.S = env, ses, S
        self.cart = env.robot
        self.dt = env.dt
        self.base = np.zeros(3)
        self.TZ = S.TABLE_Z
        self.dof6 = int(self.cart.arm_dofs["right"][5])
        self.events = []

    # ------------------------------------------------------------ primitives
    def log(self, msg):
        self.events.append(dict(t=round(self.env.t, 3), msg=msg))
        print(f"[{self.env.t:7.3f}] {msg}", flush=True)

    def wait(self, dur):
        self.ses.step(max(1, int(round(dur / self.dt))))

    def ik(self, side, pos, base_x=None, yaw=0.0, tilt=0.0, tol=2e-3, seed=None):
        init = self.cart.q_target.copy()
        init[self.cart.base_dofs] = (self.base[0] if base_x is None else base_x, 0.0, 0.0)
        if seed is not None:
            init[self.cart.arm_dofs[side]] = seed
        q, err = self.cart.ik(side, pos, yaw=yaw, tilt=tilt, init_qpos=init, return_error=True)
        if np.linalg.norm(err[:3]) > tol or np.linalg.norm(err[3:]) > 0.03:
            raise RuntimeError(f"IK failed for {side} at {np.round(pos, 4)} yaw {yaw:.2f} tilt {tilt:.2f}: err {err}")
        return q

    def go(self, arms, dur, base_x=None):
        """Min-jerk joint-space move of the given arms {side: q6} and optionally the base x, together."""
        n = max(1, int(round(dur / self.dt)))
        s = min_jerk(n)
        q0 = {side: self.cart.q_target[self.cart.arm_dofs[side]].copy() for side in arms}
        b0 = self.base[0]
        for k in range(n):
            for side, q in arms.items():
                self.cart.set_arm_target(side, q0[side] + s[k] * (np.asarray(q) - q0[side]))
            if base_x is not None:
                self.cart.set_base_target(b0 + s[k] * (base_x - b0), 0.0, 0.0)
            self.ses.step(1)
        if base_x is not None:
            self.base[0] = base_x

    def go_path(self, side, start, end, dur, n_via=8, base_x=None):
        self.go_paths({side: (start, end)}, dur, n_via=n_via, base_x=base_x)

    def go_paths(self, moves, dur, n_via=8, base_x=None):
        """Cartesian straight-line moves of one or both hands at once: moves = {side: (start, end)}, start/end =
        (pos, yaw, tilt). IK at n_via waypoints (each seeded by the previous one), then a min-jerk timed walk through
        them. Keeps long moves away from jars and lids."""
        paths = {}
        for side, (start, end) in moves.items():
            qs = [self.cart.q_target[self.cart.arm_dofs[side]].copy()]
            for u in np.linspace(0, 1, n_via + 1)[1:]:
                pos = (1 - u) * np.asarray(start[0]) + u * np.asarray(end[0])
                yaw = (1 - u) * start[1] + u * end[1]
                tilt = (1 - u) * start[2] + u * end[2]
                qs.append(self.ik(side, pos, base_x=base_x, yaw=yaw, tilt=tilt, seed=qs[-1], tol=4e-3))
            paths[side] = np.array(qs)
        n = max(1, int(round(dur / self.dt)))
        b0 = self.base[0]
        for k, sk in enumerate(min_jerk(n)):
            for side, qs in paths.items():
                x = sk * (len(qs) - 1)
                i = min(int(x), len(qs) - 2)
                self.cart.set_arm_target(side, qs[i] + (x - i) * (qs[i + 1] - qs[i]))
            if base_x is not None:
                self.cart.set_base_target(b0 + sk * (base_x - b0), 0.0, 0.0)
            self.ses.step(1)
        if base_x is not None:
            self.base[0] = base_x

    def home_pose(self, side):
        """(pos, yaw, tilt) of the hand at the cart's home pose, at the current base offset."""
        from yam_cart import HOME_TCP

        p = np.array(HOME_TCP[side], float)
        p[0] += self.base[0]
        return p, 0.0, 0.0

    def grip_until(self, side, timeout):
        n = int(round(timeout / self.dt))
        for _ in range(n):
            self.ses.step(1)
            if self.cart.grasped(side):
                break
        self.wait(0.15)

    def open_wait(self, widths, timeout=1.2, speed=0.2):
        """Open hands {side: pad gap} and step until every pad gap has really opened (>= goal - 2 mm). After a grasp the
        finger target sits past closed (squeeze), so a fixed wait is not enough: an early move dragged the lid back
        0.108 turn and the jar 22 mm (scratch/rtwist_mason4.log)."""
        for side, w in widths.items():
            self.cart.set_gripper(side, w, speed=speed)
        for _ in range(int(round(timeout / self.dt))):
            self.ses.step(1)
            if all(self.cart.gripper_width(s) >= w - 0.002 for s, w in widths.items()):
                break
        self.wait(0.05)

    def tcp(self, side):
        return self.cart.tcp_pose(side)[0]

    # ------------------------------------------------------------ task
    def lid_pose(self, name):
        p, R = self.env.pose(self.env.lids[name])
        return p, R

    def jar_pose(self, name):
        return self.env.pose(self.env.jars[name])

    def home(self, dur=1.2, base_x=None):
        h = self.cart.home_qpos
        self.go({s: h[self.cart.arm_dofs[s]] for s in ("left", "right")}, dur, base_x=base_x)

    def plan(self, name):
        S, TZ = self.S, self.TZ
        sp = S.part(name)
        lid_xy = np.array(S.LID_XY[name][0])
        jar_xy = self.jar_pose(name)[0][:2]
        c = dict(name=name, sp=sp, P=sp["pitch"], lid_xy=lid_xy, jar_xy=jar_xy, r_lid=sp["lid_r_outer_mm"] * 1e-3,
                 r_jar=sp["spec"]["r_body"] * 1e-3, b_pick=lid_xy[0] - X_PICK, b_work=jar_xy[0] - X_WORK)
        # wrist yaw: q6 = q6(yaw 0) - yaw. The lid lies at yaw 0; pick it with the hand yaw that puts q6 at Q6_A over
        # the jar, so the lid arrives at yaw 0 (mod 2 pi) relative to the jar (thread lands N_UP turns out).
        c["q_twist0"] = self.ik("right", (jar_xy[0], jar_xy[1], TZ + 0.10), base_x=c["b_work"], yaw=0.0)
        c["yaw"] = float(c["q_twist0"][5] - Q6_A)
        return c

    def pick_carry(self, c):
        cart, TZ, name, yaw = self.cart, self.TZ, c["name"], c["yaw"]
        lid_xy, jar_xy, b_pick, b_work = c["lid_xy"], c["jar_xy"], c["b_pick"], c["b_work"]
        self.log(f"{name}: drive to lid (base x {b_pick:+.3f}), hand yaw {yaw:+.2f} rad")
        q_above = self.ik("right", (lid_xy[0], lid_xy[1], TZ + 0.08), base_x=b_pick, yaw=yaw)
        self.go({"right": q_above}, 1.6, base_x=b_pick)
        cart.set_gripper("right", min(2 * c["r_lid"] + 0.024, 0.096))
        q_pick = self.ik("right", (lid_xy[0], lid_xy[1], TZ + 0.002), yaw=yaw)
        self.go({"right": q_pick}, 0.9)
        self.wait(0.1)
        cart.grasp("right", force=LID_GRIP)
        self.grip_until("right", 0.8)
        lp, _ = self.lid_pose(name)
        c["off"] = off = lp[2] - self.tcp("right")[2]  # lid origin above the TCP while gripped
        self.log(f"{name}: lid gripped, width {cart.gripper_width('right') * 1e3:.1f} mm, lid-TCP dz {off * 1e3:.2f} mm")
        q_lift = self.ik("right", (lid_xy[0], lid_xy[1], TZ + CARRY_Z), yaw=yaw)
        self.go({"right": q_lift}, 0.9)
        self.log(f"{name}: drive to jar (base x {b_work:+.3f})")
        q_carry = self.ik("right", (lid_xy[0] + (b_work - b_pick), lid_xy[1], TZ + CARRY_Z), base_x=b_work, yaw=yaw)
        self.go({"right": q_carry}, 1.4, base_x=b_work)

    def hold_jar(self, c, from_home=True):
        cart, TZ, name, jar_xy = self.cart, self.TZ, c["name"], c["jar_xy"]
        cart.set_gripper("left", min(2 * c["r_jar"] + 0.012, 0.096))
        if from_home:
            # together: right hand carries the lid in a straight line to above the jar mouth (above every jar), left
            # hand goes high from home, turning to the horizontal side grasp, to in front of the jar
            lid_xy, b_pick, b_work, yaw = c["lid_xy"], c["b_pick"], c["b_work"], c["yaw"]
            self.go_paths({"right": (((lid_xy[0] + (b_work - b_pick), lid_xy[1], TZ + CARRY_Z), yaw, 0.0),
                                     ((jar_xy[0], jar_xy[1], TZ + CARRY_Z), yaw, 0.0)),
                           "left": (self.home_pose("left"),
                                    ((jar_xy[0], jar_xy[1] - 0.12, TZ + HOLD_Z + 0.04), 0.0, math.pi / 2))}, 1.6)
        q_pre = self.ik("left", (jar_xy[0], jar_xy[1] - 0.10, TZ + HOLD_Z), tilt=math.pi / 2)
        self.go({"left": q_pre}, 0.7)
        q_hold = self.ik("left", (jar_xy[0], jar_xy[1] + 0.015, TZ + HOLD_Z), tilt=math.pi / 2)
        self.go({"left": q_hold}, 0.9)
        cart.grasp("left", force=JAR_GRIP)
        self.grip_until("left", 0.8)
        self.log(f"{name}: jar held, width {cart.gripper_width('left') * 1e3:.1f} mm")

    def land(self, c):
        """Lower the lid onto the mouth: lid origin at rim + n_up pitches + 0.4 mm, n_up from the lid's yaw."""
        S, name, jar_xy, yaw, P, off = self.S, c["name"], c["jar_xy"], c["yaw"], c["P"], c["off"]
        jar_p, Rj = self.jar_pose(name)
        lp, Rl = self.lid_pose(name)
        Rrel = Rj.T @ Rl
        q_land = self.ik("right", (jar_xy[0], jar_xy[1], jar_p[2] + c["sp"]["rim_z"] + N_UP * P - off), yaw=yaw)
        yaw_land = math.atan2(Rrel[1, 0], Rrel[0, 0]) - (q_land[5] - float(self.cart.qpos()[self.dof6]))
        yaw_land = (yaw_land + np.pi) % (2 * np.pi) - np.pi
        n_up = N_UP + yaw_land / (2 * np.pi)
        z_tcp = jar_p[2] + c["sp"]["rim_z"] + n_up * P + 0.4e-3 - off
        q_land = self.ik("right", (jar_xy[0], jar_xy[1], z_tcp), yaw=yaw)
        self.log(f"{name}: lid yaw rel jar at landing {math.degrees(yaw_land):+.1f} deg -> thread lands {n_up:.3f} turns out")
        self.go({"right": q_land}, 1.0)
        self.wait(0.2)
        s = self.env.lid_on_jar(name, name)
        self.log(f"{name}: lid on mouth: h {s['height_above_rim'] * 1e3:.3f} mm, xy {s['offset_xy'] * 1e3:.2f} mm, "
                 f"tilt {s['tilt_deg']:.2f} deg")
        return z_tcp

    def twist_on(self, c, z_tcp):
        name, P, yaw, off = c["name"], c["P"], c["yaw"], c["off"]
        jar_p, _ = self.jar_pose(name)
        z_seat_tcp = jar_p[2] + c["sp"]["rim_z"] + 0.15e-3 - off
        z0 = z_tcp
        for k in range(4):
            stalled, dturn = self.stroke(name, z0, z_seat_tcp, P, yaw)
            s = self.env.lid_on_jar(name, name)
            self.log(f"{name}: stroke {k + 1}: wrist turned {dturn:.3f} turn, lid h {s['height_above_rim'] * 1e3:.3f} mm, "
                     f"xy {s['offset_xy'] * 1e3:.2f} mm, tilt {s['tilt_deg']:.2f}, engaged {self.env.engaged_turn[(name, name)]:.3f} turn, "
                     f"stalled {stalled}")
            if stalled or s["height_above_rim"] < 0.5e-3:
                break
            z0 = z0 - P * dturn
            self.regrasp(name, yaw, max(z0, z_seat_tcp), c["r_lid"])

    def release(self, c):
        cart, TZ, jar_xy = self.cart, self.TZ, c["jar_xy"]
        self.open_wait({"right": min(2 * c["r_lid"] + 0.020, 0.096), "left": min(2 * c["r_jar"] + 0.010, 0.096)})
        tcp = self.tcp("right")
        yaw_now = float(c["q_twist0"][5] - self.cart.q_target[self.dof6])
        up = ((tcp[0], tcp[1], tcp[2] + 0.025), yaw_now, 0.0)  # +6 cm is out of reach above the mason jar
        q_up = self.ik("right", up[0], yaw=yaw_now)
        back = ((jar_xy[0], jar_xy[1] - 0.11, TZ + HOLD_Z + 0.01), 0.0, math.pi / 2)
        q_back = self.ik("left", back[0], tilt=math.pi / 2)
        self.go({"right": q_up, "left": q_back}, 0.8)
        self.go_paths({"left": (back, self.home_pose("left")), "right": (up, self.home_pose("right"))}, 1.4)
        self.log(f"{c['name']}: done; success so far {json.dumps(self.env.success()[c['name']])}")

    def screw(self, name):
        c = self.plan(name)
        self.pick_carry(c)
        self.hold_jar(c)
        z_tcp = self.land(c)
        self.twist_on(c, z_tcp)
        self.release(c)

    def stroke(self, name, z_start, z_seat, P, yaw):
        """Turn joint 6 from Q6_A to Q6_B with velocity feed-forward and a torque cap; the hand descends one pitch per
        turn (never below the seated height). Stops early when the wrist stalls (lid seated). Returns (stalled, turns)."""
        cart, e = self.cart, self.cart.entity
        side = "right"
        jar_xy = self.jar_pose(name)[0][:2]
        total = (Q6_B - Q6_A) / (2 * np.pi)
        z_end = max(z_start - P * total, z_seat)
        q_a = self.ik(side, (jar_xy[0], jar_xy[1], z_start), yaw=yaw)
        q_b = self.ik(side, (jar_xy[0], jar_xy[1], z_end), yaw=yaw)
        cap = WRIST_CAP[name]
        e.set_dofs_force_range([-cap], [cap], [self.dof6])
        t_ramp = 0.5  # wrist inertia incl. rotor armature ~0.01 kg m2: 1.6 rad/s in 0.5 s needs 0.03 N m
        T = (Q6_B - Q6_A) / STROKE_W + t_ramp
        n = int(round(T / self.dt))
        q6_ref = Q6_A
        stall_n = 0
        stalled = False
        q6_start = q6_prev = float(cart.qpos()[self.dof6])
        lid_yaw0 = self.env.rel_yaw[(name, name)]
        for k in range(n):
            t = (k + 1) * self.dt
            w = max(0.0, STROKE_W * min(1.0, t / t_ramp, (T - t) / t_ramp))
            q6_ref = min(q6_ref + w * self.dt, Q6_B)
            z_ref = max(z_start - P * (q6_ref - Q6_A) / (2 * np.pi), z_seat)  # follow the thread, not below the seat
            frac = 1.0 if z_start - z_end < 1e-6 else (z_start - z_ref) / (z_start - z_end)
            q_arm = q_a + frac * (q_b - q_a)
            q_arm[5] = q6_ref
            cart.set_arm_target(side, q_arm)
            e.control_dofs_position_velocity([q6_ref], [w], [self.dof6])
            self.ses.step(1)
            q6 = float(cart.qpos()[self.dof6])
            # stalled = wrist lagging its reference by > 0.12 rad and (almost) not turning
            if k > 10 and q6_ref - q6 > 0.12 and abs(q6 - q6_prev) < 0.05 * self.dt:
                stall_n += 1
            else:
                stall_n = 0
            q6_prev = q6
            if stall_n * self.dt > 0.25:
                stalled = True
                break
            # the lid (perceived) has stopped following the hand: pads slipping on a seated lid -> stop
            lid_turn = -(self.env.rel_yaw[(name, name)] - lid_yaw0)
            if (q6 - q6_start) - lid_turn > 0.12:
                stalled = True
                self.log(f"{name}: lid stopped following the wrist ({math.degrees((q6 - q6_start) - lid_turn):.0f} deg) - seated")
                break
        q6 = float(cart.qpos()[self.dof6])
        self.log(f"{name}: stroke end: wrist {(q6 - q6_start) / (2 * np.pi):.3f} turn, lid "
                 f"{-(self.env.rel_yaw[(name, name)] - lid_yaw0) / (2 * np.pi):.3f} turn")
        # hold the wrist where it is (no further tightening torque), restore the normal torque limit
        q_hold = cart.q_target[cart.arm_dofs[side]].copy()
        q_hold[5] = q6
        cart.set_arm_target(side, q_hold)
        e.set_dofs_force_range([-10.0], [10.0], [self.dof6])
        return stalled, (q6 - q6_start) / (2 * np.pi)

    def regrasp(self, name, yaw, z_tcp, r_lid):
        cart = self.cart
        jar_xy = self.jar_pose(name)[0][:2]
        e0 = self.env.engaged_turn[(name, name)]
        self.open_wait({"right": min(2 * r_lid + 0.016, 0.096)})
        q_back = self.ik("right", (jar_xy[0], jar_xy[1], z_tcp + 0.001), yaw=yaw)
        q_back[5] = Q6_A
        self.go({"right": q_back}, 1.2)
        q_down = q_back.copy()
        q_down[:5] = self.ik("right", (jar_xy[0], jar_xy[1], z_tcp), yaw=yaw)[:5]
        self.go({"right": q_down}, 0.2)
        cart.grasp("right", force=LID_GRIP)
        self.grip_until("right", 0.6)
        s = self.env.lid_on_jar(name, name)
        self.log(f"{name}: regrasped: lid turned {self.env.engaged_turn[(name, name)] - e0:+.3f} turn during the regrasp, "
                 f"h {s['height_above_rim'] * 1e3:.3f} mm, width {cart.gripper_width('right') * 1e3:.1f} mm")


def main(args):
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    import run
    import station as S

    env = S.build(gs, kind="puzzle", robot=True, drive=False, dt=args.dt,
                  coupler_kw=ROBOT_COUPLER)
    out = args.out if getattr(args, "out", None) else run.OUT
    ses = run.Session(env, out, video=not args.no_video)
    demo = Demo(env, ses, S)
    demo._q6_prev = float(env.robot.qpos()[demo.dof6])
    ses.phase = "settle"
    demo.wait(0.3)
    ses.still("robot_t0", "wide")
    t0 = time.time()
    order = args.jars.split(",") if getattr(args, "jars", None) else ORDER
    for name in order:
        ses.phase = f"screw_{name}"
        demo.screw(name)
        ses.still(f"robot_after_{name}", "close")
    ses.phase = "home"
    demo.home(1.4, base_x=0.0)
    demo.wait(0.5)
    ses.still("robot_end_wide", "wide")
    ses.still("robot_end_close", "close")
    succ = env.success()
    print("SUCCESS", json.dumps(succ), flush=True)
    ses.close(extra_meta=dict(part="robot", dt=env.dt, d_hat=S.D_HAT, success=succ, events=demo.events,
                              wall_s=time.time() - t0))
    json.dump(dict(success=succ, events=demo.events), open(os.path.join(out, "robot_result.json"), "w"), indent=1)
