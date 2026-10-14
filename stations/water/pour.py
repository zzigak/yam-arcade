"""Pouring controller shared by the oracle mount and the robot.

The pitcher pose is parametrised by its spout lip (the pour pivot) and a tilt beta about the pitcher's local y axis
(beta > 0 dips the spout). Yaw is fixed (spout toward station -x). For each cup:

  travel   lip to a pre-pour point above/next to the cup, pitcher upright (min-jerk; via a raised retreat point)
  tilt     fast tilt to a few degrees short of the geometric flow-onset angle, lip held at the schedule height
  pour     slow tilt; stop when cup volume + water in flight toward it reaches the target (closed loop on the
           simulated fill: a stand-in for the vision / scale a real robot would use - disclosed in NOTES.md)
  untilt   back to upright, then a short drip pause

The lip height follows a schedule z_lip(beta) computed from the pitcher's own vertices so that the pitcher never
comes within 1 cm of the table, and stays above the cup's rim wherever it overhangs the cup.
"""
import numpy as np

from station import TABLE_Z, quat_to_R, yaw_quat, PLACEMENT

DEG = np.pi / 180.0


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_y(beta):
    return np.array([np.cos(beta / 2), 0.0, np.sin(beta / 2), 0.0])


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def ease_out(s):
    """Starts at full speed (3x the mean rate) and decelerates smoothly to rest: cuts the stream off quickly."""
    s = np.clip(s, 0.0, 1.0)
    return 1.0 - (1.0 - s) ** 3


class PourGeometry:
    def __init__(self, pitcher, yaw_deg=180.0):
        self.c = pitcher
        spec = pitcher.spec
        self.yaw_q = yaw_quat(yaw_deg)
        self.R_yaw = quat_to_R(self.yaw_q)
        lip_in = pitcher.inner_points(0.0, spec.z_rim)
        self.u_lip = np.array([lip_in[0] + spec.wall, 0.0, spec.z_rim])   # outer edge of the V lip (local)
        verts = np.concatenate([m.vertices for m in pitcher.visual_parts().values()])
        rng = np.random.default_rng(0)
        self.hull_pts = verts[rng.choice(len(verts), min(4000, len(verts)), replace=False)]
        # cavity samples for the flow-onset angle (2.5 mm grid, below the lid)
        h = 0.0025
        xs = np.arange(-0.07, 0.095, h)
        ys = np.arange(-0.055, 0.055, h)
        zs = np.arange(spec.z_floor + 0.5 * h, spec.z_rim - (spec.lid.t_below if spec.lid else 0.0), h)
        P = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
        cav, _ = pitcher.classify_local(P)
        self.cav_pts, self.cav_dv = P[cav], h ** 3

    def R(self, beta):
        """Station-frame rotation of the pitcher at tilt beta."""
        return quat_to_R(self.quat(beta))

    def quat(self, beta):
        return quat_mul(self.yaw_q, quat_y(beta))

    def base_pose(self, lip_pos, beta):
        """Pitcher base (link origin) position and quaternion for the lip at lip_pos (station) and tilt beta."""
        R = self.R(beta)
        return lip_pos - R @ self.u_lip, self.quat(beta)

    def onset_tilt(self, volume_m3):
        """Tilt at which the free surface of `volume_m3` of water reaches the lip (rad)."""
        for beta in np.arange(0.0, 120.0, 0.5) * DEG:
            R = quat_to_R(quat_y(beta))
            h = (self.cav_pts @ R.T)[:, 2]
            n = int(volume_m3 / self.cav_dv)
            if n >= len(h):
                return 0.0
            surf = np.partition(h, n)[n]
            if surf >= (R @ self.u_lip)[2]:
                return beta
        return 120.0 * DEG

    def lip_height(self, beta, cup_rim_z, cup_r_in, cup_r_out, clear_table=0.012, clear_rim=0.012,
                   lip_above_rim=0.03):
        """Lowest admissible lip height (station z) at tilt beta when pouring into a cup whose rim is at cup_rim_z
        (the lip sits lip_x_offset(cup_r_in) toward +x from the cup axis)."""
        R = self.R(beta)
        rel = (self.hull_pts - self.u_lip) @ R.T        # pitcher points relative to the lip, station axes
        z = cup_rim_z + lip_above_rim
        x_off = self.lip_x_offset(cup_r_in)
        dx = rel[:, 0] + x_off
        dy = rel[:, 1]
        over_cup = np.hypot(dx, dy) < cup_r_out + 0.012
        z = max(z, TABLE_Z + clear_table - rel[:, 2].min())
        if over_cup.any():
            z = max(z, cup_rim_z + clear_rim - rel[over_cup, 2].min())
        return z

    @staticmethod
    def lip_x_offset(cup_r_in):
        """Lip sits this far toward the pitcher (+x) from the cup axis, so the stream lands near the centre."""
        return 0.30 * cup_r_in


class OracleDriver:
    """6-DoF PD mount: disclosed oracle 'hand' driving the pitcher directly (see NOTES.md)."""

    def __init__(self, env):
        self.env = env
        self.ent = env.entities["pitcher"]
        self.mount_pos, self.mount_q = env.pitcher_mount_pose
        self.R_m = quat_to_R(self.mount_q)
        self.prev = None

    def command(self, base_pos, base_quat, dt):
        # mount-frame target: slides = R_m^T (p - p_mount); rotations: Rz(yaw) Ry(pitch) Rx(roll) = R_m^T R
        p = self.R_m.T @ (np.asarray(base_pos) - self.mount_pos)
        Rl = self.R_m.T @ quat_to_R(base_quat)
        pitch = np.arcsin(-np.clip(Rl[2, 0], -1, 1))
        yaw = np.arctan2(Rl[1, 0], Rl[0, 0])
        roll = np.arctan2(Rl[2, 1], Rl[2, 2])
        q = np.array([p[0], p[1], p[2], yaw, pitch, roll])
        v = np.zeros(6) if self.prev is None else (q - self.prev) / dt
        self.prev = q
        self.ent.control_dofs_position_velocity(q, v, self.env.oracle_idx)

    def update(self):
        pass

    def prepare(self):
        return
        yield

    def release(self):
        return
        yield


class PourController:
    def __init__(self, env, driver, order=("tumbler", "mug", "teacup"), fill=0.72, dt=1.0 / 300,
                 tilt_rate=7.0 * DEG, onset_margin=8.0 * DEG, beta_max=100.0 * DEG, untilt_rate=120.0 * DEG,
                 stop_lag=0.25, dry_tilt=85.0 * DEG, slow=1.0):
        self.env, self.driver, self.dt = env, driver, dt
        self.geo = PourGeometry(env.containers["pitcher"])
        self.order = list(order)
        self.fill, self.tilt_rate, self.onset_margin, self.beta_max = fill, tilt_rate, onset_margin, beta_max
        self.untilt_rate, self.stop_lag, self.dry_tilt = untilt_rate, stop_lag, dry_tilt
        self.slow = slow   # duration multiplier for the transfer moves (robot mode moves more gently)
        self.log = []
        self.events = []
        p0 = np.asarray(PLACEMENT["pitcher"][:2])
        self.home_lip = np.array([p0[0], p0[1], TABLE_Z]) + self.geo.R(0.0) @ self.geo.u_lip
        self.lip = self.home_lip.copy()
        self.beta = 0.0
        self.t = 0.0
        self.vol_cache = None
        self._gen = self._program()

    # ---------------------------------------------------------------- helpers
    def cup_info(self, name):
        c = self.env.containers[name]
        pos, _ = self.env.pose(name)
        r_in = float(c._lookup(c._R_in, np.array([0.0]), np.array([c.spec.z_rim]))[0])
        r_out = r_in + c.spec.wall
        return pos[:2].copy(), TABLE_Z + c.spec.z_rim, r_in, r_out

    def pour_lip(self, name, beta):
        xy, rim_z, r_in, r_out = self.cup_info(name)
        z = self.geo.lip_height(beta, rim_z, r_in, r_out)
        return np.array([xy[0] + self.geo.lip_x_offset(r_in), xy[1], z])

    def volumes(self):
        return self.env.volumes()

    def _move(self, lip_goal, beta_goal, duration):
        lip0, beta0 = self.lip.copy(), self.beta
        n = max(1, int(round(duration / self.dt)))
        for k in range(1, n + 1):
            s = min_jerk(k / n)
            self.lip = lip0 + s * (lip_goal - lip0)
            self.beta = beta0 + s * (beta_goal - beta0)
            yield

    def _tilt_to(self, name, beta_goal, duration, profile=min_jerk):
        beta0 = self.beta
        n = max(1, int(round(duration / self.dt)))
        for k in range(1, n + 1):
            self.beta = beta0 + profile(k / n) * (beta_goal - beta0)
            self.lip = self.pour_lip(name, self.beta)
            yield

    def _wait(self, secs):
        for _ in range(max(1, int(round(secs / self.dt)))):
            yield

    # ---------------------------------------------------------------- program
    def _program(self):
        self.commanding = False
        yield from self.driver.prepare()
        self.lip = self._actual_lip()
        self.beta = 0.0
        self.commanding = True
        yield from self._wait(0.2)
        # lift straight up first so the pitcher clears the table
        yield from self._move(self.lip + np.array([0.0, 0.0, 0.03]), 0.0, 0.6 * self.slow)
        for name in self.order:
            cap = self.env.infos[name]["capacity"] * 1e6
            target = self.fill * cap
            self.events.append((self.t, f"travel:{name}"))
            pre = self.pour_lip(name, 0.0)
            via = np.array([max(self.lip[0], pre[0]) + 0.04, 0.5 * (self.lip[1] + pre[1]), max(self.lip[2], pre[2]) + 0.03])
            yield from self._move(via, 0.0, 1.0 * self.slow)
            yield from self._move(pre, 0.0, 0.9 * self.slow)
            v_p = (self.volumes()["pitcher"] if self.env.liquid is not None else 1200.0) * 1e-6
            b_on = self.geo.onset_tilt(v_p)
            self.events.append((self.t, f"tilt:{name}:onset={b_on / DEG:.1f}deg:pitcher={v_p * 1e6:.0f}ml"))
            yield from self._tilt_to(name, max(0.0, b_on - self.onset_margin), 1.0 * self.slow)
            # closed-loop pour
            self.events.append((self.t, f"pour:{name}:target={target:.0f}ml"))
            if self.env.liquid is None:   # dry rehearsal (robot reach / grasp test): tilt to dry_tilt and hold
                yield from self._tilt_to(name, self.dry_tilt, 2.0)
                yield from self._wait(0.8)
                yield from self._tilt_to(name, 0.0, 3.0 * self.beta / self.untilt_rate, profile=ease_out)
                continue
            k = 0
            hist = []   # (t, delivered ml) for the flow-rate estimate
            while True:
                if k % 3 == 0:
                    vols = self.volumes()
                    in_cup = vols[name]
                    airborne = vols["airborne"]
                    self.vol_cache = vols
                    delivered = in_cup + airborne
                    hist.append((self.t, delivered))
                    old = [h for h in hist if h[0] >= self.t - 0.15]
                    q = (delivered - old[0][1]) / max(self.t - old[0][0], 1e-3)
                    # stop early by the water that will still leave the lip while the untilt cuts the stream
                    if delivered + self.stop_lag * max(q, 0.0) >= target or self.beta >= self.beta_max:
                        break
                k += 1
                # slow down as we approach the target (avoid overshoot from the stream already in the air)
                rate = self.tilt_rate * (1.0 if in_cup < 0.5 * target else 0.5)
                self.beta = min(self.beta + rate * self.dt, self.beta_max)
                self.lip = self.pour_lip(name, self.beta)
                yield
            self.events.append((self.t, f"untilt:{name}:cup={in_cup:.0f}ml:air={airborne:.0f}ml:"
                                        f"rate={q:.0f}ml/s:beta={self.beta / DEG:.1f}deg"))
            yield from self._tilt_to(name, 0.0, 3.0 * self.beta / self.untilt_rate, profile=ease_out)
            yield from self._wait(0.5)
        # put the pitcher back down where it started
        self.events.append((self.t, "return"))
        up = self.home_lip + np.array([0.0, 0.0, 0.035])
        via = np.array([max(self.lip[0], up[0]) + 0.02, 0.5 * (self.lip[1] + up[1]), max(self.lip[2], up[2]) + 0.02])
        yield from self._move(via, 0.0, 1.0 * self.slow)
        yield from self._move(up, 0.0, 0.9 * self.slow)
        yield from self._move(self.home_lip.copy(), 0.0, 0.8 * self.slow)
        self.events.append((self.t, "placed"))
        yield from self._wait(0.3)
        self.commanding = False
        yield from self.driver.release()
        self.events.append((self.t, "released"))
        self.done = True
        while True:
            yield

    def _actual_lip(self):
        pos, quat = self.env.pose("pitcher")
        return pos + quat_to_R(quat) @ self.geo.u_lip

    def step(self):
        """Advance the program by one sim step and send the pitcher command."""
        next(self._gen)
        if self.commanding:
            base_pos, base_quat = self.geo.base_pose(self.lip, self.beta)
            self.driver.command(base_pos, base_quat, self.dt)
        self.t += self.dt

    done = False
