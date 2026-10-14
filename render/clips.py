"""Per-station clips for the 2x2 grid: one continuous camera move around one table, the sim at its own speed.

compose.py renders a clip with `--path render/clips.py --clip <station>`: the station is placed alone at the room
centre (its own table and cart only), so all four tiles share the studio lighting. Camera knots are in the station's
own frame (metres, z up). Run directly for a summary:

    python3 render/clips.py [station ...]

Each clip:  (t s, name, eye, target, lens). The eye follows a centripetal Catmull-Rom curve through the knot eyes; its
arc-length progress is a monotone C1 cubic through the knot times (flypath.Path), so moves ease in and out with no
overshoot and the clip starts and ends at rest (the grid can freeze the last frame). The station plays at SPEED[st]
(1 = real time) from video t = 0; the label carries "xN" when N != 1.

The cart's mast is a pole on the midline at (0, -0.675), 1.78 m tall: every eye keeps >= 0.5 m from it and none
looks through it at the work area. Key moments are taken from the export's own event log (meta.json "events").
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import flypath  # noqa: E402
import layout as L  # noqa: E402

FPS = L.RENDER_FPS
SNAP = os.path.join(HERE, "snap")
MAST = (0.0, -0.675, 1.78)

TASK_SHORT = {
    "mahjong": "tiles into the rack, in order",
    "jars": "screw each lid on its jar",
    "cables": "sort cables into bins by colour",
    "water": "pour into three cups",
}
SPEED = {"mahjong": 1.0, "jars": 1.0, "cables": 1.0, "water": 1.0}
# jars: frame 0 -> 1 has a 5 m/s fingertip jump (the gripper's initial state), so its clip starts at frame 1
SIM_START = {"jars": 1.0 / FPS}


def sim_len(st):
    m = json.load(open(os.path.join(SNAP, st, "meta.json")))
    return (m["n_frames"] - 1) / m["fps"]


# ------------------------------------------------------------------------------------------------ camera moves
def orbit(target, az_deg, el_deg, dist):
    """Eye at azimuth/elevation/distance from a target (station frame; az 90 = the far side, facing the robot)."""
    a, e = np.radians(az_deg), np.radians(el_deg)
    return (target[0] + dist * np.cos(e) * np.cos(a), target[1] + dist * np.cos(e) * np.sin(a),
            target[2] + dist * np.sin(e))

def knots_mahjong(T):
    # 2m into slot 1 (left arm) by ~11 s, 6p into slot 5 (right) by ~25 s, the face-down 中 stood up (left) 27-38 s,
    # then carried into slot 12 (right) by ~53 s. The glyphs face the robot, so the camera works from above the
    # robot side: in front of the mast, between and above the arms (only the arm tops show, at the bottom corners).
    return [
        (0.0, "establish 3/4", (1.20, 0.60, 1.60), (0.00, -0.05, 0.84), 32.0),
        (7.5, "rise east", (0.80, -0.20, 2.05), (-0.05, 0.02, 0.79), 30.0),
        (12.5, "over the robot", (0.02, -0.25, 2.15), (0.00, -0.04, 0.78), 37.0),
        (26.5, "over the robot+", (-0.02, -0.25, 2.14), (-0.02, -0.05, 0.78), 37.0),
        (31.0, "in: the flip", (-0.06, -0.20, 1.97), (-0.06, -0.12, 0.78), 40.0),
        (39.5, "flip+", (-0.04, -0.20, 1.97), (-0.02, -0.13, 0.78), 40.0),
        (45.0, "to slot 12", (0.06, -0.22, 2.04), (0.12, 0.04, 0.79), 38.0),
        (53.5, "seated", (0.08, -0.22, 2.06), (0.10, 0.05, 0.79), 44.0),
        # ends on the whole table and the finished rack, centred (the grid holds this frame)
        (T, "out", (0.0, -0.28, 2.26), (0.00, -0.04, 0.78), 52.0),
    ]


def knots_cables(T):
    # drop + settle 0-4 s, approach to ~22 s, grasp, carry, release into the bins ~54-61 s, retreat. Far side (the
    # robot faces the camera), both bins at x = +-0.39 in frame; a slow arc from north-east to north-west.
    # tight on the table top: pile and both bins fill the tile, grippers in frame, arm tops and mast may leave it.
    # A 60 deg arc from the north-east to the north-west, closest during the grasps and the carries.
    c = (0.0, -0.09, 0.80)
    return [
        (0.0, "establish 3/4", orbit((0.0, -0.06, 0.86), 55, 38, 1.30), (0.0, -0.06, 0.86), 30.0),
        (5.0, "pile", orbit(c, 60, 37, 1.12), c, 30.0),
        (20.0, "in: grasp", orbit(c, 72, 35, 1.02), c, 31.0),
        (38.0, "carry", orbit(c, 92, 35, 1.08), c, 30.0),
        (54.0, "both bins", orbit(c, 108, 36, 1.04), c, 30.0),
        (T, "out", orbit(c, 116, 38, 1.14), c, 30.0),
    ]


def knots_jars(T, ev):
    # three jars in a row along x (mason +0.17, jam -0.17, spice 0); each: lid onto the mouth, two strokes. The camera
    # stays on the far side (the hand is above the lid, the thread and the lid's skirt show from the side) and leans
    # in on the jar being worked: the cart drives along x between jars.
    def ttl(key, i=0):
        hits = [e["t"] for e in ev if key in e["msg"]]
        return hits[i] if len(hits) > i else None
    m_on, j_on, s_on = ttl("mason: lid on mouth"), ttl("jam: lid on mouth"), ttl("spice: lid on mouth")
    m_end, j_end, s_end = ttl("mason: done"), ttl("jam: done"), ttl("spice: done")
    m_on, j_on, s_on = m_on or 10.4, j_on or 30.4, s_on or 50.5
    m_end, j_end, s_end = m_end or 20.3, j_end or 40.4, s_end or min(T, 60.4)
    # low, at lid height, close (as the jars station's own camera hint): the lid's skirt descending on the glass
    # thread reads side-on, under the fingers; the arc runs east to west with the work (mason, jam, spice)
    return [
        (0.0, "establish 3/4", (0.90, 0.90, 1.45), (0.05, -0.08, 0.86), 32.0),
        (m_on - 2.0, "in: mason", (0.50, 0.62, 1.04), (0.17, -0.06, 0.88), 45.0),
        (m_end - 2.0, "mason+", (0.46, 0.60, 1.03), (0.17, -0.06, 0.88), 46.0),
        (j_on - 2.0, "in: jam", (-0.02, 0.66, 1.02), (-0.17, -0.06, 0.85), 45.0),
        (j_end - 2.0, "jam+", (-0.05, 0.64, 1.02), (-0.17, -0.06, 0.85), 46.0),
        (s_on - 2.0, "in: spice", (-0.20, 0.64, 1.03), (0.00, -0.06, 0.85), 46.0),
        (min(s_end, T - 3.0), "spice+", (-0.22, 0.62, 1.03), (0.00, -0.06, 0.85), 47.0),
        # eases out to all three jars with their lids on, still close (the grid holds this frame)
        (T, "out", (-0.32, 0.85, 1.22), (0.00, -0.06, 0.86), 38.0),
    ]


def knots_water(T, ev):
    # cups in a row along y at x = -0.02 (tumbler +0.18, mug +0.04, teacup -0.10), pitcher behind them at x = +0.25;
    # Big YAM's right arm pours. From the west the cups spread across the frame with the pitcher behind; the camera
    # leans in on each cup as it is poured.
    def tev(tag, i=0):
        hits = [e[0] for e in ev if isinstance(e, list) and str(e[1]).startswith(tag)]
        return hits[i] if len(hits) > i else None
    cups = [("tumbler", 0.18), ("mug", 0.04), ("teacup", -0.10)]
    k = [(0.0, "establish 3/4", (-0.30, 1.25, 1.62), (0.05, 0.00, 0.86), 32.0)]
    for name, y in cups:
        t_pour, t_un = tev(f"pour:{name}"), tev(f"untilt:{name}")
        if t_pour is None:
            continue
        # the target sits above the cup so the tilted pitcher and the stream stay in frame
        k.append((t_pour - 2.0, f"in: {name}", (-0.80, y + 0.25, 1.32), (0.04, y + 0.02, 0.92), 36.0))
        k.append(((t_un or t_pour + 4) + 0.5, f"{name}+", (-0.78, y + 0.22, 1.32), (0.04, y + 0.02, 0.92), 36.0))
    # eases out over the pitcher's return and placement to all three filled cups (the grid holds this frame)
    k.append((T, "out", (-0.82, 0.40, 1.36), (0.08, 0.03, 0.86), 38.0))
    # drop knots that came out of order (a shorter export)
    out = [k[0]]
    for kk in k[1:]:
        if kk[0] > out[-1][0] + 0.5 and kk[0] <= T:
            out.append(kk)
    if out[-1][0] < T:
        out.append(k[-1])
    return out


class Clip:
    def __init__(self, st):
        self.station = st
        meta = json.load(open(os.path.join(SNAP, st, "meta.json")))
        ev = meta.get("events", [])
        self.sim_T = sim_len(st)
        self.sim_0 = SIM_START.get(st, 0.0)
        self.speed = SPEED.get(st, 1.0)
        self.DURATION = (self.sim_T - self.sim_0) / self.speed
        self.FPS = FPS
        self.N_FRAMES = int(round(self.DURATION * FPS)) + 1
        T = self.DURATION
        if st == "mahjong":
            k = knots_mahjong(T)
        elif st == "cables":
            k = knots_cables(T)
        elif st == "jars":
            k = knots_jars(T, ev)
        elif st == "water":
            k = knots_water(T, ev)
        else:
            raise KeyError(st)
        # knots are written for real time; a sped-up clip scales them
        self.KNOTS = [(t / self.speed if t not in (0.0, T) else t, n, e, g, l) for t, n, e, g, l in k]
        self.KNOTS = [(min(t, T), n, e, g, l) for t, n, e, g, l in self.KNOTS]
        self.path = flypath.Path(self.KNOTS)
        self.CLOCKS = {st: [(0.0, self.speed, self.sim_0, self.sim_T)]}

    def camera(self, t):
        e, g, l = self.path(t)
        return tuple(e[0]), tuple(g[0]), float(l[0])

    def sim_time(self, st, t):
        return min(self.sim_T, self.sim_0 + max(0.0, t * self.speed)) if st == self.station else 0.0

    def label(self):
        title = L.STATIONS[self.station]["title"]
        sp = "" if abs(self.speed - 1) < 1e-6 else f"  ×{self.speed:g}"
        return f"{title}{sp}", TASK_SHORT.get(self.station, "")


def get_clip(st):
    return Clip(st)


def summary(st):
    c = Clip(st)
    tt = np.arange(c.N_FRAMES) / FPS
    e, g, _ = c.path(tt)
    d = g - e
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    v = np.linalg.norm(np.diff(e, axis=0), axis=1) * FPS
    ang = np.degrees(np.arccos(np.clip((d[1:] * d[:-1]).sum(1), -1, 1))) * FPS
    mast = np.linalg.norm(e[:, :2] - np.array(MAST[:2]), axis=1)
    dz = np.clip(e[:, 2] - MAST[2], 0, None) + np.clip(0.75 - e[:, 2], 0, None)   # pole from the deck to its top
    mast3 = np.hypot(mast, dz)
    az = np.degrees(np.arctan2(e[:, 1], e[:, 0]))
    print(f"{st}: {c.DURATION:.1f} s ({c.N_FRAMES} frames) x{c.speed:g}; peak speed {v.max():.2f} m/s, peak pan "
          f"{ang.max():.1f} deg/s; eye z {e[:, 2].min():.2f}-{e[:, 2].max():.2f}; azimuth {az[0]:.0f} -> {az[-1]:.0f} "
          f"deg; nearest to mast {min(mast3.min(), 9):.2f} m (horizontal {mast.min():.2f})")
    for k in c.KNOTS:
        print(f"   {k[0]:6.1f}s {k[1]:16s} eye ({k[2][0]:+.2f},{k[2][1]:+.2f},{k[2][2]:.2f}) -> "
              f"({k[3][0]:+.2f},{k[3][1]:+.2f},{k[3][2]:.2f}) {k[4]:.0f} mm")


if __name__ == "__main__":
    for st in (sys.argv[1:] or [s for s in L.STATIONS if os.path.exists(os.path.join(SNAP, s, "meta.json"))]):
        summary(st)
