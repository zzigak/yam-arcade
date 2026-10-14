"""The arcade fly-through: one continuous camera move plus a clock per station. compose.py imports this file
(`--camera flythrough --path render/flypath.py`); run it directly for a summary or to encode the frames.

    python3 render/flypath.py summary                      # knots, timings, speeds, station clocks
    python3 render/flypath.py encode <frames_dir> <out.mp4> [--fps 30]   # h264 + speed labels

Camera
------
KNOTS are (video time s, name, eye, target, lens), world frame, metres. The eye follows a centripetal Catmull-Rom
curve through the knot eyes (geometry only); its progress along that curve, measured in arc length, is a monotone
C1 cubic (PCHIP) through (t_k, s_k). So the camera eases into each station framing and out of it without overshoot,
and its speed is continuous everywhere. The look-at point does the same along its own curve. Station framings come in
pairs (arrive, leave) a few tens of centimetres apart: a slow drift while the station plays.

Station clocks
--------------
CLOCKS[station] is a list of pieces (video start s, speed, sim from s, sim to s). A piece plays sim [a, b] at `speed`
from its start; between pieces and outside them the station holds the last shown sim time (before the first piece:
its `a`). A label "<Title> ×N" is drawn while a piece with N != 1 plays.
"""
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import layout as L  # noqa: E402

FPS = L.RENDER_FPS
SNAP = os.path.join(HERE, "snap")


def W(st, p):
    """Station-local point -> world."""
    return tuple(L.station_to_world(st, p))


def sim_len(st):
    """Last sim time (s) of a station's snapshot."""
    m = json.load(open(os.path.join(SNAP, st, "meta.json")))
    return (m["n_frames"] - 1) / m["fps"]


# ------------------------------------------------------------------------------------------------ camera knots
WIDE = dict(eye=(3.14, 5.40, 4.69), target=(0.30, 0.05, 0.45))     # the overview framing (compose OVERVIEW)
KNOTS = [
    (0.0, "establish", WIDE["eye"], WIDE["target"], 35.0),
    (5.0, "establish+", (2.95, 4.98, 4.28), (0.40, 0.30, 0.50), 35.0),
    # jars: far side (north-east), facing the robot; the jars stand in a row along x
    (10.0, "jars", W("jars", (0.42, 0.88, 1.28)), W("jars", (0.0, -0.07, 0.84)), 40.0),
    (27.5, "jars+", W("jars", (0.25, 0.82, 1.24)), W("jars", (0.0, -0.07, 0.84)), 42.0),
    # west behind the north tables, south through the gap between them, west along the corridor past the
    # mahjong cart; mahjong stays in view the whole way
    (30.0, "via north", (0.90, 2.30, 1.60), W("mahjong", (0.40, -0.10, 0.85)), 38.0),
    (31.9, "via gap", (0.00, 0.85, 1.60), W("mahjong", (0.10, -0.15, 0.85)), 36.0),
    (33.7, "via corridor", (-1.60, -0.30, 1.70), W("mahjong", (-0.20, -0.10, 0.85)), 34.0),
    # mahjong: from the robot's side (west), outside the arm span and above the near arm: the tiles stand in the
    # rack facing the robot
    (35.5, "mahjong", W("mahjong", (-1.15, -0.85, 1.62)), W("mahjong", (0.02, 0.06, 0.79)), 36.0),
    (55.0, "mahjong+", W("mahjong", (-1.10, -0.95, 1.66)), W("mahjong", (0.02, 0.04, 0.79)), 36.0),
    # the pan to cables looks east down the corridor (the east tables in the distance), never at the floor
    (57.6, "via east look", (-2.55, 0.15, 1.64), (1.20, -0.20, 0.90), 34.0),
    # cables: from the corridor's west end (north of the table), facing the robot, both bins in frame
    (60.0, "cables", W("cables", (-0.40, 1.30, 1.60)), W("cables", (0.0, -0.08, 0.80)), 35.0),
    (79.5, "cables+", W("cables", (-0.25, 1.25, 1.55)), W("cables", (0.0, -0.08, 0.80)), 36.0),
    # head for the water table looking straight at it from across the gap
    (81.8, "via water look", (-0.95, -0.75, 1.48), W("water", (-0.15, 0.0, 0.85)), 38.0),
    # water: from the west, so the three cups (a row along y) spread across the frame, the pitcher behind them
    (84.5, "water", W("water", (-1.00, 0.30, 1.30)), W("water", (0.05, -0.02, 0.85)), 40.0),
    (104.0, "water+", W("water", (-0.90, 0.45, 1.36)), W("water", (0.05, -0.02, 0.85)), 42.0),
    # pull back: round the water table's north side, rising, still on the water table, then up to the opening frame
    (108.5, "pull back", (3.30, -0.30, 2.60), W("water", (0.0, 0.0, 0.85)), 35.0),
    (113.0, "wide", WIDE["eye"], WIDE["target"], 35.0),
    (115.0, "wide+", (3.22, 5.55, 4.80), WIDE["target"], 35.0),
]
DURATION = KNOTS[-1][0]
N_FRAMES = int(round(DURATION * FPS))


# ------------------------------------------------------------------------------------------------ station clocks
def _clocks():
    T = {st: sim_len(st) for st in L.STATIONS if os.path.exists(os.path.join(SNAP, st, "meta.json"))}
    c = {}
    if "jars" in T:   # frame 0 -> 1 has a 5 m/s fingertip jump (initial gripper state), so start at frame 1
        c["jars"] = [(7.5, 3.0, 1.0 / FPS, T["jars"])]
    if "mahjong" in T:
        c["mahjong"] = [(35.5, 3.0, 0.0, T["mahjong"])]
    if "cables" in T:  # the drop and settle (0-4 s) plays live in the establishing shot, the sorting at x3 later
        c["cables"] = [(1.0, 1.0, 0.0, 4.0), (59.0, 3.0, 4.0, T["cables"])]
    if "water" in T:
        c["water"] = [(84.0, 2.0, 0.0, T["water"])]
    return c


CLOCKS = _clocks()


def piece_end(p):
    v0, k, a, b = p
    return v0 + (b - a) / k


def sim_time(st, t):
    """Sim time (s) shown by station `st` at video time t."""
    pieces = CLOCKS.get(st)
    if not pieces:
        return 0.0
    cur = pieces[0][2]
    for p in pieces:
        v0, k, a, b = p
        if t < v0:
            break
        cur = min(b, a + (t - v0) * k)
    return cur


def labels():
    """[(t0, t1, text)] for every piece played faster or slower than real time."""
    out = []
    for st, pieces in CLOCKS.items():
        for p in pieces:
            if abs(p[1] - 1.0) > 1e-6:
                out.append((p[0], piece_end(p), f"{L.STATIONS[st]['title']}  ×{p[1]:g}"))
    return sorted(out)


# ------------------------------------------------------------------------------------------------ splines
def _catmull_rom_dense(P, per_seg=200, alpha=0.5):
    """Centripetal Catmull-Rom through P (n x 3): dense samples and the index of each knot in them."""
    P = np.asarray(P, float)
    n = len(P)
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out, knot_idx = [], [0]
    for i in range(n - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]

        def tj(ti, a, b):
            return ti + max(np.linalg.norm(b - a), 1e-6) ** alpha
        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        ts = np.linspace(t1, t2, per_seg, endpoint=False)[:, None]
        A1 = (t1 - ts) / (t1 - t0) * p0 + (ts - t0) / (t1 - t0) * p1
        A2 = (t2 - ts) / (t2 - t1) * p1 + (ts - t1) / (t2 - t1) * p2
        A3 = (t3 - ts) / (t3 - t2) * p2 + (ts - t2) / (t3 - t2) * p3
        B1 = (t2 - ts) / (t2 - t0) * A1 + (ts - t0) / (t2 - t0) * A2
        B2 = (t3 - ts) / (t3 - t1) * A2 + (ts - t1) / (t3 - t1) * A3
        C = (t2 - ts) / (t2 - t1) * B1 + (ts - t1) / (t2 - t1) * B2
        out.append(C)
        knot_idx.append(knot_idx[-1] + per_seg)
    out.append(P[-1:])
    D = np.vstack(out)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(D, axis=0), axis=1))]
    return D, s, s[np.array(knot_idx)]


def _pchip_slopes(x, y):
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros_like(y)
    for k in range(1, len(y) - 1):
        if d[k - 1] * d[k] <= 0:
            m[k] = 0.0
        else:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    m[0] = m[-1] = 0.0   # the move starts and ends at rest
    return m


def _pchip(x, y, xq):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = _pchip_slopes(x, y)
    xq = np.clip(xq, x[0], x[-1])
    i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
    h = x[i + 1] - x[i]
    u = (xq - x[i]) / h
    h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    return h00 * y[i] + h10 * h * m[i] + h01 * y[i + 1] + h11 * h * m[i + 1]


class Path:
    def __init__(self, knots=KNOTS):
        self.t = np.array([k[0] for k in knots])
        self.eD, self.es, self.eks = _catmull_rom_dense([k[2] for k in knots])
        self.tD, self.ts, self.tks = _catmull_rom_dense([k[3] for k in knots])
        self.lens = np.array([k[4] for k in knots])

    @staticmethod
    def _at(D, s, sq):
        j = np.clip(np.searchsorted(s, sq) - 1, 0, len(s) - 2)
        f = np.clip((sq - s[j]) / np.maximum(s[j + 1] - s[j], 1e-12), 0, 1)[..., None]
        return D[j] * (1 - f) + D[j + 1] * f

    def __call__(self, tq):
        """eye, target, lens at video time(s) tq."""
        tq = np.atleast_1d(np.asarray(tq, float))
        eye = self._at(self.eD, self.es, _pchip(self.t, self.eks, tq))
        tgt = self._at(self.tD, self.ts, _pchip(self.t, self.tks, tq))
        lens = _pchip(self.t, self.lens, tq)
        return eye, tgt, lens


_PATH = None


def camera(t):
    """(eye xyz, target xyz, lens mm) at video time t."""
    global _PATH
    if _PATH is None:
        _PATH = Path()
    e, g, l = _PATH(t)
    return tuple(e[0]), tuple(g[0]), float(l[0])


# ------------------------------------------------------------------------------------------------ CLI
def summary():
    p = Path()
    tt = np.arange(N_FRAMES) / FPS
    e, g, l = p(tt)
    v = np.linalg.norm(np.diff(e, axis=0), axis=1) * FPS
    d = g - e
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((d[1:] * d[:-1]).sum(1), -1, 1))) * FPS
    print(f"fly-through: {DURATION:.1f} s, {N_FRAMES} frames at {FPS} fps")
    print(f"eye path {p.es[-1]:.2f} m; peak speed {v.max():.2f} m/s at t={tt[v.argmax()]:.1f}; peak pan "
          f"{ang.max():.1f} deg/s at t={tt[ang.argmax()]:.1f}; eye z {e[:, 2].min():.2f}-{e[:, 2].max():.2f} m")
    for k, (t, name, eye, tgt, lens) in enumerate(KNOTS):
        i = min(int(round(t * FPS)), N_FRAMES - 1)
        print(f"  {t:6.1f}s {name:12s} eye ({eye[0]:+.2f},{eye[1]:+.2f},{eye[2]:.2f}) target "
              f"({tgt[0]:+.2f},{tgt[1]:+.2f},{tgt[2]:.2f}) lens {lens:.0f}  speed {v[min(i, len(v) - 1)]:.2f} m/s")
    for st, pieces in CLOCKS.items():
        for v0, k, a, b in pieces:
            print(f"  clock {st:8s} video {v0:6.1f}-{piece_end((v0, k, a, b)):6.1f} s plays sim {a:5.2f}-{b:5.2f} s "
                  f"x{k:g}")
    for t0, t1, txt in labels():
        print(f"  label {t0:6.1f}-{t1:6.1f} s  {txt}")


FONT = "/usr/share/fonts/opentype/inter/Inter-Medium.otf"


def encode(frames_dir, out, fps=FPS, crf=18):
    """frames_dir/NNNNN.png (video frame index) -> h264 mp4 with the speed labels burnt in."""
    frames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png") and f[:5].isdigit())
    if not frames:
        raise SystemExit("no frames")
    first = int(frames[0][:5])
    h = int(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=height", "-of", "csv=p=0", os.path.join(frames_dir, frames[0])]))
    fs = max(14, round(h * 0.030))
    pad = round(h * 0.035)
    filt = []
    for t0, t1, txt in labels():
        # fade in and out over 0.4 s
        a = (f"if(lt(t,{t0}),0,if(lt(t,{t0 + 0.4}),(t-{t0})/0.4,if(lt(t,{t1 - 0.4}),1,"
             f"if(lt(t,{t1}),({t1}-t)/0.4,0))))")
        filt.append(f"drawtext=fontfile={FONT}:text='{txt}':fontsize={fs}:fontcolor=0x2b2a28:alpha='{a}':"
                    f"x={pad}:y=h-{pad}-th:box=1:boxcolor=white@0.72:boxborderw={round(fs * 0.45)}:"
                    f"enable='between(t,{t0},{t1})'")
    vf = ",".join(filt + ["scale=trunc(iw/2)*2:trunc(ih/2)*2"])
    cmd = ["nice", "-n", "10", "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-start_number",
           str(first), "-i", os.path.join(frames_dir, "%05d.png"), "-vf", vf, "-c:v", "libx264", "-crf", str(crf),
           "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", out]
    subprocess.run(cmd, check=True)
    print(f"encoded {out} ({len(frames)} frames from {first})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "encode":
        encode(sys.argv[2], sys.argv[3], fps=int(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else FPS)
    else:
        summary()
