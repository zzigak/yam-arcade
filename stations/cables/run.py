"""Cables station run: drop the eight cables from the air onto the table, let them settle into a tangled pile, and
(with --robot) have the YAM Ultra pick one cable out of the pile and drop it into the bin of its colour. Everything is
physics; the robot is driven by joint PD targets only.

    stations/cables/gpu_run.sh out/cables/run.log stations/cables/run.py [--robot] [--t-settle 4]

Writes out/cables/: meta.json + traj.npz + meshes/ (export.Recorder), log.json (per-frame physics log),
checks.json, preview.mp4, still_*.png.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if "--dt" in sys.argv:  # station.DT is read at import
    os.environ["CABLES_DT"] = str(eval(sys.argv[sys.argv.index("--dt") + 1], {}))
import station as S  # noqa: E402
import cable_model as CM  # noqa: E402
import export  # noqa: E402
import layout  # noqa: E402

import genesis as gs  # noqa: E402
import genesis.utils.mesh as mu  # noqa: E402
from genesis.utils.misc import tensor_to_array as A  # noqa: E402

OUT = os.path.join(S.ROOT, "out", "cables")
CAMS = {
    "front": dict(res=(1280, 720), pos=(0.95, 0.80, 1.40), lookat=(0.0, -0.08, 0.80), fov=46),
    "top": dict(res=(1280, 720), pos=(0.0, -0.02, 1.85), lookat=(0.0, -0.02, 0.75), fov=40, up=(0.0, 1.0, 0.0)),
    "close": dict(res=(1280, 720), pos=(0.25, 0.45, 1.05), lookat=(0.0, -0.06, 0.76), fov=40),
}


def kabsch_rmsd(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    return float(np.sqrt(((Xc @ R - Yc) ** 2).sum(1).mean()))


def straight_rmsd(P):
    """Kabsch RMSD between rest shape P and a straight line of the same arc-length parametrisation."""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return kabsch_rmsd(np.stack([s, 0 * s, 0 * s], 1), P)


class TubeDrawer:
    """Draws every cable as a tube (and the bin labels) with debug meshes: Genesis's FEM.Rod render tube collapses
    to a line, and only debug cameras show debug meshes."""

    def __init__(self, env):
        import trimesh
        self.env, self.nodes, self.trimesh = env, [], trimesh
        self.label = trimesh.load(env.label_mesh_path, process=False)

    def draw(self, pts):
        sc = self.env.scene
        for n in self.nodes:
            sc.clear_debug_object(n)
        self.nodes = []
        for name, P in pts.items():
            c = S.CABLE_BY_NAME[name]
            v, f = mu.create_tube(P[self.env.cables[name]["n_inner"]:], c["radius"], 14, False)
            tm = self.trimesh.Trimesh(v, f, process=False)
            tm.visual.vertex_colors = (np.array([*S.COLORS[c["color"]], 1.0]) * 255).astype(np.uint8)
            self.nodes.append(sc.draw_debug_mesh(tm))
        for col, (p, q) in self.env.bin_poses().items():
            lab = self.label.copy()
            lab.apply_transform(np.block([[S.quat_to_mat(q), p[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
            lab.visual.vertex_colors = (np.array([*S.COLORS[col], 1.0]) * 255).astype(np.uint8)
            self.nodes.append(sc.draw_debug_mesh(lab))


def bin_frame_check(env):
    """After the first step: libuipc's affine-body frame of each bin (whose pose is exported with the bin mesh) vs the
    Genesis link that carries the same mesh, and vs the requested placement (it has fallen < 1 mm by then)."""
    out = {}
    for col in S.BINS:
        p_ipc, q_ipc = env.bin_poses()[col]
        p_gs, q_gs = env.bin_poses(genesis_copy=True)[col]
        want = np.array([S.BINS[col]["xy"][0], S.BINS[col]["xy"][1], S.ZT + S.BIN_LIFT])
        # exported mesh x recorded pose vs Genesis's own vertices of that link (Genesis's copy trails libuipc by one
        # gravity step, ~1 mm)
        import trimesh
        M = np.asarray(trimesh.load(env.bin_visual_path, process=False).vertices)
        W = M @ S.quat_to_mat(q_ipc).T + p_ipc
        G = np.concatenate([A(g.get_verts()) for g in env.bins[col].base_link.geoms])
        mesh_err = max(float(np.abs(W.min(0) - G.min(0)).max()), float(np.abs(W.max(0) - G.max(0)).max()))
        out[col] = dict(export_mesh_vs_genesis_verts_mm=1e3 * mesh_err,
                        ipc_vs_genesis_mm=1e3 * float(np.linalg.norm(p_ipc - p_gs)),
                        ipc_vs_requested_mm=1e3 * float(np.linalg.norm(p_ipc - want)),
                        offset_d=env._bin_offset(col).tolist(),
                        quat_ipc=q_ipc.round(6).tolist(), quat_gs=q_gs.round(6).tolist())
    return out


# ------------------------------------------------------------------------------------------------- robot demo plan
def plan_pick(env, pts, side_for_color=None, clear=0.03):
    """Grasp candidates on the settled pile, best first: a vertex of a cable lying on the table (centreline <= 2 mm
    above resting height), no other cable (or another turn of the same one) within `clear` in top view, on the side
    of the arm whose bin matches the cable colour. Fingers close across the local tangent."""
    side_for_color = side_for_color or {c: S.arm_for_color(c) for c in S.BINS}
    cands = []
    for name, P in pts.items():
        c = S.CABLE_BY_NAME[name]
        side = side_for_color[c["color"]]
        others = np.vstack([Q for n, Q in pts.items() if n != name] + [np.full((1, 3), 1e3)])
        n = len(P)
        for i in range(4, n - 4):
            p = P[i]
            if p[2] - S.ZT - c["radius"] > 0.003:
                continue
            dxy = np.linalg.norm(others[:, :2] - p[:2], axis=1)
            own = np.linalg.norm(P[:, :2] - p[:2], axis=1)
            if dxy.min() < clear or np.any((own < clear) & (np.abs(np.arange(n) - i) > 6)):
                continue
            if (side == "left" and p[0] > 0.05) or (side == "right" and p[0] < -0.05):
                continue
            mid = abs(i - (n - 1) / 2) / ((n - 1) / 2)  # prefer the middle: the cable hangs shorter
            t = P[i + 1] - P[i - 1]
            cands.append(dict(cable=name, vertex=i, point=p.tolist(), yaw=float(np.arctan2(t[1], t[0]) + np.pi / 2),
                              side=side, score=float(min(dxy.min(), 0.1) - 0.08 * mid - 0.3 * max(0.0, p[1] + 0.05)),
                              clearance=float(dxy.min())))
    cands.sort(key=lambda d: -d["score"])
    # one candidate per cable, best first
    seen, out = set(), []
    for d in cands:
        if d["cable"] not in seen:
            seen.add(d["cable"])
            out.append(d)
    return out


GRIP_SQUEEZE = 0.006  # m: commanded pad gap below the cable diameter (3 mm per finger)
PLUG_GRIP_FORCE = 15.0  # N per finger on a connector body (pad-plug friction ~0.77: holds ~23 N of pull). 4 N let the
#                        extension lead's plug pull out while 5 cables lay across that cable; 8 N let the HDMI plug
#                        pull out when its stiff lead dragged through the pile. At dt 1/90 the drawn fingers then sit
#                        ~1.3 mm inside the plug (F / k_coupling, k = 20 m / dt^2)
GRIP_FORCE = 6.0  # N per finger cap


def grip_close(cart, side, radius):
    """Close on a cable of `radius` with a WIDTH target a little under its diameter, speed-limited and force-capped.
    Not cart.grasp(): on a 4-9 mm cable the fingers touch at q = 1.5-4 mm, too late for its stall detector, so its
    target runs past closed; the drawn Genesis fingers (which only the coupling stops) then crossed each other while
    the libuipc pads held the cable, and the coupling forces threw the arm (tests/grasp_debug.py, first run). With a
    width target the PD and the coupling spring settle within a fraction of a mm of the pads."""
    cart.set_gripper(side, max(0.0, 2 * radius - GRIP_SQUEEZE), force=GRIP_FORCE, speed=0.02)


def plan_pick_plugs(env, pts, side_for_color=None, clear=0.03):
    """Grasp candidates on the settled pile by the connector: a plug resting on the table (hull bottom <= 3 mm above
    it) with its axis near horizontal, nothing else (other cables, other plugs) within `clear` of the plug body in top
    view, on the side of the arm whose bin matches the cable colour. Fingers close across the plug body's horizontal
    width at the body centre. Shorter, lighter cables first (less cable trails from the gripper)."""
    side_for_color = side_for_color or {c: S.arm_for_color(c) for c in S.BINS}
    poses = env.plug_poses()
    tm = env.tangle_metric(pts)
    inter = {n: sum(n in p.split("|") for p in tm["interlocked_pairs"]) for n in pts}
    touch = {n: sum(n in p.split("|") and v["contacts"] > 0 for p, v in tm["pairs"].items()) for n in pts}
    plug_pts = {n: S.plug_info(S.CABLE_BY_NAME[n]["plug"])["hull_pts"] @ S.quat_to_mat(q).T + p for n, (p, q) in poses.items()}
    cands = []
    for name, (p, q) in poses.items():
        c = S.CABLE_BY_NAME[name]
        info = S.plug_info(c["plug"])
        R = S.quat_to_mat(q)
        ax = R[:, 0]
        if abs(R[2, 1]) > 0.35 or abs(ax[2]) > 0.8:  # the width axis must be ~horizontal (pads squeeze the sides)
            continue
        X = plug_pts[name]
        if X[:, 2].min() - S.ZT > 0.003:
            continue
        d = np.array([R[0, 1], R[1, 1], 0.0])  # the plug's width axis, horizontal
        d /= np.linalg.norm(d)
        width = float(np.ptp(X @ d))
        ctr = p + R @ np.array([0.5 * sum(info["body_x"]), 0.0, 0.0])
        others = [Q for n, Q in pts.items() if n != name] + [Y for n, Y in plug_pts.items() if n != name]
        own = pts[name][env.cables[name]["n_inner"] + 4:]  # the cable beyond its first few cm past the boot
        allo = np.vstack(others + [own])
        dxy = np.linalg.norm(allo[:, :2] - ctr[:2], axis=1).min()
        side = side_for_color[c["color"]]
        # each arm reaches ~0.15 m past the centre line (robots/README.md); the IK plan decides the rest
        # pads open to width + PLUG_OPEN_MARGIN: their outer faces sit width/2 + 8 mm + 8 mm from the centre; no other
        # cable's centreline inside that (a pad may nudge a neighbour's jacket: that is ordinary contact)
        if dxy < 0.5 * width + 0.016 or (side == "left" and ctr[0] > 0.15) or (side == "right" and ctr[0] < -0.15):
            continue
        L = float(env.cables[name]["framed"].cable_rest.sum())
        # untangled first (an interlocked cable cannot be lifted out of the pile without untangling it), then fewer
        # cables touching it, then shorter
        cands.append(dict(cable=name, plug=c["plug"], point=[float(ctr[0]), float(ctr[1]), float(X[:, 2].min())],
                          yaw=float(np.arctan2(d[1], d[0])), width=width, side=side, clearance=float(dxy),
                          length=L, interlocked=inter[name], touching=touch[name],
                          score=float(-10.0 * inter[name] - 0.25 * touch[name] - 2.0 * L - 2.0 * max(0.0, ctr[1] + 0.05))))
    cands.sort(key=lambda d: -d["score"])
    return cands


def demo_plan_plug(env, pick):
    """TCP keyframes for picking a cable by its plug and dropping it into its bin: top-down to the plug, lift with the
    approach axis pitching forward as the YAM Ultra's top-down reach ends, carry past the bin centre, lower into the
    bin (the trailing cable is pulled over the rim and falls in), release, retreat."""
    cart, side = env.robot, pick["side"]
    p = np.array(pick["point"])
    yaw0 = (pick["yaw"] + np.pi / 2) % np.pi - np.pi / 2
    zt = S.ZT
    bx, by = S.BINS[S.CABLE_BY_NAME[pick["cable"]]["color"]]["xy"]
    # release in the bin's far corner (away from the pile, toward +y): the trailing cable then lies along the bin's
    # diagonal (0.34 m inside) instead of across it
    ix, iy = 0.5 * S.BIN["size"][0] - S.BIN["wall"] - 0.03, 0.5 * S.BIN["size"][1] - S.BIN["wall"] - 0.03
    cx, cy = bx + np.sign(bx) * ix, by + iy
    tcp0, _ = cart.tcp_pose(side)
    # smallest approach pitch (about the closing axis) that reaches above and at the plug; the pads' lower corner
    # drops by PAD_HALF_WIDTH sin(tilt), so the TCP rises by that, and the pads must still reach below the plug's
    # mid-height
    H = S.plug_info(pick["plug"])["height"]
    tg = yaw = None
    for tilt in (0.0, 0.2, 0.35, 0.5, 0.65):
        lift_pad = 0.0015 + PAD_HALF_WIDTH * np.sin(tilt)
        if lift_pad > 0.5 * H:
            break
        for yw in (yaw0, yaw0 + np.pi):  # either finger direction (the gripper is symmetric)
            ok = True
            for z in (p[2] + lift_pad, zt + 0.06):
                _, err = cart.ik(side, (p[0], p[1], z), quat=grasp_quat(yw, tilt), return_error=True)
                ok &= np.linalg.norm(err[:3]) < 0.003 and np.linalg.norm(err[3:]) < 0.03
            if ok:
                tg, yaw = tilt, yw
                break
        if tg is not None:
            break
    if tg is None:
        return dict(ok=False, report=dict(reason="plug out of reach with vertical pads"))
    z_grasp = p[2] + 0.0015 + PAD_HALF_WIDTH * np.sin(tg)
    approach = [dict(pos=tcp0, yaw=0.0, tilt=0.0, t=0.0),
                dict(pos=(p[0], p[1], zt + 0.06), yaw=yaw, tilt=tg, mode="pad", t=3.5),
                dict(pos=(p[0], p[1], z_grasp), yaw=yaw, tilt=tg, mode="pad", t=1.2)]
    carry = [dict(pos=(p[0], p[1], z_grasp), yaw=yaw, tilt=tg, mode="pad", t=0.0),
             dict(pos=(p[0], min(p[1] - 0.06, -0.02), zt + 0.10), yaw=yaw, tilt=max(tg, 0.15), t=2.5),
             dict(pos=(0.5 * (p[0] + bx), 0.5 * (p[1] + by), zt + 0.24), yaw=yaw, tilt=0.3, t=3.0),
             dict(pos=(cx, cy, zt + 0.24), yaw=yaw, tilt=0.3, t=2.5),
             dict(pos=(cx, cy, zt + 0.035), yaw=yaw, tilt=0.15, t=1.5),
             # lay the trailing cable into the tray: along the far wall, then back toward the middle
             dict(pos=(cx, by - iy, zt + 0.035), yaw=yaw, tilt=0.15, t=1.5),
             dict(pos=(bx, by - iy, zt + 0.035), yaw=yaw, tilt=0.15, t=1.0)]
    rx, ry = bx, by - iy
    q0 = cart.qpos()
    path_a, rep_a = plan_cartesian(cart, side, approach, S.DT, q0)
    qa = q0.copy()
    qa[cart.arm_dofs[side]] = path_a[-1]
    path_c, rep_c = plan_cartesian(cart, side, carry, S.DT, qa)
    qc = qa.copy()
    qc[cart.arm_dofs[side]] = path_c[-1]
    retreat = [dict(pos=(rx, ry, zt + 0.035), yaw=yaw, tilt=0.15, t=0.0),
               dict(pos=(rx, ry, zt + 0.20), yaw=yaw, tilt=0.25, t=1.2),
               dict(pos=tuple(tcp0), yaw=0.0, tilt=0.0, t=2.0)]
    path_r, rep_r = plan_cartesian(cart, side, retreat, S.DT, qc)
    rep = dict(approach=rep_a, carry=rep_c, retreat=rep_r, grasp_tilt=tg)
    ok = all(v["max_ik_err_mm"] < 3.0 and v["max_joint_step_rad"] < MAX_IK_STEP for k, v in rep.items() if k != "grasp_tilt")
    return dict(approach=path_a, lift=path_c, retreat=path_r, report=rep, ok=ok, home_tcp=tuple(tcp0))


def grasp_quat(yaw, tilt):
    """Flange quaternion: fingers closing along (cos yaw, sin yaw, 0), approach pitched by `tilt` about the CLOSING
    axis toward (-sin yaw, cos yaw, 0) (away from the cart for |yaw| < pi/2), so the pad faces stay vertical
    (yam_cart.down_quat pitches about world x, which tilts the pads unless they close along x)."""
    from yam_cart import quat_from_mat
    xa = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    za = np.array([0.0, 0.0, -1.0])
    R0 = np.stack([xa, np.cross(za, xa), za], axis=1)
    return quat_from_mat(CM.rot(xa, tilt) @ R0)


PAD_HALF_WIDTH = 0.015  # pad box 8 x 30 x 40 mm: half its width in the pitch plane
PLUG_OPEN_MARGIN = 0.016  # pad gap before closing on a plug = plug width + this
MAX_IK_STEP = 0.15  # rad between IK samples (every 3 steps): catches IK branch flips (0.3-0.6 rad), allows 3 rad/s


def plan_feed(env, name, side, tcp_pose_now):
    """A feed for a cable whose plug is already in its bin: grasp the cable body where it lies on the table 7-24 cm
    beyond the point where it leaves the bin (clear of other cables and of the bin walls), lift it over the rim, lay
    it down inside the bin. Returns (pick, plan) or (None, None)."""
    pts = env.cable_points()
    P = pts[name]
    c = S.CABLE_BY_NAME[name]
    nin = env.cables[name]["n_inner"]
    col = c["color"]
    bp, bq = env.bin_poses()[col]
    Rb = S.quat_to_mat(bq)
    lo, hi = S.bin_inner_box()
    inside = np.all(((P - bp) @ Rb > lo) & ((P - bp) @ Rb < hi), axis=1)
    out_idx = [i for i in range(nin, len(P)) if not inside[i]]
    if not out_idx:
        return None, None
    i_exit = out_idx[0]
    others = np.vstack([Q for n, Q in pts.items() if n != name] + [np.full((1, 3), 1e3)])
    half = np.array(S.BIN["size"][:2]) / 2
    seg = env.cables[name]["framed"].seg
    order = sorted(range(i_exit + int(0.07 / seg), min(len(P) - 2, i_exit + int(0.24 / seg)) + 1),
                   key=lambda i: abs(i - (i_exit + int(0.15 / seg))))
    zt = S.ZT
    rim = zt + S.BIN["size"][2]
    for i in order:
        p = P[i]
        # wherever the cable is: on the table, on other cables, or draped in the air over the rim
        near = np.abs(others[:, 2] - p[2]) < 0.03
        if near.any() and np.linalg.norm(others[near, :2] - p[:2], axis=1).min() < 0.022:
            continue
        lb = np.abs(((p - bp) @ Rb)[:2])
        if p[2] < rim + 0.03 and np.all(lb < half + 0.03):  # pads would hit the bin wall
            continue
        z_g = max(zt + 0.0015, p[2] - c["radius"] - 0.0015)  # pad bottoms just below the cable
        t = P[i + 1] - P[i - 1]
        yaw = float(np.arctan2(t[1], t[0]) + np.pi / 2)
        yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
        drop = bp
        keys_a = [dict(pos=tuple(tcp_pose_now[0]), quat=tcp_pose_now[1], t=0.0),  # start from the actual orientation
                  dict(pos=(p[0], p[1], z_g + 0.06), yaw=yaw, tilt=0.0, mode="pad", t=2.0),
                  dict(pos=(p[0], p[1], z_g), yaw=yaw, tilt=0.0, mode="pad", t=1.2)]
        keys_c = [dict(pos=(p[0], p[1], z_g), yaw=yaw, tilt=0.0, mode="pad", t=0.0),
                  dict(pos=(p[0], p[1], max(zt + 0.13, z_g + 0.04)), yaw=yaw, tilt=0.2, t=1.5),
                  dict(pos=(drop[0], drop[1], zt + 0.13), yaw=yaw, tilt=0.25, t=1.5),
                  dict(pos=(drop[0], drop[1], zt + 0.04), yaw=yaw, tilt=0.15, t=1.0)]
        cart = env.robot
        q0 = cart.qpos()
        try:
            pa, ra = plan_cartesian(cart, side, keys_a, S.DT, q0)
            qa = q0.copy()
            qa[cart.arm_dofs[side]] = pa[-1]
            pc, rc = plan_cartesian(cart, side, keys_c, S.DT, qa)
        except Exception:
            continue
        rep = dict(approach=ra, carry=rc)
        if all(v["max_ik_err_mm"] < 3.0 and v["max_joint_step_rad"] < MAX_IK_STEP for v in rep.values()):
            return dict(cable=name, vertex=int(i), point=p.tolist(), yaw=yaw, side=side, i_exit=int(i_exit)), \
                dict(approach=pa, lift=pc, report=rep)
    return None, None


def min_jerk(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def key_quat(k):
    """Flange orientation of a keyframe: 'pad' = pitched about the closing axis (pads stay vertical, for grasping);
    otherwise yam_cart.down_quat (pitched toward +y about world x: the reach direction, for carrying)."""
    from yam_cart import down_quat
    if "quat" in k:
        return np.asarray(k["quat"], float)
    return grasp_quat(k["yaw"], k["tilt"]) if k.get("mode") == "pad" else down_quat(k["yaw"], k["tilt"])


def plan_cartesian(cart, side, keys, dt, q_start_full, ik_every=3):
    """Joint path through TCP keyframes keys = [dict(pos, yaw, tilt[, mode], t)] (the first is the start pose):
    position along straight lines, orientation by slerp between the keyframe orientations, min-jerk timing; IK every
    `ik_every` steps seeded with the previous solution (no branch jumps), joints linear in between.
    Returns (list of q_arm per step, report)."""
    from scipy.spatial.transform import Rotation as Rot, Slerp
    seed = np.array(q_start_full, dtype=float)
    arm = cart.arm_dofs[side]
    path, worst_err, worst_jump = [], 0.0, 0.0
    q_prev = seed[arm].copy()
    for k0, k1 in zip(keys, keys[1:]):
        n = max(1, int(round(k1["t"] / dt)))
        p0, p1 = np.array(k0["pos"]), np.array(k1["pos"])
        q0w, q1w = key_quat(k0), key_quat(k1)
        sl = Slerp([0.0, 1.0], Rot.from_quat([np.r_[q0w[1:], q0w[0]], np.r_[q1w[1:], q1w[0]]]))
        sols = []
        idx = list(range(ik_every, n + 1, ik_every))
        if not idx or idx[-1] != n:
            idx.append(n)
        for j in idx:
            s = min_jerk(j / n)
            pos = p0 + s * (p1 - p0)
            qx = sl([s]).as_quat()[0]
            q, err = cart.ik(side, pos, quat=np.r_[qx[3], qx[:3]], init_qpos=seed, return_error=True)
            worst_err = max(worst_err, float(np.linalg.norm(err[:3])))
            worst_jump = max(worst_jump, float(np.abs(q - q_prev).max()))
            seed[arm] = q
            sols.append((j, q.copy()))
            q_prev = q.copy()
        j_last = 0
        q_last = path[-1] if path else np.array(q_start_full, dtype=float)[arm]
        for j, q in sols:
            for jj in range(j_last + 1, j + 1):
                a = (jj - j_last) / (j - j_last)
                path.append(q_last + a * (q - q_last))
            j_last, q_last = j, q
    return path, dict(max_ik_err_mm=1e3 * worst_err, max_joint_step_rad=worst_jump)


def demo_plan(env, pick):
    """TCP keyframes for the pick-and-drop and their joint path. Top-down at the cable; the approach axis pitches
    forward (tilt) as the gripper rises, where the YAM Ultra's top-down reach ends."""
    cart, side = env.robot, pick["side"]
    p = np.array(pick["point"])
    yaw = (pick["yaw"] + np.pi / 2) % np.pi - np.pi / 2  # either finger direction works: keep the wrist turn small
    zt = S.ZT
    bx, by = S.BINS[S.CABLE_BY_NAME[pick["cable"]]["color"]]["xy"]
    tcp0, _ = cart.tcp_pose(side)
    z_above = zt + 0.05
    z_grasp = zt + 0.0015  # pad bottoms 1.5 mm above the table: the pads straddle the whole cable
    z_carry = zt + 0.25
    keys = [dict(pos=tcp0, yaw=0.0, tilt=0.0, t=0.0),
            dict(pos=(p[0], p[1], z_above), yaw=yaw, tilt=0.0, t=2.5),
            dict(pos=(p[0], p[1], z_grasp), yaw=yaw, tilt=0.0, t=1.5)]
    approach = keys
    lift = [dict(pos=(p[0], p[1], z_grasp), yaw=yaw, tilt=0.0, t=0.0),
            dict(pos=(p[0], p[1], zt + 0.12), yaw=yaw, tilt=0.15, t=2.0),
            dict(pos=(0.5 * (p[0] + bx), 0.5 * (p[1] + by), z_carry), yaw=yaw, tilt=0.3, t=2.0),
            dict(pos=(bx, by, z_carry), yaw=yaw, tilt=0.3, t=2.0)]
    q0 = cart.qpos()
    path_a, rep_a = plan_cartesian(cart, side, approach, S.DT, q0)
    qa = q0.copy()
    qa[cart.arm_dofs[side]] = path_a[-1]
    path_l, rep_l = plan_cartesian(cart, side, lift, S.DT, qa)
    ql = qa.copy()
    ql[cart.arm_dofs[side]] = path_l[-1]
    retreat = [dict(pos=(bx, by, z_carry), yaw=yaw, tilt=0.3, t=0.0),
               dict(pos=tuple(tcp0), yaw=0.0, tilt=0.0, t=2.5)]
    path_r, rep_r = plan_cartesian(cart, side, retreat, S.DT, ql)
    rep = dict(approach=rep_a, lift=rep_l, retreat=rep_r)
    ok = all(v["max_ik_err_mm"] < 3.0 and v["max_joint_step_rad"] < MAX_IK_STEP for v in rep.values())
    return dict(approach=path_a, lift=path_l, retreat=path_r, report=rep, ok=ok)


# ------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", action="store_true", help="add the YAM Ultra cart (and run the pick demo)")
    ap.add_argument("--no-demo", action="store_true", help="with --robot: cart present but idle")
    ap.add_argument("--t-settle", type=float, default=4.0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--cam", default="front")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--krb-only", action="store_true", help="debug: plain FEM.Rod cables (straight rest shape)")
    ap.add_argument("--max-time", type=float, default=None)
    ap.add_argument("--pinch", action="store_true", help="demo: pinch the bare cable instead of grasping a plug")
    ap.add_argument("--feeds", type=int, default=0,
                    help="max feeds of the trailing cable into the bin (off by default: for a stiff 0.9 m coil they "
                         "did not raise the fraction, and the cable entered the gripper housing mesh)")
    ap.add_argument("--coupling", type=float, default=20.0, help="IPC soft-constraint strength, translation (fingertip "
                    "loop strength*m/m_eff = 2.5 < 4: stable; 100 oscillated, tests/grasp_debug.py)")
    ap.add_argument("--coupling-rot", type=float, default=1000.0, help="... rotation (keeps pad copies upright)")
    ap.add_argument("--contact", default=None, help="IPC contact constitution, e.g. al-ipc")
    ap.add_argument("--dt", default=None, help="sim step, e.g. 1/60 (read before station import)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gs.init(backend=gs.gpu, logging_level="warning")
    env = S.build(robot=args.robot, camera=None if args.no_video else CAMS[args.cam], krb_only=args.krb_only,
                  coupling=args.coupling, coupling_rot=args.coupling_rot,
                  coupler_extra=None if args.contact is None else dict(contact_constitution=args.contact))
    print("cables:", {n: c["info"] for n, c in env.cables.items()}, flush=True)
    frame_check = {}
    drawer = None if args.no_video else TubeDrawer(env)

    rec = export.Recorder(args.out, station="cables", fps=layout.RENDER_FPS)
    for name, c in env.cables.items():
        rec.add_rod(name, radius=c["spec"]["radius"],
                    material=dict(kind="rubber", color=S.COLORS[c["spec"]["color"]], roughness=0.55))
    for name, e in env.plugs.items():
        kind = S.CABLE_BY_NAME[name]["plug"]
        rec.add_rigid(f"plug_{name}", S.plug_info(kind)["visual"],
                      material=dict(kind="plastic", color=S.PL.PLUGS[kind]["color"], roughness=0.4))
    for col in S.BINS:
        rec.add_rigid(f"bin_{col}", env.bin_visual_path,
                      material=dict(kind="plastic", color=S.BIN_TINT[col], roughness=0.45))
        rec.add_rigid(f"bin_{col}_label", env.label_mesh_path,
                      material=dict(kind="paint", color=S.COLORS[col], roughness=0.6))
    if env.robot is not None:
        env.robot.register(rec)

    every = int(round(1.0 / (layout.RENDER_FPS * S.DT)))
    rest = {n: c["framed"].P.copy() for n, c in env.cables.items()}
    seg_rest = {n: c["framed"].cable_rest for n, c in env.cables.items()}
    masses = {n: c["spec"]["rho_line"] * c["framed"].seg for n, c in env.cables.items()}
    frames, log = [], []
    prev, prev_t = None, None
    seg_lo, seg_hi = np.inf, 0.0
    t_wall = time.time()

    def record(t, pts, extra=None):
        nonlocal prev, prev_t
        bp = env.bin_poses()
        rigid = {}
        for col, (p, q) in bp.items():
            rigid[f"bin_{col}"] = (p, q)
            rigid[f"bin_{col}_label"] = (p, q)
        for n, pq in env.plug_poses().items():
            rigid[f"plug_{n}"] = pq
        if env.robot is not None:
            rigid.update(env.robot.poses())
        # the section moulded into the plug is inside the plug body: export the cable from the boot outwards
        rec.frame(t, rigid=rigid, rods={n: P[env.cables[n]["n_inner"]:] for n, P in pts.items()})
        r = dict(t=round(t, 4))
        ke, vmax = 0.0, 0.0
        if prev is not None:
            for n, P in pts.items():
                v = np.linalg.norm(P - prev[n], axis=1) / (t - prev_t)
                ke += 0.5 * masses[n] * float((v ** 2).sum())
                vmax = max(vmax, float(v.max()))
        r["ke"], r["vmax"] = ke, vmax
        r["cables"] = {n: dict(zgap_min=float((P[:, 2] - S.ZT).min() - env.cables[n]["spec"]["radius"]),
                               zmin=float(P[:, 2].min()), xy_min=P[:, :2].min(0).tolist(),
                               xy_max=P[:, :2].max(0).tolist(), nan=bool(np.isnan(P).any()),
                               com=P.mean(0).tolist())
                       for n, P in pts.items()}
        r["bins"] = {c: dict(pos=p.tolist(), quat=q.tolist()) for c, (p, q) in bp.items()}
        if extra:
            r.update(extra)
        log.append(r)
        prev, prev_t = {n: P.copy() for n, P in pts.items()}, t
        if drawer is not None:
            drawer.draw(pts)
            frames.append(np.asarray(A(env.cam.render(rgb=True)[0]), dtype=np.uint8))

    def step_and_watch():
        nonlocal seg_lo, seg_hi
        env.step()
        pts = env.cable_points()
        for n, P in pts.items():
            ratio = np.linalg.norm(np.diff(P, axis=0), axis=1) / seg_rest[n]
            seg_lo, seg_hi = min(seg_lo, float(ratio.min())), max(seg_hi, float(ratio.max()))
        return pts

    # ---- drop + settle
    step = 0
    pts = env.cable_points()
    record(0.0, pts)
    n_settle = int(round(args.t_settle / S.DT))
    if args.max_time is not None:
        n_settle = min(n_settle, int(round(args.max_time / S.DT)))
    for _ in range(n_settle):
        pts = step_and_watch()
        step += 1
        if step == 1:
            frame_check = bin_frame_check(env)
            print("bin frame check:", frame_check, flush=True)
        if step % every == 0:
            record(step * S.DT, pts, dict(seg_ratio=[seg_lo, seg_hi], phase="drop"))
            seg_lo, seg_hi = np.inf, 0.0
        if step % 90 == 0:
            print(f"t={step * S.DT:.2f}s  ke={log[-1]['ke']:.2e} vmax={log[-1]['vmax']:.3f}  "
                  f"wall {time.time() - t_wall:.0f}s", flush=True)
    timing = dict(drop_settle=dict(sim_s=step * S.DT, wall_s=time.time() - t_wall))
    settled = env.cable_points()
    settle_rec = dict(t=step * S.DT, tangle=env.tangle_metric(settled), success=env.success(settled),
                      ke=log[-1]["ke"], vmax=log[-1]["vmax"])
    print("settled:", json.dumps({k: settle_rec["tangle"][k] for k in ("crossings", "contacts", "touching_pairs",
                                                                        "interlocked_pairs", "middle")}), flush=True)
    retention = {}
    for n, P in settled.items():
        rm = kabsch_rmsd(P, rest[n])
        rs = straight_rmsd(rest[n])
        th0, th = CM.turning_angles(rest[n]), CM.turning_angles(P)
        retention[n] = dict(rmsd_mm=1e3 * rm, straight_rmsd_mm=1e3 * rs, retention=1.0 - rm / rs,
                            curvature_ratio=float(th.sum() / th0.sum()),
                            curvature_corr=float(np.corrcoef(th0, th)[0, 1]))
    settle_rec["retention"] = retention

    # ---- robot demo
    demo = None
    if env.robot is not None and not args.no_demo:
        cart = env.robot
        by_plug = bool(env.plugs) and not args.pinch
        cands = plan_pick_plugs(env, settled) if by_plug else plan_pick(env, settled)
        print("pick candidates:", [(c["cable"], round(c["score"], 3)) for c in cands], flush=True)
        pick = plan = None
        for cnd in cands:
            pl_ = demo_plan_plug(env, cnd) if by_plug else demo_plan(env, cnd)
            print("plan", cnd["cable"], pl_["report"], "ok" if pl_["ok"] else "REJECTED", flush=True)
            if pl_["ok"]:
                pick, plan = cnd, pl_
                break
        demo = dict(pick=pick, plan_report=None if plan is None else plan["report"])
        if pick is not None:
            side = pick["side"]

            joint_speed = [0.0]

            def run_path(path, phase, extra_steps=0):
                nonlocal step, seg_lo, seg_hi
                t_ph, s_ph = time.time(), step
                q_last = cart.q_target[cart.arm_dofs[side]].copy()
                for qk in list(path):
                    joint_speed[0] = max(joint_speed[0], float(np.abs(np.asarray(qk) - q_last).max() / S.DT))
                    q_last = np.asarray(qk)
                for qk in list(path) + [None] * extra_steps:
                    if qk is not None:
                        cart.set_arm_target(side, qk)
                    pts_ = step_and_watch()
                    step += 1
                    if step % every == 0:
                        tcp, fq = cart.tcp_pose(side)
                        extra_ = {}
                        if by_plug:  # plug pose in the flange frame (slip while held)
                            pp, pq = env.plug_poses()[pick["cable"]]
                            Rf = S.quat_to_mat(fq)
                            fl = A(cart.ee_link[side].get_pos()).astype(np.float64)
                            extra_ = dict(plug_in_flange=(Rf.T @ (pp - fl)).tolist(),
                                          plug_R_in_flange=(Rf.T @ S.quat_to_mat(pq)).tolist())
                        record(step * S.DT, pts_, dict(seg_ratio=[seg_lo, seg_hi], phase="demo", demo_phase=phase, **extra_,
                                                       tcp=tcp.tolist(), width=float(cart.gripper_width(side)),
                                                       grasp_vertex=pts_[pick["cable"]][pick.get("vertex", 0)].tolist(),
                                                       finger_force=cart.finger_force(side).tolist(),
                                                       grasped=bool(cart.grasped(side))))
                        seg_lo, seg_hi = np.inf, 0.0
                    if step % 90 == 0:
                        print(f"t={step * S.DT:.2f}s demo {phase} wall {time.time() - t_wall:.0f}s", flush=True)
                tm_ = timing.setdefault(phase, dict(sim_s=0.0, wall_s=0.0))
                tm_["sim_s"] += (step - s_ph) * S.DT
                tm_["wall_s"] += time.time() - t_ph

            if by_plug:
                cart.set_gripper(side, min(0.09, pick["width"] + PLUG_OPEN_MARGIN))
                run_path(plan["approach"], "approach", extra_steps=15)
                cart.grasp(side, force=PLUG_GRIP_FORCE)
                run_path([], "close", extra_steps=int(1.0 / S.DT))
            else:
                cart.set_gripper(side, 0.045)
                run_path(plan["approach"], "approach", extra_steps=20)
                grip_close(cart, side, S.CABLE_BY_NAME[pick["cable"]]["radius"])
                run_path([], "close", extra_steps=int(1.6 / S.DT))
            run_path(plan["lift"], "lift_carry", extra_steps=int(0.3 / S.DT))
            cart.open(side)
            run_path([], "release", extra_steps=int(1.0 / S.DT))
            run_path(plan["retreat"], "retreat", extra_steps=int(1.0 / S.DT))
            # feeds: carry the part of the cable still outside the bin in by its body (as the Newton version)
            demo["feeds"] = []
            col = S.CABLE_BY_NAME[pick["cable"]]["color"]
            for k_feed in range(args.feeds if by_plug else 0):
                frac = env.success()["per_cable"][pick["cable"]]["fractions"][col]
                if frac >= 0.75:
                    break
                fpick, fplan = plan_feed(env, pick["cable"], side, cart.tcp_pose(side))
                print("feed", k_feed, "fraction", round(frac, 3), fpick, None if fplan is None else fplan["report"],
                      flush=True)
                if fpick is None:
                    break
                cart.set_gripper(side, 0.04)
                run_path(fplan["approach"], f"feed{k_feed}_approach", extra_steps=10)
                grip_close(cart, side, S.CABLE_BY_NAME[pick["cable"]]["radius"])
                run_path([], f"feed{k_feed}_close", extra_steps=int(1.2 / S.DT))
                run_path(fplan["lift"], f"feed{k_feed}_carry", extra_steps=int(0.3 / S.DT))
                cart.open(side)
                run_path([], f"feed{k_feed}_release", extra_steps=int(0.8 / S.DT))
                up, upq = cart.tcp_pose(side)
                pr, _ = plan_cartesian(cart, side, [dict(pos=tuple(up), quat=upq, t=0.0),
                                                    dict(pos=(up[0], up[1], S.ZT + 0.16), yaw=fpick["yaw"], tilt=0.2,
                                                         t=1.0)], S.DT, cart.qpos())
                run_path(pr, f"feed{k_feed}_up", extra_steps=5)
                demo["feeds"].append(dict(pick=fpick, fraction_before=frac,
                                          fraction_after=env.success()["per_cable"][pick["cable"]]["fractions"][col]))
            if demo["feeds"]:
                here, hq = cart.tcp_pose(side)
                pr, _ = plan_cartesian(cart, side, [dict(pos=tuple(here), quat=hq, t=0.0),
                                                    dict(pos=tuple(plan["home_tcp"]), yaw=0.0, tilt=0.0, t=2.0)],
                                       S.DT, cart.qpos())
                run_path(pr, "return", extra_steps=int(0.5 / S.DT))
            demo["max_arm_joint_speed_rad_s"] = joint_speed[0]
            final = env.cable_points()
            demo["success"] = env.success(final)
            demo["tangle_after"] = {k: v for k, v in env.tangle_metric(final).items() if k != "pairs"}

    wall = time.time() - t_wall
    rec.close(extra_meta=dict(dt=S.DT, camera=dict(eye=[0.42, -0.78, 1.18], target=[0.0, -0.06, 0.77], lens=35.0),
                              cables={n: dict(color=c["spec"]["color"], EI=c["spec"]["EI"],
                                                       length=float(c["framed"].cable_rest.sum()))
                                               for n, c in env.cables.items()}))
    for v in timing.values():
        v["wall_per_sim_s"] = v["wall_s"] / max(v["sim_s"], 1e-9)
    import subprocess
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    n_jobs = len(subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.split())
    json.dump(dict(log=log, settle=settle_rec, demo=demo, wall_s=wall, args=vars(args), bin_frame_check=frame_check,
                   timing=timing, gpu=dict(name=gpu, compute_processes=n_jobs),
                   cable_info={n: c["info"] for n, c in env.cables.items()}),
              open(os.path.join(args.out, "log.json"), "w"), default=float)
    if frames:
        import imageio.v2 as imageio
        imageio.mimwrite(os.path.join(args.out, "preview.mp4"), frames, fps=layout.RENDER_FPS, quality=8,
                         macro_block_size=8)
        imageio.imwrite(os.path.join(args.out, "still_start.png"), frames[0])
        imageio.imwrite(os.path.join(args.out, "still_settled.png"), frames[min(len(frames) - 1,
                                                                                  int(args.t_settle * 30))])
        imageio.imwrite(os.path.join(args.out, "still_end.png"), frames[-1])
        drawer.draw(env.cable_points())
        for name in ("top", "close"):
            c = CAMS[name]
            env.cam.set_pose(pos=c["pos"], lookat=c["lookat"], up=c.get("up", (0.0, 0.0, 1.0)))
            imageio.imwrite(os.path.join(args.out, f"still_{name}.png"),
                            np.asarray(A(env.cam.render(rgb=True)[0]), dtype=np.uint8))
    print(f"done in {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
