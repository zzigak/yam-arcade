"""Measure linear_4310 fingertip pads + palm from the meshes and write assets/robots/gripper_collision.json:
box collision geoms (half sizes + centres) in each tip body frame and in the gripper mount frame.

Finger joint q (joint7/joint8): 0 = closed (distal pads meet), 0.0475 = fully open. The distal 4 cm of each tip is
the pad; each collision pad is recessed RECESS behind the mesh face so the two pads keep a gap at q = 0 (libuipc
rejects touching link geometry at init), and the palm box stops 2 mm short of the finger carriages."""
import json, os, sys
import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECESS = 0.0010
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "assets/robots/yam_cart_yam.xml"))
d = mujoco.MjData(m)


def body_verts(gname):
    g = m.geom(gname); mid = g.dataid[0]
    vs = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    R = np.zeros(9); mujoco.mju_quat2Mat(R, g.quat)
    return vs @ R.reshape(3, 3).T + g.pos


def pose(b):
    return d.xpos[m.body(b).id].copy(), d.xmat[m.body(b).id].reshape(3, 3).copy()


d.qpos[:] = 0.0
d.qpos[m.joint("left_joint7").qposadr[0]] = d.qpos[m.joint("left_joint8").qposadr[0]] = 0.0475
mujoco.mj_forward(m, d)
pg, Rg = pose("left_gripper")  # mount frame: +z = approach
out = {}
for t in ("tip_left", "tip_right"):
    pb, Rb = pose(f"left_{t}")
    W = body_verts(f"left_{t}_visual") @ Rb.T + pb
    G = (W - pg) @ Rg  # mount frame
    zmax = G[:, 2].max()
    distal = G[G[:, 2] > zmax - 0.040]
    lo, hi = distal.min(0), distal.max(0)
    # closing axis = mount x (tip_left sits at -x when open?): inner face is the one facing the gripper axis x = 0
    if (lo[0] + hi[0]) < 0:  # finger on -x side: inner face = hi[0]
        x0, x1 = hi[0] - RECESS - 0.008, hi[0] - RECESS
    else:
        x0, x1 = lo[0] + RECESS, lo[0] + RECESS + 0.008
    y_half = min(0.015, 0.5 * (hi[1] - lo[1]))
    yc = 0.5 * (lo[1] + hi[1])
    z0, z1 = zmax - 0.040, zmax - 0.0005
    c_g = np.array([0.5 * (x0 + x1), yc, 0.5 * (z0 + z1)])
    half_g = np.array([0.5 * (x1 - x0), y_half, 0.5 * (z1 - z0)])
    # into the tip body frame (box axes stay aligned with the mount frame -> rotate half sizes)
    Rbg = Rb.T @ Rg  # mount axes expressed in tip body frame
    c_b = Rb.T @ (pg + Rg @ c_g - pb)
    qb = np.zeros(4); mujoco.mju_mat2Quat(qb, Rbg.flatten())
    out[t] = [dict(pos=c_b.tolist(), half=half_g.tolist(), quat=qb.tolist())]
    print(t, "distal bounds (mount frame)", lo.round(4), hi.round(4), "pad centre (mount)", c_g.round(4), "half", half_g.round(4))
    print("  mount axes in tip frame", Rbg.round(4).tolist())
H = body_verts("left_gripper_visual")
lo, hi = H.min(0), H.max(0)
# finger carriages start at:
carriage_z = min(((body_verts(f"left_{t}_visual") @ pose(f"left_{t}")[1].T + pose(f"left_{t}")[0] - pg) @ Rg)[:, 2].min()
                 for t in ("tip_left", "tip_right"))
z1 = carriage_z - 0.002
out["palm"] = [dict(pos=[0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + z1)],
                    half=[0.5 * (hi[0] - lo[0]) - 0.001, 0.5 * (hi[1] - lo[1]) - 0.001, 0.5 * (z1 - lo[2])])]
print("housing", lo.round(4), hi.round(4), "carriage z", round(carriage_z, 4), "palm", out["palm"])
out["note"] = __doc__
json.dump(out, open(os.path.join(ROOT, "assets/robots/gripper_collision.json"), "w"), indent=1)
