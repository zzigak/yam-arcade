"""CPU (rigid only): YAM Ultra cart at home vs the initial cables and the bins; reach to the pile and over the bins."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import station as S
import genesis as gs
from genesis.utils.misc import tensor_to_array as A
from yam_cart import add_cart
from scipy.spatial import Delaunay
gs.init(backend=gs.cpu, logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=S.DT), show_viewer=False)
cart = add_cart(scene, gs, "yam_ultra", ipc=False)
scene.build(); cart.after_build()
e = cart.entity
polys = S.initial_polylines()
allp = np.vstack(list(polys.values()))
worst = []
for link in e.links:
    vs = [A(g.get_verts()) for g in link.geoms]
    if not vs: continue
    V = np.vstack(vs)
    lo, hi = V.min(0), V.max(0)
    # distance from cable points to the link AABB
    d = np.linalg.norm(np.maximum(0, np.maximum(lo - allp, allp - hi)), axis=1).min()
    worst.append((d, link.name, lo.round(3).tolist(), hi.round(3).tolist()))
worst.sort()
for w in worst[:6]: print("cable-link AABB dist %.3f %s lo %s hi %s" % w)
# bins vs robot
for col, b in S.BINS.items():
    blo = np.array([b["xy"][0] - S.BIN["size"][0]/2, b["xy"][1] - S.BIN["size"][1]/2, S.ZT])
    bhi = np.array([b["xy"][0] + S.BIN["size"][0]/2, b["xy"][1] + S.BIN["size"][1]/2, S.ZT + S.BIN["size"][2]])
    dmin = min(np.linalg.norm(np.maximum(0, np.maximum(blo - w[3], np.array(w[2]) - bhi))) for w in [(0, 0, l[2], l[3]) for l in worst])
    print(col, "bin vs nearest robot link AABB gap", round(float(dmin), 3))
for side, pt in (("left", (-0.39, -0.04, S.ZT + 0.22)), ("right", (0.39, -0.04, S.ZT + 0.22)),
                 ("left", (-0.10, -0.05, S.ZT + 0.005)), ("right", (0.10, -0.05, S.ZT + 0.005)),
                 ("right", (0.15, 0.10, S.ZT + 0.005)), ("left", (-0.15, 0.10, S.ZT + 0.005)), ("right", (0.0, -0.15, S.ZT + 0.005))):
    for yaw in (0.0, 0.8, 1.57):
        q, tilt = cart.ik_grasp(side, pt, yaw=yaw)
        print(side, pt, "yaw", yaw, "->", "tilt", tilt)
        if tilt is not None: break
print("home TCP", [cart.tcp_pose(s)[0].round(3).tolist() for s in ("left", "right")])
