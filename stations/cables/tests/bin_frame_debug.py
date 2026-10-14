"""Tiny GPU debug: where is libuipc's affine-body frame of a mesh bin relative to its Genesis link / mesh frame?"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import station as S, cable_model as CM
import genesis as gs, trimesh
from genesis.utils.misc import tensor_to_array as A
gs.init(backend=gs.gpu, logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=S.DT), rigid_options=gs.options.RigidOptions(enable_collision=False),
                 coupler_options=gs.options.IPCCouplerOptions(contact_d_hat=CM.D_HAT, contact_resistance=1e7,
                 newton_semi_implicit_enable=False, enable_rigid_rigid_contact=True), show_viewer=False)
scene.add_entity(gs.morphs.Box(pos=(0, 0, S.ZT - 0.015), size=(1.2, 0.7, 0.03), fixed=True), material=gs.materials.Rigid(coup_type="ipc_only"))
obj = os.path.join(S.ASSETS, "bin.obj"); M = trimesh.load(obj, process=False)
b = scene.add_entity(gs.morphs.Mesh(file=obj, pos=(0.39, -0.04, S.ZT + 0.002), fixed=False, convexify=False, decimate=False),
                     material=gs.materials.Rigid(coup_type="ipc_only", rho=S.BIN["mass"] / M.volume))
scene.build()
link = b.base_link
g = link.geoms[0]
print("mesh AABB", M.bounds.tolist())
print("geom init_pos", A(g.init_pos) if hasattr(g, "init_pos") else None, "init_verts AABB", np.asarray(g.init_verts).min(0), np.asarray(g.init_verts).max(0))
print("link pos", A(link.get_pos()), "inertial_pos", getattr(link, "inertial_pos", None), "COM mesh", M.center_mass)
from uipc.backend import SceneVisitor
from uipc.geometry import SimplicialComplexSlot
def slot_info():
    for slot in SceneVisitor(scene.sim.coupler._ipc_scene).geometries():
        if not isinstance(slot, SimplicialComplexSlot): continue
        gg = slot.geometry(); st = gg.meta().find("solver_type")
        if st is None or st.view()[0] != "rigid": continue
        if int(gg.meta().find("link_idx").view()[0]) != link.idx: continue
        L = gg.positions().view().squeeze(-1); Tm = gg.transforms().view()[0]
        return L, Tm
L, Tm = slot_info()
print("slot local AABB", L.min(0), L.max(0)); print("slot T\n", Tm)
for i in range(3):
    scene.step()
    L, Tm = slot_info()
    abd = scene.sim.coupler._abd_data_by_link[link][0].transform
    print(f"step {i+1}: slot T t={Tm[:3,3]}  abd t={abd[:3,3]}  genesis link={A(link.get_pos())}  slot local AABB min {L.min(0)}")
gv = A(g.get_verts())
print("genesis collision geom world z range", gv[:, 2].min(), gv[:, 2].max())
for vg in link.vgeoms:
    try:
        vv = A(vg.get_vverts()); print("vgeom world z range", vv[:, 2].min(), vv[:, 2].max())
    except Exception as e:
        print("vgeom", e)
L, Tm = slot_info()
W = L @ Tm[:3, :3].T + Tm[:3, 3]
print("libuipc world z range", W[:, 2].min(), W[:, 2].max())
