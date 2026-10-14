"""Where does the framed U-bend's small deformation after landing come from? Three U-bends side by side on the floor
plane: dropped 3 cm (mu 0.5), placed 2 mm above (mu 0.5), dropped 3 cm onto a near-frictionless floor (mu 0.02).
Logs per-hinge turning angles and RMSD vs rest over time; out/cables/tests/ubend_diag.json."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import cable_model as CM, station as S
import genesis as gs
from genesis.utils.misc import tensor_to_array as A
sys.path.insert(0, HERE)
from rest_shape_test import kabsch_rmsd

mu_floor = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
gs.init(backend=gs.gpu, logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=S.DT), rigid_options=gs.options.RigidOptions(enable_collision=False),
                 coupler_options=gs.options.IPCCouplerOptions(contact_d_hat=CM.D_HAT, contact_resistance=1e7,
                                                              newton_semi_implicit_enable=False), show_viewer=False)
scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=1.0))
spec = S.CABLE_BY_NAME["hdmi_ubend"]
P0, _ = CM.resample(CM.rest_curve(spec["shape"]), spec["seg"])
# combined friction = sqrt(mu_cable * mu_floor) with mu_floor = 1: 0.5 and 0.02
cases = {"drop_mu0.5": (-0.4, 0.03, 0.25), "placed_mu0.5": (0.0, 0.002, 0.25), "drop_mu0.02": (0.4, 0.03, 0.0004)}
ents = {}
for name, (x, dz, mu) in cases.items():
    P = CM.place(P0, (x, 0.0), 90.0, spec["radius"] + dz)
    fr = CM.Framed(P, spec["radius"], spec["EI"])
    h = CM.add_framed_cable(scene, gs, fr, spec["rho_line"], mu, (0.1, 0.4, 0.8))
    ents[name] = (fr, h)
scene.build()
log = []
for i in range(int(4.0 / S.DT) + 1):
    if i:
        scene.step()
    if i % 9 == 0:
        rec = dict(t=round(i * S.DT, 3))
        for name, (fr, h) in ents.items():
            X = A(h["cable"].get_state().pos).reshape(-1, 3).astype(np.float64)
            th0, th = CM.turning_angles(fr.P), CM.turning_angles(X)
            G = [A(g.get_state().pos).reshape(-1, 3).astype(np.float64) for g in h["ghosts"]]
            # frame roll: angle of ghost 0 about the edge, relative to its rest direction transported with the edge
            rec[name] = dict(rmsd_mm=1e3 * kabsch_rmsd(X, fr.P), curv=float(th.sum() / th0.sum()),
                             dth_deg=np.degrees(th - th0).round(2).tolist(), zmin=float(X[:, 2].min()),
                             ghost_z_mean=float(np.mean([g[:, 2].mean() for g in G])), cable_z_mean=float(X[:, 2].mean()))
        log.append(rec)
        if i % 90 == 0:
            print(rec["t"], {n: (round(v["rmsd_mm"], 2), round(v["curv"], 4), round(1e3 * (v["ghost_z_mean"] - v["cable_z_mean"]), 2)) for n, v in rec.items() if n != "t"}, flush=True)
json.dump(log, open(os.path.join(S.ROOT, "out", "cables", "tests", "ubend_diag.json"), "w"))
