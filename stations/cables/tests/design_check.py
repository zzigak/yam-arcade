"""CPU check of the framed-rod design: per-hinge fit residual and whole-cable EI (two planes) and GJ vs target."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cable_model as CM
specs = [
 dict(shape="straight", length=0.5, radius=0.003, seg=0.01, EI=5e-4),
 dict(shape="ubend", length=0.6, R=0.05, radius=0.004, seg=0.013, EI=1.5e-3),
 dict(shape="coil", length=0.8, R=0.055, turns=2.0, pitch=0.013, radius=0.0045, seg=0.014, EI=3e-3),
 dict(shape="scurve", length=0.7, R=0.08, angle=np.radians(120), radius=0.003, seg=0.01, EI=5e-4),
 dict(shape="wave", length=0.5, R=0.04, angle=np.radians(100), n_arcs=4, radius=0.002, seg=0.009, EI=4e-5),
 dict(shape="loop", length=0.6, R=0.045, pitch=0.012, radius=0.003, seg=0.01, EI=4e-4),
]
for s in specs:
    t0 = time.time()
    if s["shape"] == "straight":
        P = np.stack([np.linspace(0, s["length"], 51), np.zeros(51), np.zeros(51)], 1)
    else:
        P, l = CM.resample(CM.rest_curve(s), s["seg"])
    fr = CM.Framed(P, s["radius"], s["EI"])
    m = fr.effective_moduli()
    print(f"{s['shape']:8s} n={fr.n} springs={len(fr.springs)} fit_res={fr.fit_residual:.2e} EI1={m['EI_1']/s['EI']:.3f} "
          f"EI2={m['EI_2']/s['EI']:.3f} GJ={m['GJ']/fr.GJ:.3f} k_hinge=[{fr.kappa[fr.family==1].min():.1e},{fr.kappa[fr.family==1].max():.1e}] "
          f"k_frame={fr.kappa_frame:.1e}  {time.time()-t0:.1f}s")
