import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cable_model as CM
P = np.stack([np.linspace(0, 0.5, 51), np.zeros(51), np.zeros(51)], 1)
for hf in (0.5, 1.0, 1.5, 2.0):
    fr = CM.Framed(P, 0.003, 5e-4, h_frac=hf)
    m = fr.effective_moduli()
    print(hf, f"fit {fr.fit_residual:.3f}", {k: round(v/(5e-4 if k!='GJ' else fr.GJ), 3) for k, v in m.items()})
s = dict(shape="coil", length=0.8, R=0.055, turns=2.0, pitch=0.013)
P, l = CM.resample(CM.rest_curve(s), 0.014)
fr = CM.Framed(P, 0.0045, 3e-3)
print("coil", fr.fit_residual, fr.effective_moduli(), fr.kappa[fr.family==1].max())
