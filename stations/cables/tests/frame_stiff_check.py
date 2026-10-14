import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cable_model as CM
cases = [("straight", np.stack([np.linspace(0, 0.5, 51), np.zeros(51), np.zeros(51)], 1), 0.003, 5e-4),
         ("coil", CM.resample(CM.rest_curve(dict(shape="coil", length=0.8, R=0.055, turns=2.0, pitch=0.013)), 0.014)[0], 0.0045, 3e-3),
         ("ubend", CM.resample(CM.rest_curve(dict(shape="ubend", length=0.6, R=0.05)), 0.013)[0], 0.004, 1.5e-3)]
for name, P, r, EI in cases:
    for fs in (3.0, 10.0, 30.0):
        fr = CM.Framed(P, r, EI, frame_stiff=fs)
        print(name, fs, f"kframe {fr.kappa_frame:.1e}", "relaxed EI_1/EI_2:", round(CM.relaxed_bending_EI(fr, axis='b')/EI, 3), round(CM.relaxed_bending_EI(fr, axis='n')/EI, 3), "springs", len(fr.springs))
