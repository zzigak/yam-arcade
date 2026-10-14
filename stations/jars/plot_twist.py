"""Plot lid height above the rim vs lid rotation from the twist test log -> out/jars/twist_test/descent.png."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "out", "jars", "twist_test")
P = json.load(open(os.path.join(HERE, "assets", "params.json")))["parts"]
log = json.load(open(os.path.join(OUT, "run_log.json")))
ph = np.array([r["phase"] for r in log])
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
cols = {"rest": "0.6", "tighten": "#c0392b", "hold": "0.3", "unscrew": "#2471a3", "release": "0.6"}
for ax, n in zip(axs, ("spice", "jam", "mason")):
    h = np.array([r[f"rel_{n}"]["h"] for r in log]) * 1e3
    yaw = np.array([r[f"rel_{n}"]["yaw"] for r in log])
    turns = -(yaw - yaw[0]) / (2 * np.pi)  # clockwise from above = tightening = positive
    for p, c in cols.items():
        m = ph == p
        ax.plot(turns[m], h[m], ".", ms=1.5, color=c, label=p)
    pitch = P[n]["pitch"] * 1e3
    x = np.linspace(-0.6, 1.2, 10)
    h_rest = h[np.where(ph == "rest")[0][-1]]
    ax.plot(x, h_rest - pitch * x, "k--", lw=0.7, label=f"pitch {pitch:.2f} mm/turn")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lid rotation, turns clockwise from start")
    ax.set_ylabel("lid seat plane above jar rim, mm")
    ax.set_ylim(-0.5, max(h) + 0.5)
    ax.set_title(f"{n}: P = {pitch:.2f} mm")
axs[0].legend(fontsize=7, markerscale=5)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "descent.png"), dpi=120)
print("wrote", os.path.join(OUT, "descent.png"))
