# Copied from ~/projects/franka-deformables/calib/elastica.py (heavy-elastica reference for the cantilever fit).
"""Heavy cantilever (inextensible elastica under its own weight): the reference for the bending calibration.

theta(s) is the slope angle below horizontal along arc length s, clamped flat at the ledge edge
(theta(0) = 0) and moment-free at the tip (theta'(L) = 0). Moment balance with the horizontal lever
arm gives
    B theta''(s) = -W (L - s) cos(theta(s))
with W the weight per unit area (N/m^2) and B the bending rigidity per unit width (N m). We shoot
from the TIP: for a hanging strip the tip angle lies in [0, 90 deg) and theta(0) is monotonic in it,
which a shot from the clamp is not once the strip is floppy.
"""
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def _solve(B, W, L, n=400):
    k = W / B

    def rhs(s, y):
        return [y[1], -k * (L - s) * math.cos(y[0])]

    def from_tip(th_tip, s_eval=None):
        return solve_ivp(rhs, (L, 0.0), [th_tip, 0.0], t_eval=s_eval, rtol=1e-10, atol=1e-12)

    # theta(0) is not monotonic in the tip angle for floppy strips; the physical root is the one whose
    # slope rises monotonically from the clamp to the tip, so scan for sign changes and keep that one.
    g = lambda a: from_tip(a).y[0, -1]  # noqa: E731
    grid = np.linspace(0.0, math.pi / 2 - 1e-6, 200)
    vals = [g(a) for a in grid]
    th_tip = None
    for a0, a1, v0, v1 in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if v0 == 0 or v0 * v1 < 0:
            r = brentq(g, a0, a1) if v0 != 0 else a0
            th = from_tip(r, np.linspace(L, 0.0, 200)).y[0]
            if np.all(np.diff(th[::-1]) >= -1e-9):
                th_tip = r
    assert th_tip is not None, "no monotonic elastica solution"
    s = np.linspace(L, 0.0, n)
    th = from_tip(th_tip, s).y[0][::-1]
    s = s[::-1]
    ds = s[1] - s[0]
    mid = 0.5 * (th[1:] + th[:-1])
    x = np.concatenate([[0.0], np.cumsum(np.cos(mid) * ds)])
    z = np.concatenate([[0.0], np.cumsum(np.sin(mid) * ds)])
    return x, z


def tip_drop(B, W, L):
    """(tip drop below the ledge in m, chord angle in degrees)."""
    x, z = _solve(B, W, L)
    return z[-1], math.degrees(math.atan2(z[-1], x[-1]))


def profile(B, W, L):
    """(x, z) of the centreline from the ledge edge, z = drop (positive down)."""
    return _solve(B, W, L)


def rigidity_for_drop(drop, W, L):
    """Invert tip_drop: the B that makes an overhang L droop by `drop`."""
    return math.exp(brentq(lambda lb: tip_drop(math.exp(lb), W, L)[0] - drop, math.log(W * 0.008 ** 3), math.log(1e-1)))


if __name__ == "__main__":
    W = 0.30 * 9.81
    c = 0.024
    B = W * c ** 3
    print(f"Peirce check: overhang 2c -> chord angle {tip_drop(B, W, 2 * c)[1]:.1f} deg (Peirce: 41.5)")
    for L in (0.04, 0.05, 0.06, 0.08):
        d, ang = tip_drop(B, W, L)
        print(f"B={B:.2e} N m, overhang {L * 100:.0f} cm: tip drop {d * 100:.2f} cm, chord {ang:.1f} deg")
    for cc in (0.012, 0.048):
        print(f"c = {cc * 100:.1f} cm, overhang 5 cm: tip drop {tip_drop(W * cc ** 3, W, 0.05)[0] * 100:.2f} cm")
