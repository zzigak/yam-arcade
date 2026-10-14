"""The four containers of the water station (procedural stand-ins, see NOTES.md for the numbers' justification).

Container frame: origin at the centre of the base on the support plane, +z up. Pitcher spout along +x, handle along -x;
cup handles along -x.
"""
import numpy as np

from geometry import Container, ContainerSpec, Handle, Lid, Spout, rounded_path

# Brita-style pitcher: rounded-rectangle plan, clear SAN/PP-like plastic (rho 1050), 2.5 mm wall, ~2.1 L to the brim,
# lid over the back 3/4 of the top, V-lip spout at +x, D-shaped handle at -x with a 3.5 cm finger gap.
PITCHER = ContainerSpec(
    name="pitcher",
    z_floor=0.004,
    z_rim=0.210,
    prof_z=(0.004, 0.060, 0.120, 0.150, 0.170, 0.185, 0.198, 0.210),
    prof_a=tuple(np.interp([0.004, 0.060, 0.120, 0.150, 0.170, 0.185, 0.198, 0.210], [0.004, 0.210], [0.059, 0.063])),
    prof_b=tuple(np.interp([0.004, 0.060, 0.120, 0.150, 0.170, 0.185, 0.198, 0.210], [0.004, 0.210], [0.044, 0.047])),
    wall=0.0025,
    density=1050.0,
    plan_n=3.2,
    spout=Spout(length=0.022, half_angle=0.55, z_start=0.140, power=1.5),
    lid=Lid(x_cut=0.030, t_below=0.006, t_above=0.005),
    handle=Handle(
        path=rounded_path([(-0.059, 0.192), (-0.110, 0.192), (-0.110, 0.068), (-0.056, 0.052)], radius=0.018),
        half_w=0.011,
        half_t=0.008,
    ),
    coll_n_theta=40,
    coll_z=(0.004, 0.080, 0.140, 0.165, 0.185, 0.198, 0.210),
    rgba=(0.80, 0.88, 0.95, 0.35),
)

# Mug: stoneware (rho 2300), 4.5 mm wall, 8 mm base, inner diameter 71 mm x 76.5 mm deep -> ~303 ml to the brim.
MUG = ContainerSpec(
    name="mug",
    z_floor=0.008,
    z_rim=0.0845,
    prof_z=(0.008, 0.0845),
    prof_a=(0.0355, 0.0355),
    prof_b=None,
    wall=0.0045,
    density=2300.0,
    handle=Handle(
        path=rounded_path([(-0.0375, 0.073), (-0.064, 0.071), (-0.066, 0.030), (-0.0375, 0.022)], radius=0.014),
        half_w=0.0055,
        half_t=0.0065,
    ),
    coll_z=(0.008, 0.040, 0.0845),
    rgba=(0.20, 0.36, 0.55, 1.0),
)

# Tall tumbler: soda-lime glass (rho 2500), 3 mm wall, 12 mm heavy base, tapered 54 -> 66 mm inner diameter,
# 125 mm deep -> ~355 ml to the brim.
TUMBLER = ContainerSpec(
    name="tumbler",
    z_floor=0.012,
    z_rim=0.137,
    prof_z=(0.012, 0.137),
    prof_a=(0.027, 0.033),
    prof_b=None,
    wall=0.003,
    density=2500.0,
    coll_z=(0.012, 0.055, 0.095, 0.137),
    rgba=(0.90, 0.95, 0.97, 0.25),
)

# Teacup: porcelain (rho 2400), 2.8 mm wall, wide shallow flared bowl (87 mm rim, 36 mm deep) on a foot ring,
# ~150 ml to the brim.
_TC_Z = (0.0095, 0.0125, 0.0159, 0.0201, 0.0261, 0.0329, 0.0397, 0.0456)
_TC_R = (0.0190, 0.0249, 0.0295, 0.0333, 0.0371, 0.0399, 0.0421, 0.0435)
TEACUP = ContainerSpec(
    name="teacup",
    z_floor=_TC_Z[0],
    z_rim=_TC_Z[-1],
    prof_z=_TC_Z,
    prof_a=_TC_R,
    prof_b=None,
    wall=0.0028,
    density=2400.0,
    foot=(0.0215, 0.0068),
    handle=Handle(
        path=rounded_path([(-0.0430, 0.0395), (-0.0590, 0.0385), (-0.0565, 0.0205), (-0.0315, 0.0160)], radius=0.008),
        half_w=0.0042,
        half_t=0.0048,
    ),
    coll_n_theta=28,
    rgba=(0.96, 0.95, 0.92, 1.0),
)

PROCEDURAL = {s.name: s for s in (PITCHER, MUG, TUMBLER, TEACUP)}

# Which cups the station uses: "sam3d" = the SAM 3D meshes (assets/sam3d/<name>/) as visuals with the convex-piece
# collision fitted to their measured cavities (sam3d_fit.py); "procedural" = the stand-ins above.
CUPS = "sam3d"
_FITTED = {}


def spec(name):
    if CUPS == "sam3d" and name in ("mug", "tumbler", "teacup"):
        if name not in _FITTED:
            from sam3d_fit import fit
            _FITTED[name] = fit(name)[0]
        return _FITTED[name]
    return PROCEDURAL[name]


ALL = PROCEDURAL   # names; use spec(name) for the active definition


def build(name):
    return Container(spec(name))
