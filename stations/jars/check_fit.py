"""Static geometric checks of the procedural threads (no physics).

For each jar/lid pair, place the collision lid along its screw path and measure the true minimum distance between
the two collision meshes (vertex-to-triangle both ways, plus a penetration test via signed distance):
  - seated design pose (lid origin at rim, riding on its lower flank): distance ~ 0, no penetration
  - lifted by G_AXIAL (lid centred in the groove), then screwed out n turns along the RIGHT-handed helix:
    the clearance must stay at the designed flank gap (G_AXIAL * cos 30 deg = 0.346 mm) for every n
  - the same path with the WRONG hand must collide (sanity check that the helices really interlock)
  - lids of the wrong jar: a mismatched lid cannot be centred on another jar's mouth without collision or falls over it

Run with ~/projects/yam-arcade/.venvs/jars/bin/python; writes assets/fit_check.json.
"""
import json
import math
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets")


def load(fn):
    """In millimetres: trimesh's closest-point code has absolute tolerances that break on sub-mm triangles in metres."""
    m = trimesh.load(os.path.join(A, fn), process=False)
    m.apply_scale(1000.0)
    return m


def min_dist(jar, lid, T):
    """Min distance between jar mesh and lid mesh transformed by T (all in mm). Negative = penetration depth."""
    lv = trimesh.transform_points(lid.vertices, T)
    # restrict to the neighbourhood of the neck to keep it fast
    lo, hi = jar.bounds
    _, d1, _ = trimesh.proximity.closest_point(jar, lv)
    Tinv = np.linalg.inv(T)
    jv = trimesh.transform_points(jar.vertices, Tinv)
    near = np.linalg.norm(jv[:, :2], axis=1) < np.linalg.norm(lid.vertices[:, :2], axis=1).max() + 2.0
    jv = jv[near & (jv[:, 2] > lid.vertices[:, 2].min() - 2.0)]
    if len(jv):
        _, d2, _ = trimesh.proximity.closest_point(lid, jv)
    else:
        d2 = np.array([np.inf])
    # penetration: lid vertices inside the jar, jar vertices inside the lid
    s1 = trimesh.proximity.signed_distance(jar, lv)  # >0 inside
    s2 = trimesh.proximity.signed_distance(lid, jv) if len(jv) else np.array([-np.inf])
    pen = max(s1.max(), s2.max())
    d = min(d1.min(), d2.min())
    return (-pen if pen > 0 else d), pen


def T_screw(z, psi):
    T = np.eye(4)
    c, s = math.cos(psi), math.sin(psi)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[2, 3] = z
    return T


def main():
    params = json.load(open(os.path.join(A, "params.json")))
    g = params["g_axial_mm"]
    out = {}
    for name, p in params["parts"].items():
        jar, lid = load(p["jar_col"]["file"]), load(p["lid_col"]["file"])
        rim, P = p["rim_z"] * 1e3, p["pitch"] * 1e3
        res = dict()
        d, pen = min_dist(jar, lid, T_screw(rim, 0.0))
        res["seated_design"] = dict(min_dist_mm=d, penetration_mm=pen)
        path = []
        for n in np.arange(0.0, 1.51, 0.125):
            d, _ = min_dist(jar, lid, T_screw(rim + g + n * P, 2 * np.pi * n))
            path.append((float(n), d))
        res["right_hand_path_min_dist_mm"] = path
        wrong = []
        for n in (0.25, 0.5, 0.75, 1.0):
            d, _ = min_dist(jar, lid, T_screw(rim + g + n * P, -2 * np.pi * n))
            wrong.append((float(n), d))
        res["wrong_hand_path_min_dist_mm"] = wrong
        out[name] = res
        print(name, "seated design: dist %.4f mm pen %.4f mm" % (res["seated_design"]["min_dist_mm"],
                                                                 res["seated_design"]["penetration_mm"]))
        print("   right-hand path (turns, mm):", " ".join(f"{n:.3g}:{d:.3f}" for n, d in path))
        print("   wrong-hand path (turns, mm):", " ".join(f"{n:.3g}:{d:.3f}" for n, d in wrong))
    # cross pairs: can lid X sit coaxially on jar Y's mouth? report the radial interference / clearance
    names = list(params["parts"])
    cross = {}
    for jn in names:
        pj = params["parts"][jn]
        for ln in names:
            pl = params["parts"][ln]
            r_lid_crest = pl["lid_r_root_mm"] - pl["spec"]["h"]
            r_jar_crest = pj["spec"]["r_root"] + pj["spec"]["h"]
            r_jar_root = pj["spec"]["r_root"]
            if r_lid_crest > r_jar_crest:
                verdict = "lid thread passes over the jar thread (too big: falls over the neck)"
            elif r_lid_crest > r_jar_root:
                verdict = "threads overlap radially (can engage)" if ln == jn else "overlap but pitch/diameter mismatch"
            else:
                verdict = "lid thread crest inside the jar neck (too small: cannot go on)"
            cross[f"{ln}_lid_on_{jn}_jar"] = verdict
    out["cross_pairs"] = cross
    for k, v in cross.items():
        print(f"   {k}: {v}")
    json.dump(out, open(os.path.join(A, "fit_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
