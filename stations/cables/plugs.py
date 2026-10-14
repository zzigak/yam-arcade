"""Rigid connectors at one end of each cable (the robot picks cables by them, as a person does).

Plug frame: origin on the cable axis at the boot face (where the cable leaves the plug), +x toward the contact tip, z
up. The cable runs straight through the plug from the tip vertex to the boot vertex in N_INNER segments that are 100x
stiffer (moulded into the plug); libuipc SoftVertexStitch springs (zero rest length) join cable vertex 0 to the hull's
tip vertex and cable vertex N_INNER to the boot vertex, so the plug holds both position and direction of the cable end
(the same construction as ~/projects/franka-deformables/genesis_cable.py). Contact between a cable and its own plug is
off (the inner section is inside the hull); everything else is normal IPC contact.

Collision = convex hull of the visual parts with the two attach vertices inserted on the axis; visual = the parts
(body, boot, metal shell / pin). Sizes and masses are those of common connectors.
"""
import json
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets", "plugs")
STITCH_KAPPA = 1.0e6  # N/m, as genesis_cable.py (calibrated there: anchor drift < 1 um)
INNER_STIFFEN = 100.0  # bending/torsion of the cable section moulded into the plug, x the cable's

# parts: list of (kind, x0, x1, size_y, size_z, z_offset); kind 'box' or 'oct' (octagonal prism, size_y = diameter).
# 'shell': number of trailing parts that are the thin metal shell / pin. They ARE part of the collision hull (an earlier
# version left them out and a tilted plug's shell dipped 4 mm into the table); the stitched cable section runs to the
# shell's tip, bulging slightly out of the hull where the shell is thinner than the cable (it still collides with
# everything but its own plug, and it is not drawn). The grasp point is the body centre ('body_x').
PLUGS = {
    "iec_c13": dict(what="IEC C13 mains connector", mass=0.035, color=(0.10, 0.10, 0.11),
                    parts=[("box", 0.0, 0.012, 0.016, 0.016, 0.0), ("box", 0.012, 0.050, 0.032, 0.022, 0.0)]),
    "hdmi": dict(shell=1, what="HDMI type A plug", mass=0.012, color=(0.10, 0.10, 0.11),
                 parts=[("box", 0.0, 0.010, 0.012, 0.010, 0.0), ("box", 0.010, 0.036, 0.021, 0.011, 0.0),
                        ("box", 0.036, 0.047, 0.014, 0.0045, 0.0)]),
    "euro": dict(what="2-pin Euro plug (pins omitted)", mass=0.025, color=(0.92, 0.92, 0.90),
                 parts=[("box", 0.0, 0.010, 0.014, 0.014, 0.0), ("box", 0.010, 0.040, 0.024, 0.018, 0.0)]),
    "rj45": dict(what="RJ45 plug with boot", mass=0.005, color=(0.80, 0.82, 0.84),
                 parts=[("box", 0.0, 0.012, 0.010, 0.010, 0.0), ("box", 0.012, 0.035, 0.014, 0.011, 0.0)]),
    "barrel": dict(shell=1, what="5.5 x 2.1 mm DC barrel plug", mass=0.004, color=(0.10, 0.10, 0.11),
                   parts=[("oct", 0.0, 0.024, 0.010, 0.010, 0.0), ("oct", 0.024, 0.033, 0.0055, 0.0055, 0.0)]),
    "usbc": dict(shell=1, what="USB-C plug", mass=0.003, color=(0.12, 0.12, 0.13),
                 parts=[("box", 0.0, 0.020, 0.012, 0.0065, 0.0), ("box", 0.020, 0.0285, 0.0084, 0.0026, 0.0)]),
    "micro_usb": dict(shell=1, what="micro-USB plug", mass=0.0025, color=(0.12, 0.12, 0.13),
                      parts=[("box", 0.0, 0.018, 0.011, 0.006, 0.0), ("box", 0.018, 0.024, 0.0070, 0.0020, 0.0)]),
    "jack35": dict(shell=1, what="3.5 mm audio jack", mass=0.003, color=(0.12, 0.12, 0.13),
                   parts=[("oct", 0.0, 0.026, 0.008, 0.008, 0.0), ("oct", 0.026, 0.040, 0.0035, 0.0035, 0.0)]),
}


def _part_mesh(kind, x0, x1, sy, sz, zo):
    if kind == "box":
        m = trimesh.creation.box(extents=(x1 - x0, sy, sz))
    else:
        m = trimesh.creation.cylinder(radius=0.5 * sy / np.cos(np.pi / 8), height=x1 - x0, sections=8)
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 8, [1, 0, 0]))
    m.apply_translation((0.5 * (x0 + x1), 0.0, zo))
    return m


def _insert_point(mesh, p, tri):
    v = np.vstack([mesh.vertices, p])
    k = len(v) - 1
    a, b, c = mesh.faces[tri]
    faces = np.delete(mesh.faces, tri, axis=0)
    faces = np.vstack([faces, [[a, b, k], [b, c, k], [c, a, k]]])
    return trimesh.Trimesh(v, faces, process=False), k


def make_plug(kind):
    """Writes assets/plugs/<kind>{_visual.obj,_hull.obj,.urdf,.json}; returns dict(tip, boot, length, volume, files)."""
    os.makedirs(ASSETS, exist_ok=True)
    spec = PLUGS[kind]
    parts = [_part_mesh(*p) for p in spec["parts"]]
    vis = trimesh.util.concatenate(parts)
    body = trimesh.util.concatenate(parts[:len(parts) - spec.get("shell", 0)])
    hull = trimesh.convex.convex_hull(vis.vertices)
    pts = {}
    mesh = hull
    for name, d, start in (("tip", 1.0, -1.0), ("boot", -1.0, 1.0)):
        origin = np.array([[0.5 * (vis.bounds[0, 0] + vis.bounds[1, 0]), 0.0, 0.0]])
        loc, _, tri = mesh.ray.intersects_location(origin, np.array([[d, 0.0, 0.0]]))
        i = int(np.argmax(d * loc[:, 0]))
        mesh, _ = _insert_point(mesh, loc[i], int(tri[i]))
        pts[name] = loc[i]
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0, kind
    f_vis = os.path.join(ASSETS, f"{kind}_visual.obj")
    f_hull = os.path.join(ASSETS, f"{kind}_hull.obj")
    f_urdf = os.path.join(ASSETS, f"{kind}.urdf")
    # flat-shaded visual (unmerged vertices)
    flat = trimesh.Trimesh(vis.vertices[vis.faces].reshape(-1, 3), np.arange(3 * len(vis.faces)).reshape(-1, 3),
                           process=False)
    flat.export(f_vis)
    mesh.export(f_hull)
    open(f_urdf, "w").write(f"""<?xml version="1.0"?>
<robot name="{kind}">
  <link name="{kind}">
    <visual><geometry><mesh filename="{os.path.basename(f_vis)}"/></geometry></visual>
    <collision><geometry><mesh filename="{os.path.basename(f_hull)}"/></geometry></collision>
  </link>
</robot>
""")
    info = dict(tip=pts["tip"].tolist(), boot=pts["boot"].tolist(), length=float(pts["tip"][0] - pts["boot"][0]),
                volume=float(mesh.volume), hull=f_hull, visual=f_vis, urdf=f_urdf,
                width=float(body.extents[1]), height=float(body.extents[2]), mass=spec["mass"],
                body_x=[float(body.bounds[0, 0]), float(body.bounds[1, 0])])
    json.dump(info, open(os.path.join(ASSETS, f"{kind}.json"), "w"), indent=1)
    return info


def n_inner(length, radius, d_hat):
    """Segments inside the plug: as many as the rod rule (spacing - 2 r > d_hat, with 0.5 mm margin) allows, >= 2."""
    return max(2, int(np.floor(length / (2 * radius + d_hat + 0.0005))))
