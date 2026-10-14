"""Rasterizer preview of each cart beside its table: robots/preview_<variant>.png (three views side by side).
Left arm at the home (ready) pose, right arm reaching a top-down grasp pose on the table with fingers half open.

    ~/simquarry/genesis-world/.venv/bin/python ~/projects/yam-arcade/robots/preview.py [variant ...]
"""
import os
import sys

import imageio.v2 as imageio
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import genesis as gs  # noqa: E402
import layout  # noqa: E402
from test_rigid import add_table  # noqa: E402
from yam_cart import VARIANTS, add_cart  # noqa: E402
from genesis.utils.misc import tensor_to_array as A  # noqa: E402

VIEWS = [  # pos, lookat, fov
    ((1.9, -2.3, 1.75), (0.0, -0.35, 0.75), 38),
    ((-2.4, -0.45, 1.25), (0.0, -0.45, 0.7), 40),
    ((0.0, 0.9, 2.1), (0.0, -0.35, 0.8), 50),
]


def run(variant, res=(720, 600)):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        vis_options=gs.options.VisOptions(background_color=layout.BACKGROUND_RGB, ambient_light=(0.3, 0.3, 0.3),
                                          show_world_frame=False, lights=[dict(type="directional", dir=(-0.4, 0.5, -1.0),
                                                                               color=(1.0, 1.0, 1.0), intensity=2.5)]),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=layout.FLOOR_RGB))
    add_table(scene)
    cart = add_cart(scene, gs, variant)
    cams = [scene.add_camera(res=res, pos=p, lookat=l, fov=f, GUI=False) for p, l, f in VIEWS]
    scene.build()
    cart.after_build()
    q = cart.home_qpos.copy()
    target = (0.25, -0.05, layout.TABLE_TOP_Z + 0.03)
    q[cart.arm_dofs["right"]] = cart.ik("right", target, yaw=0.0, init_qpos=q)
    q[cart.finger_dofs["right"]] = 0.02
    cart.set_qpos(q)
    scene.step()
    tcp = cart.tcp_pose("right")[0]
    print(variant, "right TCP", tcp.round(4), "target", target)
    imgs = [np.asarray(A(c.render(rgb=True)[0]), dtype=np.uint8) for c in cams]
    out = os.path.join(HERE, f"preview_{variant}.png")
    imageio.imwrite(out, np.concatenate(imgs, axis=1))
    print("wrote", out)


if __name__ == "__main__":
    gs.init(backend=gs.cpu, logging_level="warning")
    for v in sys.argv[1:] or VARIANTS:
        run(v)
