"""Rigid-only checks for one or all cart variants (gs.cpu):
loads, joint list, home-pose hold (sag), table clearance at home, gripper travel, top-down IK reach over WORK_ZONE.

    ~/simquarry/genesis-world/.venv/bin/python ~/projects/yam-arcade/robots/test_rigid.py [variant ...]

Writes robots/checks/rigid_<variant>.json.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import genesis as gs  # noqa: E402
import layout  # noqa: E402
from yam_cart import GRASP_TILTS, MAX_WIDTH, SIDES, VARIANTS, add_cart  # noqa: E402
from genesis.utils.misc import tensor_to_array as A  # noqa: E402


def add_table(scene, legs=True):
    T = layout.TABLE
    top = scene.add_entity(gs.morphs.Box(pos=(0, 0, T["height"] - T["top_thickness"] / 2),
                                         size=(T["size_x"], T["size_y"], T["top_thickness"]), fixed=True),
                           surface=gs.surfaces.Default(color=(0.75, 0.62, 0.45)))
    ents = [top]
    if legs:
        h = T["height"] - T["top_thickness"]
        for sx in (-1, 1):
            for sy in (-1, 1):
                x = sx * (T["size_x"] / 2 - T["leg_inset"])
                y = sy * (T["size_y"] / 2 - T["leg_inset"])
                ents.append(scene.add_entity(gs.morphs.Box(pos=(x, y, h / 2), size=(T["leg_size"],) * 2 + (h,), fixed=True),
                                             surface=gs.surfaces.Default(color=(0.6, 0.6, 0.6))))
    return ents


def run(variant, grid_n=(11, 7)):
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False,
                     rigid_options=gs.options.RigidOptions(enable_self_collision=True))
    scene.add_entity(gs.morphs.Plane())
    table = add_table(scene)
    cart = add_cart(scene, gs, variant)
    scene.build()
    t0 = time.time()
    cart.after_build()
    e = cart.entity
    res = dict(variant=variant, n_dofs=e.n_dofs, n_links=e.n_links)
    res["joints"] = {j.name: (A(j.dofs_limit).round(4).tolist() if j.n_dofs else None) for j in e.joints}
    res["home_qpos"] = cart.home_qpos.round(4).tolist()
    tcp0 = {s: cart.tcp_pose(s)[0] for s in SIDES}
    res["home_tcp"] = {s: tcp0[s].round(4).tolist() for s in SIDES}

    # 1) hold home for 3 s with PD + gravity compensation
    contacts_table = 0
    for i in range(300):
        cart.update()
        scene.step()
        for tb in table:
            c = e.get_contacts(with_entity=tb)
            contacts_table += len(A(c["geom_a"]).reshape(-1))
    drift = {s: float(np.linalg.norm(cart.tcp_pose(s)[0] - tcp0[s])) for s in SIDES}
    qerr = float(np.abs(cart.qpos() - cart.q_target)[np.r_[cart.arm_dofs["left"], cart.arm_dofs["right"]]].max())
    res["hold"] = dict(tcp_drift_m=drift, max_arm_joint_err_rad=qerr, table_contacts=contacts_table)
    # geometric clearance: lowest robot link AABB vs table top, and nearest robot AABB y to the table near edge
    # among links below the table top
    aabb = {l.name: A(l.get_AABB()) for l in e.links if len(l.geoms)}
    res["hold"]["robot_zmin"] = float(min(v[0, 2] for v in aabb.values()))

    # 2) gripper travel: open, then close on nothing
    for s in SIDES:
        cart.open(s)
    for i in range(150):
        cart.update(); scene.step()
    w_open = {s: cart.gripper_width(s) for s in SIDES}
    for s in SIDES:
        cart.grasp(s, force=10.0)
    for i in range(200):
        cart.update(); scene.step()
    w_closed = {s: cart.gripper_width(s) for s in SIDES}
    tips = {s: [A(e.get_link(f"{s}_{n}").get_pos()) for n in ("tip_left", "tip_right")] for s in SIDES}
    res["gripper"] = dict(width_open=w_open, width_closed=w_closed, max_width_nominal=MAX_WIDTH,
                          finger_q_closed={s: cart.finger_q(s).round(4).tolist() for s in SIDES})
    for s in SIDES:
        cart.open(s)
    for i in range(150):
        cart.update(); scene.step()

    # 3) top-down IK reach over WORK_ZONE at table + 2 cm and + 10 cm, fingers closing along x and along y
    WZ = layout.WORK_ZONE
    xs = np.linspace(*WZ["x"], grid_n[0])
    ys = np.linspace(*WZ["y"], grid_n[1])
    home = cart.home_qpos.copy()
    res["reach"] = dict(grid=dict(x=xs.round(3).tolist(), y=ys.round(3).tolist(), heights=[0.02, 0.10]))
    for mode, tilts in (("topdown", (0.0,)), ("tilted", GRASP_TILTS)):
        reach = {}
        tilt_used = np.full((2, len(xs), len(ys)), np.nan)
        for s in SIDES:
            ok = np.zeros((2, len(xs), len(ys)), bool)
            for hi, dz in enumerate((0.02, 0.10)):
                for ix, x in enumerate(xs):
                    for iy, y in enumerate(ys):
                        for yaw in (0.0, np.pi / 2):
                            q, tilt = cart.ik_grasp(s, (x, y, layout.TABLE_TOP_Z + dz), yaw=yaw, tilts=tilts,
                                                    init_qpos=home)
                            if q is not None:
                                ok[hi, ix, iy] = True
                                tilt_used[hi, ix, iy] = np.fmin(tilt_used[hi, ix, iy], tilt)
                                break
            reach[s] = ok
        union = reach["left"] | reach["right"]
        sym = lambda a, h: ["".join("#" if a[h, ix, iy] else "." for ix in range(len(xs))) for iy in range(len(ys))][::-1]
        res["reach"][mode] = dict(
            frac_left=float(reach["left"].mean()), frac_right=float(reach["right"].mean()),
            frac_union=float(union.mean()), frac_union_by_height=[float(union[0].mean()), float(union[1].mean())],
            map_union_2cm=sym(union, 0), map_union_10cm=sym(union, 1), map_left_2cm=sym(reach["left"], 0))
        if mode == "tilted":
            res["reach"][mode]["min_tilt_2cm"] = [" ".join("%.2f" % tilt_used[0, ix, iy] for ix in range(len(xs)))
                                                  for iy in range(len(ys))][::-1]
    res["wall_s"] = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    gs.init(backend=gs.cpu, logging_level="warning")
    os.makedirs(os.path.join(HERE, "checks"), exist_ok=True)
    for v in sys.argv[1:] or VARIANTS:
        r = run(v)
        json.dump(r, open(os.path.join(HERE, "checks", f"rigid_{v}.json"), "w"), indent=1)
        print(json.dumps({k: r[k] for k in ("variant", "home_tcp", "hold", "gripper", "wall_s")}, indent=1))
        for mode in ("topdown", "tilted"):
            rr = r["reach"][mode]
            print(f"[{mode}] reach left {rr['frac_left']:.2f} right {rr['frac_right']:.2f} union {rr['frac_union']:.2f} "
                  f"by height {rr['frac_union_by_height']}")
            print("union @ +2 cm (rows: far y at top; cols: x from -0.52 to 0.52)")
            print("\n".join(rr["map_union_2cm"]))
            print("left arm only @ +2 cm")
            print("\n".join(rr["map_left_2cm"]))
