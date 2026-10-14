"""Build one Genesis-loadable MJCF per YAM variant: a wheeled cart (I2RT Flow Base replica + lift column + arm deck)
carrying two arms of that variant with I2RT linear_4310 grippers.

    ~/simquarry/genesis-world/.venv/bin/python ~/projects/yam-arcade/robots/build_models.py

Writes ~/projects/yam-arcade/assets/robots/yam_cart_<variant>.xml (+ meshes/<variant>/*.stl copied from the I2RT
checkout). The arm + gripper composition follows i2rt.robots.utils.combine_arm_and_gripper_xml: the arm's terminal
`gripper` mount body gets pos/quat/axis from gripper/linear_4310.yml (per arm), and the gripper's children are merged
into it under a <frame> carrying the gripper root's own pos/quat.

Cart-local frame of the MJCF: origin on the floor under the cart footprint centre, +y = forward (towards the table),
+z up. add_cart() places it at (0, ROBOT_MOUNT['cart_center_y'], 0) in the station frame; the three virtual joints
base_x / base_y / base_yaw then drive it.
"""
import copy
import math
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import layout  # noqa: E402

I2RT = os.path.join(ROOT, "assets", "i2rt_upstream", "i2rt")
OUT = os.path.join(ROOT, "assets", "robots")
VARIANTS = ("yam", "yam_pro", "yam_ultra", "big_yam")
GRIPPER = "linear_4310"

# Peak torques from i2rt/motor_drivers/utils.py MotorConstants.TORQUE_MAX, motor lists from robots/config/<arm>_v1.yml.
MOTOR_TMAX = {"DM4310": 10.0, "DM4340": 28.0, "DM6248": 120.0}

# Finished look. The I2RT MJCF rgba values are CAD-export debug colours (purple base, orange link1, ...), so the
# arms get the real anodised finish instead: black links, a dark-grey base, silver wrist and fingertips.
PALETTE = {
    "anodised_black": (0.045, 0.045, 0.05),
    "graphite": (0.20, 0.20, 0.22),
    "silver": (0.70, 0.71, 0.73),
    "white_shell": (0.90, 0.90, 0.88),
    "top_plate": (0.16, 0.16, 0.18),
    "rubber": (0.03, 0.03, 0.03),
    "aluminium": (0.78, 0.79, 0.80),
    "estop_red": (0.80, 0.08, 0.06),
    "camera": (0.12, 0.12, 0.13),
}
# per-variant tint so the four stations differ subtly (yam_pro is grey in I2RT's own MJCF)
ARM_FINISH = {
    "yam": dict(base="graphite", link="anodised_black", wrist="silver"),
    "yam_pro": dict(base="graphite", link="graphite", wrist="silver"),
    "yam_ultra": dict(base="anodised_black", link="anodised_black", wrist="silver"),
    "big_yam": dict(base="graphite", link="anodised_black", wrist="silver"),
}

ARMATURE = {"joint1": "0.05", "joint2": "0.05", "joint3": "0.05", "joint4": "0.01", "joint5": "0.01", "joint6": "0.01",
            "joint7": "0.5", "joint8": "0.5"}

# Contact bitmasks (MuJoCo rule: collide if (ct1 & ca2) or (ct2 & ca1)). Everything else in a scene uses
# Genesis' default 0xFFFF masks, so all robot geoms collide with props; left and right arm collide with each other;
# nothing collides with its own arm or with the cart (adjacent convex hulls overlap at rest).
MASK = {"left": (2, 1 | 4), "right": (4, 1 | 2), "cart": (8, 1)}

# Flow Base dimensions: caster layout and wheel radius from i2rt/flow_base/flow_base_controller.py (h_x, h_y = +-0.2,
# r = 0.05); shell sized from the product drawing (i2rt/flow_base/assets/flow_base.png).
CASTER_XY = 0.20
WHEEL_R = 0.05
WHEEL_W = 0.035
SHELL = 0.52  # square footprint, m
SHELL_Z = (0.075, 0.245)
PLATE_T = 0.012
COLUMN = (0.09, 0.07)  # lift column cross-section x, y
DECK_T = 0.04
DECK_DEPTH = 0.12
MAST_TOP = 1.70  # overhead D405: I2RT's station puts it 0.954 m above the arm bases


def rgba(name, a=1.0):
    return "%g %g %g %g" % (*PALETTE[name], a)


def fmt(v):
    return " ".join(f"{x:.10g}" for x in v)


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qrot(q, v):
    qv = np.concatenate([[0.0], v])
    return qmul(qmul(q, qv), q * np.array([1, -1, -1, -1]))[1:]


def floats(s):
    return np.array([float(x) for x in s.split()])


def deepest_body(el):
    while True:
        kids = [c for c in el if c.tag == "body"]
        if not kids:
            return el
        el = kids[0]


def arm_body(variant, side, mesh_names):
    """<body name='<side>_base'> holding the arm + merged gripper, names prefixed with <side>_."""
    arm_xml = os.path.join(I2RT, "robot_models", "arm", variant, "v1", f"{variant}.xml")
    arm = ET.parse(arm_xml).getroot()
    wb = arm.find("worldbody")
    cfg = yaml.safe_load(open(os.path.join(I2RT, "robots", "config", f"{GRIPPER}.yml")))
    mount_cfg = cfg["last_joint_mount"][variant]["v1"]

    if variant == "big_yam":  # its MJCF has the base as a bare worldbody geom; the URDF base inertial goes on a body
        base = ET.Element("body", name="base", pos="0 0 0", quat="1 0 0 0")
        ET.SubElement(base, "inertial", pos="0 0 0.0314825", mass="1.11529",
                      diaginertia="0.00127239 0.00108285 0.00120806")
        for c in list(wb):
            base.append(c)
    else:
        base = wb.find("body")
    mount = deepest_body(base)
    mount.set("pos", mount_cfg["pos"])
    mount.set("quat", mount_cfg["quat"])
    mount.find("joint").set("axis", mount_cfg["axis"])

    grip = ET.parse(os.path.join(I2RT, "robot_models", "gripper", GRIPPER, f"{GRIPPER}.xml")).getroot()
    gbody = grip.find(".//body[@name='gripper']")
    gpos, gquat = floats(gbody.get("pos", "0 0 0")), floats(gbody.get("quat", "1 0 0 0"))
    frame = ET.Element("frame", pos=fmt(gpos), quat=fmt(gquat))
    for c in gbody:
        if c.tag != "inertial":
            frame.append(copy.deepcopy(c))
    mount.append(frame)
    gi = gbody.find("inertial")
    ph = mount.find("inertial")
    mount.remove(ph)
    gi = copy.deepcopy(gi)
    gi.set("pos", fmt(gpos + qrot(gquat, floats(gi.get("pos")))))
    gi.set("quat", fmt(qmul(gquat, floats(gi.get("quat")))))
    mount.insert(0, gi)

    finish = ARM_FINISH[variant]
    ct, ca = MASK[side]
    for el in base.iter():
        if el.tag in ("body", "joint", "site") and el.get("name"):
            el.set("name", f"{side}_{el.get('name')}")
        if el.tag == "joint":
            el.attrib.pop("actuatorfrcrange", None)
            # reflected rotor inertia (geared DM motors); also keeps the light wrist links stable at dt = 10 ms
            el.set("armature", ARMATURE[el.get("name").split("_")[-1]])
            # fingers: geared rack drive -> heavy viscous damping; it is also what damps the finger against libuipc's
            # contact (the capped PD actuator gives no damping once it saturates at the force cap)
            el.set("damping", "0.1" if el.get("type") != "slide" else "50")
        if el.tag == "site":
            el.set("group", "5")
    # colours + contact masks per geom; mesh geoms of the arm links are both visual and (convex-hull) collision,
    # the gripper housing and fingertip meshes are visual only (box collision added below, see add_gripper_collision)
    for body in base.iter("body"):
        bname = body.get("name")[len(side) + 1:]
        for g in [c for c in body if c.tag == "geom"] + [c for f in body if f.tag == "frame" for c in f if c.tag == "geom"]:
            mesh = g.get("mesh")
            mesh_names.add(mesh)
            if bname == "base":
                col = finish["base"]
            elif bname in ("link5",):
                col = finish["wrist"]
            elif bname.startswith("tip"):
                col = "silver"
            elif bname == "gripper":
                col = "anodised_black"
            else:
                col = finish["link"]
            g.set("rgba", rgba(col))
            g.set("name", f"{side}_{bname}_visual")
            if bname in ("gripper",) or bname.startswith("tip"):
                g.set("contype", "0")
                g.set("conaffinity", "0")
                g.set("group", "2")
            else:
                g.set("contype", str(ct))
                g.set("conaffinity", str(ca))
                g.set("group", "0")
    base.set("name", f"{side}_base")
    return base


# Gripper collision boxes, in the frame of the tip body / mount body, measured from the meshes by
# robots/measure_gripper.py and pasted here (half sizes). Kept separate from each other by >= 3 mm at the
# fully-closed stop so libuipc never sees overlapping link geometry.
PAD = None  # filled from measure_gripper.json if present
PALM = None


def add_gripper_collision(base, side):
    meas = os.path.join(OUT, "gripper_collision.json")
    if not os.path.exists(meas):
        return
    import json
    m = json.load(open(meas))
    ct, ca = MASK[side]
    for body in base.iter("body"):
        n = body.get("name")
        if n in (f"{side}_tip_left", f"{side}_tip_right"):
            key = n[len(side) + 1:]
            for i, b in enumerate(m[key]):
                ET.SubElement(body, "geom", name=f"{n}_pad{i}", type="box", size=fmt(b["half"]), pos=fmt(b["pos"]),
                              quat=fmt(b["quat"]),
                              contype=str(ct), conaffinity=str(ca), group="3", rgba="1 0 0 0.3",
                              friction="1.0 0.005 0.0001")
        if n == f"{side}_gripper":
            for i, b in enumerate(m["palm"]):
                ET.SubElement(body, "geom", name=f"{n}_palm{i}", type="box", size=fmt(b["half"]), pos=fmt(b["pos"]),
                              quat=fmt(b.get("quat", (1, 0, 0, 0))), contype=str(ct), conaffinity=str(ca), group="3",
                              rgba="0 0 1 0.3")


def cart_geoms(cart, mount):
    """Procedural I2RT Flow Base + lift column + arm deck + overhead camera mast, all on one link."""
    ct, ca = MASK["cart"]

    def box(name, center, half, col, collide=True, quat=None):
        g = ET.SubElement(cart, "geom", name=name, type="box", pos=fmt(center), size=fmt(half), rgba=rgba(col))
        if quat is not None:
            g.set("quat", fmt(quat))
        if collide:
            g.set("contype", str(ct))
            g.set("conaffinity", str(ca))
            g.set("group", "0")
        else:
            g.set("contype", "0")
            g.set("conaffinity", "0")
            g.set("group", "2")
        return g

    def cyl(name, center, r, half_len, col, quat=(1, 0, 0, 0)):
        return ET.SubElement(cart, "geom", name=name, type="cylinder", pos=fmt(center), size=f"{r} {half_len}",
                             quat=fmt(quat), rgba=rgba(col), contype="0", conaffinity="0", group="2")

    # casters: wheel axle along x (rolling forward), fork above, swivel plate under the shell
    qx = (math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0)  # cylinder axis z -> x
    for i, (sx, sy) in enumerate(((-1, 1), (1, 1), (1, -1), (-1, -1))):
        x, y = sx * CASTER_XY, sy * CASTER_XY
        cyl(f"wheel{i}", (x, y + 0.02, WHEEL_R), WHEEL_R, WHEEL_W / 2, "rubber", qx)
        cyl(f"hub{i}", (x, y + 0.02, WHEEL_R), 0.022, WHEEL_W / 2 + 0.002, "aluminium", qx)
        for side_x in (-1, 1):
            box(f"fork{i}_{side_x}", (x + side_x * (WHEEL_W / 2 + 0.006), y + 0.01, WHEEL_R + 0.012),
                (0.004, 0.022, 0.045), "graphite", collide=False)
        box(f"swivel{i}", (x, y, SHELL_Z[0] - 0.008), (0.035, 0.035, 0.008), "graphite", collide=False)
    # two-tier white shell + dark top plate + bumper seam + e-stop + charge port
    h = SHELL / 2
    zmid = 0.5 * (SHELL_Z[0] + SHELL_Z[1])
    box("shell_low", (0, 0, 0.5 * (SHELL_Z[0] + zmid)), (h - 0.01, h - 0.01, 0.5 * (zmid - SHELL_Z[0])), "white_shell")
    box("shell_high", (0, 0, 0.5 * (zmid + SHELL_Z[1])), (h, h, 0.5 * (SHELL_Z[1] - zmid) - 0.002), "white_shell")
    box("top_plate", (0, 0, SHELL_Z[1] + PLATE_T / 2), (h, h, PLATE_T / 2), "top_plate")
    ztop = SHELL_Z[1] + PLATE_T
    ET.SubElement(cart, "geom", name="estop", type="cylinder", pos=fmt((0.17, -0.20, ztop + 0.01)),
                  size="0.016 0.01", rgba=rgba("estop_red"), contype="0", conaffinity="0", group="2")
    for i, sx in enumerate((-1, 1)):  # the two rails on the top plate
        box(f"rail{i}", (sx * 0.12, -0.08, ztop + 0.008), (0.012, 0.17, 0.008), "aluminium", collide=False)
    # lift column (I2RT linear rail) at the footprint centre, gusset plates either side
    col_top = mount["deck_z"] - DECK_T
    cz = 0.5 * (ztop + col_top)
    box("column", (0, 0, cz), (COLUMN[0] / 2, COLUMN[1] / 2, 0.5 * (col_top - ztop)), "white_shell")
    box("column_plate", (0, 0, ztop + 0.004), (0.10, 0.09, 0.004), "aluminium", collide=False)
    for i, sx in enumerate((-1, 1)):
        ang = math.atan2(0.16, 0.10)
        q = (math.cos((math.pi / 2 - ang) / 2 * -sx), 0, math.sin((math.pi / 2 - ang) / 2 * -sx), 0)
        box(f"gusset{i}", (sx * 0.075, 0, ztop + 0.075), (0.006, 0.03, 0.085), "aluminium", collide=False, quat=q)
    # carriage from the column face to the deck, then the deck carrying both arm bases
    dy = mount["deck_y"]
    deck_back = dy + 0.035 - DECK_DEPTH
    box("carriage", (0, 0.5 * (COLUMN[1] / 2 + deck_back), mount["deck_z"] - DECK_T - 0.04),
        (0.07, 0.5 * (deck_back - COLUMN[1] / 2) + 0.01, 0.06), "aluminium")
    half_x = mount["spacing"] / 2 + 0.11
    # deck front flush with the arm base plates' front edge (3.5 cm ahead of the base centres); it must stay behind
    # the table's near edge because the deck top is level with the table top
    front = dy + 0.035
    assert front + layout.ROBOT_MOUNT["cart_center_y"] <= layout.NEAR_EDGE_Y - 0.009, "deck would hit the table"
    box("deck", (0, front - DECK_DEPTH / 2, mount["deck_z"] - DECK_T / 2), (half_x, DECK_DEPTH / 2, DECK_T / 2),
        "graphite")
    # overhead camera mast: from the column top up to the D405 (pose from I2RT's YAM station, mesh d405_top.stl)
    mast_y = -0.035
    box("mast", (0, mast_y - 0.02, 0.5 * (col_top + MAST_TOP)), (0.015, 0.015, 0.5 * (MAST_TOP - col_top)), "aluminium")
    # I2RT station: left_base -> top_camera xyz (-0.166, -0.305, 0.954), quat (0.183, -0.683, 0.683, -0.183) in the
    # arm frame (arm +x forward). Our arm frame is yawed +90 deg into the cart frame (arm +x = cart +y).
    qyaw = np.array([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])
    qcam = qmul(qyaw, np.array([0.18301153618854285, -0.6830116697534223, 0.683014178599981, -0.1830122084286862]))
    cam_pos = (0.0, mast_y, MAST_TOP + 0.03)
    ET.SubElement(cart, "geom", name="top_camera", type="mesh", mesh="d405_top", pos=fmt(cam_pos), quat=fmt(qcam),
                  rgba=rgba("camera"), contype="0", conaffinity="0", group="2")
    ET.SubElement(cart, "site", name="top_camera", pos=fmt(cam_pos), quat=fmt(qcam), group="5")


def split_visual_collision(root):
    """Genesis draws a geom that is both visual and collision with the default grey, dropping its rgba; so each such
    geom becomes a visual-only geom (group 2) plus a collision-only copy (group 3, same shape and contact masks)."""
    for parent in root.iter():
        for i, g in reversed(list(enumerate(list(parent)))):
            if g.tag != "geom" or g.get("contype", "1") == "0" or g.get("group") == "3":
                continue
            col = copy.deepcopy(g)
            col.set("name", g.get("name") + "_col")
            col.set("group", "3")
            col.set("rgba", "0.5 0.5 0.5 0")
            g.set("contype", "0")
            g.set("conaffinity", "0")
            g.set("group", "2")
            parent.insert(i + 1, col)


def build(variant):
    m = layout.ROBOT_MOUNT
    deck_y = m["arm_base_y"] - m["cart_center_y"]  # arm bases, cart frame
    mount = dict(deck_y=deck_y, deck_z=m["arm_base_z"], spacing=m["arm_spacing"])
    mesh_dir = os.path.join(OUT, "meshes", variant)
    os.makedirs(mesh_dir, exist_ok=True)

    root = ET.Element("mujoco", model=f"yam_cart_{variant}")
    ET.SubElement(root, "compiler", angle="radian", meshdir=mesh_dir, autolimits="true")
    ET.SubElement(root, "option", timestep="0.01")
    asset = ET.SubElement(root, "asset")
    wb = ET.SubElement(root, "worldbody")

    # virtual planar base: slide x, slide y, hinge yaw (small masses on the two carrier bodies)
    bx = ET.SubElement(wb, "body", name="cart_x", pos="0 0 0")
    ET.SubElement(bx, "inertial", pos="0 0 0", mass="0.5", diaginertia="0.01 0.01 0.01")
    ET.SubElement(bx, "joint", name="base_x", type="slide", axis="1 0 0", range="-5 5", armature="10")
    by = ET.SubElement(bx, "body", name="cart_y", pos="0 0 0")
    ET.SubElement(by, "inertial", pos="0 0 0", mass="0.5", diaginertia="0.01 0.01 0.01")
    ET.SubElement(by, "joint", name="base_y", type="slide", axis="0 1 0", range="-5 5", armature="10")
    cart = ET.SubElement(by, "body", name="cart", pos="0 0 0")
    # Flow Base ~ 35 kg with battery, column + deck ~ 10 kg
    ET.SubElement(cart, "inertial", pos="0 0 0.25", mass="45", diaginertia="2.0 2.0 1.6")
    ET.SubElement(cart, "joint", name="base_yaw", type="hinge", axis="0 0 1", range="-6.3 6.3", armature="5")
    cart_geoms(cart, mount)

    mesh_names = set()
    qyaw = (math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4))  # arm +x (forward) -> cart +y
    for side, sx in (("left", -1), ("right", 1)):  # facing +y, the robot's left is -x
        b = arm_body(variant, side, mesh_names)
        b.set("pos", fmt((sx * m["arm_spacing"] / 2, deck_y, m["arm_base_z"])))
        b.set("quat", fmt(qyaw))
        add_gripper_collision(b, side)
        cart.append(b)

    arm_assets = os.path.join(I2RT, "robot_models", "arm", variant, "v1", "assets")
    grip_assets = os.path.join(I2RT, "robot_models", "gripper", GRIPPER, "assets")
    for name in sorted(mesh_names):
        src = os.path.join(grip_assets if name in ("gripper", "tip_left", "tip_right") else arm_assets, f"{name}.stl")
        shutil.copy2(src, os.path.join(mesh_dir, f"{name}.stl"))
        ET.SubElement(asset, "mesh", name=name, file=f"{name}.stl")
    shutil.copy2(os.path.join(I2RT, "robot_models", "station", "assets", "d405_top.stl"),
                 os.path.join(mesh_dir, "d405_top.stl"))
    ET.SubElement(asset, "mesh", name="d405_top", file="d405_top.stl")

    split_visual_collision(root)
    # the linear_4310 drives both fingers from one motor through a rack: joint8 = joint7 (I2RT gripper MJCF equality)
    eq = ET.SubElement(root, "equality")
    for side in ("left", "right"):
        ET.SubElement(eq, "joint", joint1=f"{side}_joint7", joint2=f"{side}_joint8", polycoef="0 1 0 0 0")
    ET.indent(root)
    path = os.path.join(OUT, f"yam_cart_{variant}.xml")
    header = (f"<!-- generated by robots/build_models.py from the I2RT checkout ({variant} + {GRIPPER}); "
              "do not edit by hand -->\n")
    with open(path, "w") as f:
        f.write(header + ET.tostring(root, encoding="unicode"))
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for v in VARIANTS:
        print(build(v))
