"""Shared layout for the YAM arcade: four tables, each a station with its own task, each with a wheeled bimanual
YAM cart parked at its near edge. Every station script and the Blender compositor import this file.

Frames
------
Station-local frame (what every station sim and every export uses):
    origin on the floor directly under the centre of the table top, +z up, metres.
    The robot cart stands on the -y side of the table and faces +y.
    Table top surface: z = TABLE["height"]; top spans x in [-sx/2, sx/2], y in [-sy/2, sy/2].
World frame (only the compositor uses it): station-local frame placed by STATIONS[name]["world"] = (x, y, yaw_deg).
"""
import math

TABLE = dict(size_x=1.20, size_y=0.70, height=0.75, top_thickness=0.03, leg_inset=0.06, leg_size=0.05)
TABLE_TOP_Z = TABLE["height"]
NEAR_EDGE_Y = -0.5 * TABLE["size_y"]  # the edge the robot works from
FAR_EDGE_Y = 0.5 * TABLE["size_y"]

# Bimanual YAM mount on the cart, in the station-local frame. The robot agent owns these numbers
# (robots/yam_cart.py) and may revise them; stations must read them from here, never hard-code.
# Revised 2026-10-13 by the robot agent (robots/README.md, "Mount and reach"): arm bases moved 3.5 cm forward so the
# 7 cm-deep YAM base plates end 1 cm behind the table edge; YAM / Pro / Ultra then reach the far half of WORK_ZONE
# only with a forward-tilted grasp (cart.ik_grasp tries tilts automatically); Big YAM reaches all of it top-down.
ROBOT_MOUNT = dict(
    arm_spacing=0.61,          # left/right arm bases 0.61 m apart along x (same as I2RT's YAM station)
    arm_base_y=-0.395,         # arm base centres 4.5 cm behind the near table edge (base plate front at -0.36)
    arm_base_z=TABLE_TOP_Z,    # arm base flange level with the table top
    arm_yaw_deg=90.0,          # arm +x (its forward) points along station +y, i.e. over the table
    cart_center_y=-0.62,       # wheeled base footprint centre (0.52 m square I2RT Flow Base; front face at -0.36)
)

# Usable area for task objects (comfortably inside both arms' reach), station-local x/y bounds.
WORK_ZONE = dict(x=(-0.52, 0.52), y=(-0.30, 0.22))

# Which YAM variant works at which table. Grippers are I2RT's gripper models.
STATIONS = {
    "mahjong": dict(arm="yam", gripper="linear_4310", world=(-1.75, 1.55, 0.0),
                    title="Mahjong", task="put the scattered tiles into the rack in order"),
    "jars": dict(arm="yam_pro", gripper="linear_4310", world=(1.75, 1.55, 0.0),
                 title="Jars", task="screw each lid onto the glass jar it fits"),
    "cables": dict(arm="yam_ultra", gripper="linear_4310", world=(-1.75, -1.55, 0.0),
                   title="Cables", task="untangle the cables and sort them into the two boxes by colour"),
    "water": dict(arm="big_yam", gripper="linear_4310", world=(1.75, -1.55, 0.0),
                  title="Water", task="pour water from the pitcher into the three cups"),
}

# Studio look: white/light-grey seamless room.
BACKGROUND_RGB = (0.92, 0.92, 0.93)
FLOOR_RGB = (0.80, 0.80, 0.81)
RENDER_FPS = 30


def station_to_world(name, p):
    """Station-local point -> world point."""
    x, y, yaw = STATIONS[name]["world"]
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return (x + c * p[0] - s * p[1], y + s * p[0] + c * p[1], p[2])
