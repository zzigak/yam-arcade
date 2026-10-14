"""Mahjong station: geometry and task constants shared by make_assets.py (asset venv) and station.py (Genesis venv).

Pure python (math only) so both interpreters can import it.

Frames
------
Tile link frame (= its mesh frame = its collision-box frame, origin at the box centre = centre of mass):
    +x across the width (26 mm), +y up the face towards the top of the glyph (35 mm), +z out of the engraved face
    (21 mm thick). A tile lying face-up on the table with its glyph upright for the robot has identity rotation.
Rack link frame: origin on the table under the middle of the rack's plinth, +x along the rack (the robot's left to
    right), +y away from the robot, +z up. Tiles stand in it facing -y (the robot), leaning back by RACK_ALPHA.
"""
import math

# ----------------------------------------------------------------------------------------------------------- tiles
TILE_W, TILE_H, TILE_T = 0.026, 0.035, 0.021   # riichi tile: width x height x thickness (m)
TILE_EDGE_R = 0.002                             # edge/corner radius of the visual mesh (collision is the full box)
TILE_FACE_LAYER = 0.013                         # ivory front layer; the remaining 8 mm is the coloured back layer
TILE_DENSITY = 1450.0                           # melamine / urea resin, kg/m^3
TILE_MASS = TILE_DENSITY * TILE_W * TILE_H * TILE_T  # 27.7 g (box volume; edge rounding removes < 1 %)

# Suits m (characters), p (dots), s (bamboo), z (honours: 1-4 = East South West North winds, 5-7 = White Green Red
# dragons). Standard sort order is m < p < s < z, then by number.
SUIT_ORDER = "mpsz"
HONOR_GLYPHS = {1: "東", 2: "南", 3: "西", 4: "北", 5: "白", 6: "發", 7: "中"}
ALL_KINDS = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + [f"{n}z" for n in range(1, 8)]

# The 14 tiles on the table: a complete riichi hand (four sets + a pair):
#   1-2-3 characters, 4-5-6 dots, 7-8-9 bamboo, East x3 (pung), Red dragon x2 (pair).
HAND = ["1m", "2m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "1z", "1z", "7z", "7z"]


def sort_key(kind):
    return SUIT_ORDER.index(kind[1]), int(kind[0])


HAND_SORTED = sorted(HAND, key=sort_key)
assert HAND_SORTED == HAND


def kind_name(kind):
    n, s = int(kind[0]), kind[1]
    if s == "z":
        return {1: "East", 2: "South", 3: "West", 4: "North", 5: "White dragon", 6: "Green dragon",
                7: "Red dragon"}[n]
    return f"{n} {dict(m='characters', p='dots', s='bamboo')[s]}"


# ------------------------------------------------------------------------------------------------------------ rack
# A slotted wooden tile stand: an L-shaped cradle (shelf + backboard) tilted back by RACK_ALPHA, on a riser and a
# plinth. The shelf tilts down towards the backboard, so a standing tile is held by gravity in the shelf/backboard
# corner (no front lip needed; 15 deg is below the 21.8 deg friction angle, so a tile does not slide back on its own). Each tile stands on its bottom face on the shelf and leans its back face on the backboard. Low dividers
# between slots fix the slot positions; their pitch leaves a 10 mm gap between neighbouring tiles so the gripper's
# 5.5 mm fingertips fit beside a tile when its neighbours are already in place.
RACK_ALPHA = math.radians(15.0)  # lean of the standing tiles from vertical
N_SLOTS = len(HAND)
SLOT_PITCH = 0.036
RACK_CLEAR = 0.006               # slack between a seated tile's front face and the lip (room to lower a tile in)
RACK_BACK_T, RACK_BACK_H = 0.008, 0.022
RACK_DIV_T, RACK_DIV_H = 0.003, 0.006
RACK_END_T, RACK_END_H = 0.008, 0.016
RACK_PLINTH_T = 0.006
RACK_CORNER_Z = 0.013            # height of the shelf/backboard corner line above the table
RACK_SHELF_U_LOW = -0.0095       # how far the tilted cradle block reaches below the shelf surface (along -u)
RACK_DENSITY = 650.0             # beech-ish hardwood, kg/m^3
RACK_LEN = N_SLOTS * SLOT_PITCH + 2 * RACK_END_T

SIN_A, COS_A = math.sin(RACK_ALPHA), math.cos(RACK_ALPHA)
# In the rack's (y, z) profile plane: b = "into the backboard" (back and down), u = "up the backboard".
RACK_B = (COS_A, -SIN_A)
RACK_U = (SIN_A, COS_A)
RACK_B_FRONT = -(TILE_T + RACK_CLEAR)  # b of the shelf's front face (relative to the corner line); no front lip, so
                                       # the lower edge of a standing tile's face stays visible


def _corner_y():
    """Corner line y chosen so the plinth (front of riser .. back of backboard) is centred on y = 0."""
    # front-most point of the solid: the cradle block's bottom-front corner; back-most: backboard's top-back corner.
    yf = RACK_B_FRONT * RACK_B[0] + RACK_SHELF_U_LOW * RACK_U[0]
    yb = RACK_BACK_T * RACK_B[0] + RACK_BACK_H * RACK_U[0]
    return -0.5 * (yf + yb)


RACK_CORNER_Y = _corner_y()


def rack_bu_to_yz(b, u):
    return (RACK_CORNER_Y + b * RACK_B[0] + u * RACK_U[0], RACK_CORNER_Z + b * RACK_B[1] + u * RACK_U[1])


def slot_x(k):
    return (k - 0.5 * (N_SLOTS - 1)) * SLOT_PITCH


def seated_tile_pose_rack(k):
    """Pose (pos, quat_wxyz) of a tile seated in slot k, in the rack frame: back face on the backboard, bottom face on
    the shelf, face towards -y. Rotation: about +x by (90 deg - alpha), so tile +y -> rack u and tile +z -> -b."""
    y, z = rack_bu_to_yz(-0.5 * TILE_T, 0.5 * TILE_H)
    th = 0.5 * (math.pi / 2 - RACK_ALPHA)
    return (slot_x(k), y, z), (math.cos(th), math.sin(th), 0.0, 0.0)


def rack_boxes():
    """Collision = visual geometry of the rack as boxes: list of (name, centre_xyz, size_xyz, rot_x_rad)."""
    boxes = []

    def tilted(name, x0, x1, b0, b1, u0, u1):
        bc, uc = 0.5 * (b0 + b1), 0.5 * (u0 + u1)
        y, z = rack_bu_to_yz(bc, uc)
        boxes.append((name, (0.5 * (x0 + x1), y, z), (x1 - x0, b1 - b0, u1 - u0), -RACK_ALPHA))

    half = 0.5 * RACK_LEN
    # tilted cradle: shelf block and backboard
    tilted("shelf", -half, half, RACK_B_FRONT, RACK_BACK_T, RACK_SHELF_U_LOW, 0.0)
    tilted("backboard", -half, half, 0.0, RACK_BACK_T, RACK_SHELF_U_LOW, RACK_BACK_H)
    # dividers between slots (not at the ends: the end walls bound slots 0 and 13)
    for k in range(1, N_SLOTS):
        xd = 0.5 * (slot_x(k - 1) + slot_x(k))
        tilted(f"div{k:02d}", xd - 0.5 * RACK_DIV_T, xd + 0.5 * RACK_DIV_T, RACK_B_FRONT, 0.001,
               -0.001, RACK_DIV_H)
    tilted("end_l", -half, -half + RACK_END_T, RACK_B_FRONT, RACK_BACK_T, RACK_SHELF_U_LOW, RACK_END_H)
    tilted("end_r", half - RACK_END_T, half, RACK_B_FRONT, RACK_BACK_T, RACK_SHELF_U_LOW, RACK_END_H)
    # axis-aligned riser under the tilted block, top kept below the shelf's lowest point (the corner line)
    y_front = rack_bu_to_yz(RACK_B_FRONT, RACK_SHELF_U_LOW)[0] + 0.002
    y_back = rack_bu_to_yz(RACK_BACK_T, RACK_SHELF_U_LOW)[0] - 0.001
    z0, z1 = RACK_PLINTH_T - 0.002, RACK_CORNER_Z - 0.0015
    boxes.append(("riser", (0.0, 0.5 * (y_front + y_back), 0.5 * (z0 + z1)), (RACK_LEN, y_back - y_front, z1 - z0), 0.0))
    # plinth
    yf = rack_bu_to_yz(RACK_B_FRONT, RACK_SHELF_U_LOW)[0] - 0.004
    yb = rack_bu_to_yz(RACK_BACK_T, RACK_BACK_H)[0] + 0.004
    boxes.append(("plinth", (0.0, 0.5 * (yf + yb), 0.5 * RACK_PLINTH_T), (RACK_LEN + 0.004, yb - yf, RACK_PLINTH_T), 0.0))
    return boxes


# ----------------------------------------------------------------------------------------------- station layout
RACK_POS = (0.0, 0.085)  # rack plinth centre on the table (station frame x, y)
SCATTER_SEED = 7
