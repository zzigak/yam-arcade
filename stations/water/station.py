"""Water station: a Brita-style pitcher of water and three cups on the table; pour into all three.

    env = build(backend="gpu", pitcher_mount="free" | "oracle", robot=None | <robot helper>)
    env.scene.step()
    env.volumes()   -> dict of ml per container + spilled / in-flight
    env.success()   -> (bool, details)

Liquid: Genesis SPH (WCSPH) coupled to rigid bodies through the legacy coupler (see NOTES.md for why).
"""
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import layout  # noqa: E402
from containers import ALL as SPECS, build as build_container  # noqa: E402
from geometry import FLUID_OFFSET  # noqa: E402

ASSET_DIR = os.path.join(HERE, "assets")

# ----------------------------------------------------------------------------------------------- parameters
PARTICLE_SIZE = 0.004          # SPH particle diameter d (m): teacup half-fill = ~1200 particles, stream 3-5 wide
WATER_ML = 1200.0              # initial water in the pitcher
CUP_NAMES = ("tumbler", "mug", "teacup")   # pour order (fullest pitcher -> tallest glass, lip lowest)
BOUNDARY_DENSITY = True       # analytic wall share of the SPH density (coupler_patch.BoundarySPHSolver)
FILL_TARGET = 0.5              # success: every cup holds >= 50 % of its brim capacity ...
SPILL_LIMIT_ML = 25.0          # ... and less than 25 ml ended up outside every container

TABLE_Z = layout.TABLE_TOP_Z
# Station-local placement (x, y, yaw_deg); container frames sit on the table top.
PLACEMENT = {
    "pitcher": (0.25, 0.06, 180.0),    # spout toward -x (the cups), handle toward +x (the right arm)
    "tumbler": (-0.02, 0.18, 0.0),     # cups in a column at x = -0.02, handles toward -x (away from the pitcher)
    "mug": (-0.02, 0.04, 0.0),
    "teacup": (-0.02, -0.10, 0.0),
}

LIQUID = dict(
    rho=1000.0,        # water
    stiffness=50000.0,  # WCSPH Tait B (Pa): c = sqrt(B * 7 / rho) = 18.7 m/s, ~10x the fastest flow -> ~1 % compression
    exponent=7.0,      # Tait exponent for water
    mu=3e-4,           # kinematic viscosity (m^2/s): artificial viscosity at the usual WCSPH level (alpha ~0.02 for
                       # h = 8 mm, c = 18.7 m/s); ~300x water's 1e-6, which a 4 mm resolution cannot resolve anyway
    gamma=0.005,       # cohesion ("surface tension") coefficient of Genesis' SPH
)

# Liquid-wall coupling: no-slip walls (coup_friction 1: the tangential velocity relative to the wall is removed with
# the normal one, as for a viscous liquid at a wall; 0 lets wall particles skate and boil), 2 mm influence band,
# perfectly inelastic, <= 1 mm SDF cells on the convex wall pieces (at least 16 cells per axis instead of Genesis' 32:
# with ~1900 pieces the default minimum alone was 78 M SDF cells = 1.6 GB of GPU memory).
RIGID_COUP = dict(coup_friction=1.0, coup_softness=0.002, coup_restitution=0.0, sdf_cell_size=0.001, sdf_min_res=16,
                  sdf_max_res=128)
# Stiff contacts for the grasp: a 2 ms constraint time constant (default 10 ms lets an 80 N grasp sink ~4 mm into the
# handle and the full pitcher rotated 10-60 deg in the grip) and 5 noslip iterations (Genesis' post-solve against
# tangential creep).  Residual creep remains: the pitcher slides ~0.3 mm/s down the grip bar while held (NOTES.md,
# known issues).  Tried: noslip 20 (no better); the elliptic cone (impratio 100) cut the creep to ~1 mm in a dry
# rehearsal but its wet robot run slipped at a fast tilt, the pitcher came down on the teacup and knocked the cups
# over - so the exported run uses this setting.
RIGID_OPTIONS = dict(enable_self_collision=False, constraint_timeconst=0.002, noslip_iterations=5)
FRICTION = dict(pitcher=0.6, tumbler=0.5, mug=0.55, teacup=0.55, table=0.5)


def load_raw(raw_dir):
    """raw.npz -> plain dict of arrays, decompressed ONCE. (Indexing an NpzFile re-reads and decompresses the whole
    array on every access: per-frame raw["particles"][i] in a loop cost a full T x N x 3 copy each time.)"""
    with np.load(os.path.join(raw_dir, "raw.npz")) as z:
        return {k: z[k] for k in z.files}


def yaw_quat(deg):
    a = np.radians(deg) / 2
    return np.array([np.cos(a), 0.0, 0.0, np.sin(a)])


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# ----------------------------------------------------------------------------------------------- assets
def write_assets():
    """Generate (deterministically) the container OBJs / collision pieces; returns name -> (Container, info)."""
    out = {}
    for name in ("pitcher", "tumbler", "mug", "teacup"):
        c = build_container(name)
        out[name] = (c, c.write(ASSET_DIR))
    return out


def _mjcf(name, info, mount, rgba, friction, init_pose=None):
    """MJCF text for one container. mount: 'free' (free joint) or 'oracle' (3 slides + 3 hinges, PD-driven)."""
    m, com, I = info["mass"], info["com"], info["inertia"]
    full = f"{I[0,0]:.9g} {I[1,1]:.9g} {I[2,2]:.9g} {I[0,1]:.9g} {I[0,2]:.9g} {I[1,2]:.9g}"
    assets = [f'<mesh name="c{i:03d}" file="{p}"/>' for i, p in enumerate(info["collision"])]
    assets += [f'<mesh name="v_{k}" file="{p}"/>' for k, p in info["visual"].items()]
    geoms = [f'<geom type="mesh" mesh="c{i:03d}" group="3" contype="1" conaffinity="1" friction="{friction} 0.005 0.0001"/>'
             for i in range(len(info["collision"]))]
    geoms += [f'<geom type="mesh" mesh="v_{k}" group="1" contype="0" conaffinity="0" rgba="{" ".join(map(str, rgba))}"/>'
              for k in info["visual"]]
    body_inner = (f'<inertial pos="{com[0]:.9g} {com[1]:.9g} {com[2]:.9g}" mass="{m:.9g}" fullinertia="{full}"/>\n'
                  + "\n".join(geoms))
    pos, quat = init_pose
    if mount == "free":
        body = (f'<body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{" ".join(map(str, quat))}">\n'
                f'<freejoint name="{name}_free"/>\n{body_inner}\n</body>')
    else:
        tiny = '<inertial pos="0 0 0" mass="0.02" diaginertia="2e-5 2e-5 2e-5"/>'
        body = (f'<body name="{name}_mount" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{" ".join(map(str, quat))}">\n'
                f'<body name="{name}_mx">{tiny}<joint name="mx" type="slide" axis="1 0 0" limited="false"/>\n'
                f'<body name="{name}_my">{tiny}<joint name="my" type="slide" axis="0 1 0" limited="false"/>\n'
                f'<body name="{name}_mz">{tiny}<joint name="mz" type="slide" axis="0 0 1" limited="false"/>\n'
                f'<body name="{name}_ryaw">{tiny}<joint name="ryaw" type="hinge" axis="0 0 1" limited="false"/>\n'
                f'<body name="{name}_rpitch">{tiny}<joint name="rpitch" type="hinge" axis="0 1 0" limited="false"/>\n'
                f'<body name="{name}"><joint name="rroll" type="hinge" axis="1 0 0" limited="false"/>\n'
                f'{body_inner}\n</body></body></body></body></body></body></body>')
    return (f'<mujoco model="{name}">\n<compiler angle="radian" inertiafromgeom="false"/>\n'
            f'<asset>\n' + "\n".join(assets) + f'\n</asset>\n<worldbody>\n{body}\n</worldbody>\n</mujoco>\n')


def rest_spacing(d=PARTICLE_SIZE):
    """Genesis SPH gives each particle the volume 0.8 d^3 and a cubic lattice of spacing s has rest density rho0 when
    s^3 = 0.8 d^3: the water's rest spacing is s = 0.928 d (3.71 mm for d = 4 mm)."""
    return d * 0.8 ** (1.0 / 3.0)


def water_init_particles(container, water_ml, pose, d=PARTICLE_SIZE):
    """Cubic lattice at the rest spacing filling the pitcher bottom-up: N = water_ml / (0.8 d^3) particles.

    Starting at rest density (instead of Genesis' lattice at spacing d, which is 20 % under-dense) avoids a 2.5 cm
    collapse-and-rebound of the column at t = 0.  Returns station-frame positions and the fill height (local z)."""
    spec = container.spec
    s = rest_spacing(d)
    n = int(round(water_ml * 1e-6 / (0.8 * d ** 3)))
    margin = FLUID_OFFSET + 0.5 * s
    rmax = float(container._R_in.max())
    lo = np.array([-rmax, -rmax, spec.z_floor + margin])
    xs = np.arange(lo[0], rmax, s)
    ys = np.arange(lo[1], rmax, s)
    zs = np.arange(lo[2], spec.z_rim - 0.005, s)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    r = np.hypot(P[:, 0], P[:, 1])
    phi = np.arctan2(P[:, 1], P[:, 0])
    R_in = container._lookup(container._R_in, phi, np.clip(P[:, 2], spec.z_floor, spec.z_rim))
    P = P[r < R_in - margin]
    P = P[np.lexsort((np.hypot(P[:, 0], P[:, 1]), P[:, 2]))][:n]
    assert len(P) == n, f"{spec.name} too small for the requested water"
    R = quat_to_R(pose[1])
    return P @ R.T + pose[0], float(P[:, 2].max() + 0.5 * s)


# ----------------------------------------------------------------------------------------------- env
@dataclass
class Env:
    scene: object
    containers: dict          # name -> Container (geometry)
    infos: dict               # name -> asset info (mass, capacity, files)
    entities: dict            # name -> genesis entity
    links: dict               # name -> genesis link carrying the container geometry
    liquid: object
    table: object
    robot: object = None
    pitcher_mount: str = "free"
    v_particle: float = 0.8 * PARTICLE_SIZE ** 3   # rest volume per particle (m^3): Genesis SPH particle volume
    cams: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- state access
    def particles(self):
        p = self.liquid.get_particles_pos()
        p = p.cpu().numpy() if hasattr(p, "cpu") else np.asarray(p)
        return p.reshape(-1, 3)

    def pose(self, name):
        link = self.links[name]
        pos = link.get_pos()
        quat = link.get_quat()
        pos = pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
        quat = quat.cpu().numpy() if hasattr(quat, "cpu") else np.asarray(quat)
        return pos.reshape(3), quat.reshape(4)

    def to_local(self, name, pts):
        pos, quat = self.pose(name)
        return (pts - pos) @ quat_to_R(quat)

    def classify(self, pts=None, rim_margin=0.0):
        """Per-particle label: index into ['pitcher', *CUP_NAMES] or -1 (outside every container);
        plus per-container in-wall masks."""
        pts = self.particles() if pts is None else pts
        names = ["pitcher", *CUP_NAMES]
        label = np.full(len(pts), -1)
        in_wall = {}
        for i, n in enumerate(names):
            loc = self.to_local(n, pts)
            cav, wall = self.containers[n].classify_local(loc, rim_margin=rim_margin)
            label[cav & (label < 0)] = i
            in_wall[n] = wall
        return label, in_wall, names

    def volumes(self, pts=None):
        """Water volume (ml) per container and outside all of them; uses the calibrated particle volume."""
        pts = self.particles() if pts is None else pts
        label, _, names = self.classify(pts)
        out = {n: float(np.sum(label == i)) * self.v_particle * 1e6 for i, n in enumerate(names)}
        outside = label < 0
        on_support = outside & (pts[:, 2] < TABLE_Z + 0.03)
        out["spilled"] = float(np.sum(on_support)) * self.v_particle * 1e6     # on the table / floor
        out["airborne"] = float(np.sum(outside & ~on_support)) * self.v_particle * 1e6  # in the stream / splashes
        return out

    def frame_stats(self, pts=None):
        """Volumes (ml) and per-container counts of particles inside the visual wall solids, in one pass."""
        pts = self.particles() if pts is None else pts
        label, in_wall, names = self.classify(pts)
        out = {n: float(np.sum(label == i)) * self.v_particle * 1e6 for i, n in enumerate(names)}
        outside = label < 0
        on_support = outside & (pts[:, 2] < TABLE_Z + 0.03)
        out["spilled"] = float(np.sum(on_support)) * self.v_particle * 1e6
        out["airborne"] = float(np.sum(outside & ~on_support)) * self.v_particle * 1e6
        return out, {n: int(w.sum()) for n, w in in_wall.items()}

    def particle_speeds(self):
        v = self.liquid.get_particles_vel()
        v = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
        return np.linalg.norm(v.reshape(-1, 3), axis=1)

    def success(self, vols=None):
        vols = self.volumes() if vols is None else vols
        detail = {}
        ok = True
        for n in CUP_NAMES:
            cap = self.infos[n]["capacity"] * 1e6
            frac = vols[n] / cap
            detail[n] = dict(ml=vols[n], capacity_ml=cap, fill=frac, ok=frac >= FILL_TARGET)
            ok &= frac >= FILL_TARGET
        spill = vols["spilled"] + vols["airborne"]
        detail["spill"] = dict(ml=spill, limit_ml=SPILL_LIMIT_ML, ok=spill < SPILL_LIMIT_ML)
        ok &= spill < SPILL_LIMIT_ML
        return bool(ok), detail


def container_init_pose(name):
    x, y, yaw = PLACEMENT[name]
    return np.array([x, y, TABLE_Z]), yaw_quat(yaw)


def build(backend="gpu", pitcher_mount="free", robot_builder=None, show_viewer=False, dt=1.0 / 300, substeps=16,
          water_ml=WATER_ML, particle_size=PARTICLE_SIZE, cameras=None, liquid=None, vis_mode="particle",
          pressure_solver="WCSPH", fast_coupling=True, coup=None, ballast_kg=0.0, rigid=None,
          boundary_density=BOUNDARY_DENSITY, water_in=None, logging_level="warning"):
    """water_ml = 0 builds a dry scene (no SPH); ballast_kg then adds that mass at the water's centre of mass to the
    pitcher (rigid-only robot rehearsals on the CPU)."""
    import genesis as gs

    gs.init(backend=gs.gpu if backend == "gpu" else gs.cpu, precision="32", logging_level=logging_level)
    made = write_assets()
    containers = {n: c for n, (c, _) in made.items()}
    infos = {n: info for n, (_, info) in made.items()}
    liquid = {**LIQUID, **(liquid or {})}
    rcoup = {**RIGID_COUP, **(coup or {})}
    rigid_opts = {**RIGID_OPTIONS, **(rigid or {})}
    if isinstance(rigid_opts.get("friction_cone"), str):
        rigid_opts["friction_cone"] = getattr(gs.friction_cone, rigid_opts["friction_cone"])

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps, gravity=(0.0, 0.0, -9.81)),
        sph_options=gs.options.SPHOptions(
            particle_size=particle_size,
            pressure_solver=pressure_solver,
            lower_bound=(-0.62, -0.40, 0.0),
            upper_bound=(0.62, 0.40, 1.40),
        ),
        rigid_options=gs.options.RigidOptions(**rigid_opts),
        vis_options=gs.options.VisOptions(
            background_color=layout.BACKGROUND_RGB,
            ambient_light=(0.35, 0.35, 0.35),
            lights=[dict(type="directional", dir=(-0.4, 0.6, -1.0), color=(1.0, 1.0, 1.0), intensity=3.5)],
            show_world_frame=False,
            rendered_envs_idx=[0],
        ),
        viewer_options=gs.options.ViewerOptions(camera_pos=(0.0, -1.2, 1.4), camera_lookat=(0.0, 0.0, 0.8),
                                                camera_fov=40),
        show_viewer=show_viewer,
    )
    if fast_coupling:
        import coupler_patch
        coupler_patch.apply(scene)
        coupler_patch.apply_boundary_density(scene, enabled=boundary_density)
    floor = scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(needs_coup=True, coup_friction=0.0),
        surface=gs.surfaces.Default(color=layout.FLOOR_RGB),
    )
    T = layout.TABLE
    table = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, TABLE_Z - 0.5 * T["top_thickness"]),
            size=(T["size_x"], T["size_y"], T["top_thickness"]),
            fixed=True,
        ),
        material=gs.materials.Rigid(friction=FRICTION["table"], needs_coup=True, **rcoup),
        surface=gs.surfaces.Default(color=(0.86, 0.80, 0.70)),
    )

    entities, links = {}, {}
    if ballast_kg > 0:
        info = made["pitcher"][1]
        m0, c0, I0 = info["mass"], info["com"], info["inertia"]
        cb = np.array([0.0, 0.0, 0.075])    # 1.2 L fills to ~14 cm: COM ~7.5 cm above the base
        m = m0 + ballast_kg
        com = (m0 * c0 + ballast_kg * cb) / m
        Ib = ballast_kg * np.diag([0.0035, 0.0030, 0.0020])   # ~ water block 12 x 9 x 14 cm
        I = I0 + m0 * (np.dot(c0 - com, c0 - com) * np.eye(3) - np.outer(c0 - com, c0 - com)) \
            + Ib + ballast_kg * (np.dot(cb - com, cb - com) * np.eye(3) - np.outer(cb - com, cb - com))
        made["pitcher"] = (made["pitcher"][0], {**info, "mass": m, "com": com, "inertia": I})
    for name, (c, info) in made.items():
        mount = pitcher_mount if name == "pitcher" else "free"
        xml = _mjcf(name, info, mount, c.spec.rgba, FRICTION[name], init_pose=container_init_pose(name))
        path = os.path.join(ASSET_DIR, f"{name}_{mount}.xml")
        with open(path, "w") as f:
            f.write(xml)
        ent = scene.add_entity(
            morph=gs.morphs.MJCF(file=path, align=False),
            material=gs.materials.Rigid(friction=FRICTION[name], needs_coup=True, **rcoup),
        )
        entities[name] = ent
        links[name] = ent.get_link(name)

    water, water_pos, z_fill = None, None, 0.0
    if water_in is not None:     # test fills: {container: ml}
        water_pos = np.concatenate([water_init_particles(containers[k], ml, container_init_pose(k), d=particle_size)[0]
                                    for k, ml in water_in.items()])
    elif water_ml > 0:
        water_pos, z_fill = water_init_particles(containers["pitcher"], water_ml, container_init_pose("pitcher"),
                                                 d=particle_size)
    if water_pos is not None:
        water = scene.add_entity(
            morph=gs.morphs.Nowhere(n_particles=len(water_pos)),
            material=gs.materials.SPH.Liquid(sampler="regular", **liquid),
            surface=gs.surfaces.Default(color=(0.35, 0.62, 0.95), vis_mode=vis_mode),
        )

    robot = robot_builder(scene) if robot_builder is not None else None

    cams = {}
    for cname, cfg in (cameras or {}).items():
        cams[cname] = scene.add_camera(res=cfg.get("res", (1280, 720)), pos=cfg["pos"], lookat=cfg["lookat"],
                                       fov=cfg.get("fov", 40), GUI=False)
    if fast_coupling:
        coupler_patch.alloc_link_bounds(scene)
    scene.build(n_envs=0)
    if fast_coupling:
        coupler_patch.setup_link_bounds(scene)
    env = Env(scene=scene, containers=containers, infos=infos, entities=entities, links=links, liquid=water,
              table=table, robot=robot, pitcher_mount=pitcher_mount, cams=cams,
              v_particle=0.8 * particle_size ** 3)
    if water is not None:
        water.set_particles_pos(water_pos.astype(np.float32))
        water.set_particles_vel(np.zeros_like(water_pos, dtype=np.float32))
        water.set_particles_active(np.ones(len(water_pos), dtype=bool))
    env.z_fill = z_fill
    env.coupler_class = type(scene.sim._coupler).__name__ + "+" + type(scene.sim.sph_solver).__name__
    env.pitcher_mount_pose = container_init_pose("pitcher")
    if pitcher_mount == "oracle":
        setup_oracle(env)
    return env


# ----------------------------------------------------------------------------------------------- oracle mount
ORACLE_DOFS = ("mx", "my", "mz", "ryaw", "rpitch", "rroll")


def setup_oracle(env, kp_lin=3.0e4, kp_ang=600.0):
    """PD gains of the oracle mount (disclosed: the pitcher is driven by a 6-DoF PD 'hand', not the robot)."""
    ent = env.entities["pitcher"]
    idx = [ent.get_joint(n).dofs_idx_local[0] for n in ORACLE_DOFS]
    m_tot = env.infos["pitcher"]["mass"] + WATER_ML * 1e-3
    kv_lin = 2.0 * np.sqrt(kp_lin * m_tot)
    kv_ang = 2.0 * np.sqrt(kp_ang * 0.03)
    ent.set_dofs_kp([kp_lin] * 3 + [kp_ang] * 3, idx)
    ent.set_dofs_kv([kv_lin] * 3 + [kv_ang] * 3, idx)
    ent.set_dofs_force_range([-200.0] * 3 + [-50.0] * 3, [200.0] * 3 + [50.0] * 3, idx)
    env.oracle_idx = idx
    env.oracle_m = m_tot
