"""Water station run: settle, pour into the three cups, settle; writes raw per-frame data + preview frames.

    python run.py --mode oracle   # pitcher driven by a disclosed 6-DoF PD mount
    python run.py --mode robot    # Big YAM cart grasps the pitcher handle and pours (see robot_pour.py)

Outputs (in --out, default out/water/raw_<mode>):
    raw.npz       per recorded frame (30 fps): t, particles (N, 3), poses, volumes, controller state
    frames/       preview PNGs from the Genesis camera(s)
    events.json   controller events, timing
Then export_water.py turns raw.npz into the compositor export (out/water/) with splashsurf surfaces + checks.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import station as st  # noqa: E402

CAMERAS = dict(
    # far side of the table looking back at the robot (the arms do not hide the pour), and a low view from the left
    front=dict(pos=(0.30, 1.05, 1.30), lookat=(0.06, -0.02, 0.86), fov=40, res=(1280, 720)),
    side=dict(pos=(-0.72, -0.30, 1.02), lookat=(0.04, 0.04, 0.84), fov=40, res=(1280, 720)),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("oracle", "robot"), default="oracle")
    ap.add_argument("--out", default=None)
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--final", type=float, default=1.5)
    ap.add_argument("--max-secs", type=float, default=40.0)
    ap.add_argument("--substeps", type=int, default=40)
    ap.add_argument("--render-every", type=int, default=1, help="render a preview frame every k recorded frames")
    ap.add_argument("--order", default="tumbler,mug,teacup")
    ap.add_argument("--fill", type=float, default=0.72)
    args = ap.parse_args()
    out = args.out or os.path.join(st.ROOT, "out", "water", f"raw_{args.mode}")
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)

    t0 = time.time()
    robot_builder = None
    if args.mode == "robot":
        import robot_pour
        robot_builder = robot_pour.builder
    env = st.build(backend="gpu", pitcher_mount="oracle" if args.mode == "oracle" else "free", cameras=CAMERAS,
                   substeps=args.substeps, robot_builder=robot_builder)
    print(f"build {time.time() - t0:.1f}s particles={env.particles().shape[0]} coupler={env.coupler_class}", flush=True)

    from pour import OracleDriver, PourController
    dt = env.scene.sim_options.dt
    if args.mode == "oracle":
        driver = OracleDriver(env)
    else:
        import robot_pour
        driver = robot_pour.RobotDriver(env)
    ctrl = PourController(env, driver, order=args.order.split(","), fill=args.fill, dt=dt,
                          slow=1.5 if args.mode == "robot" else 1.0)

    fps = 30
    spf = int(round(1.0 / (fps * dt)))
    rec = dict(t=[], particles=[], phase=[], beta=[], lip=[], speed_mean=[], speed_max=[], speed_p99=[])
    pose_keys = ["pitcher", *st.CUP_NAMES]
    for k in pose_keys:
        rec[f"{k}/pos"], rec[f"{k}/quat"] = [], []
    vol_keys = ["pitcher", *st.CUP_NAMES, "spilled", "airborne"]
    for k in vol_keys:
        rec[f"vol/{k}"] = []
    for k in pose_keys:
        rec[f"inwall/{k}"] = []
    robot_poses = []
    n_frame, t_sim = 0, 0.0
    phase = "settle"
    settle_frames = int(args.settle * fps)
    final_frames = None
    tw = time.time()
    while True:
        # ---- record
        p = env.particles()
        vols, inwall = env.frame_stats(p)
        sp = env.particle_speeds()
        rec["speed_mean"].append(sp.mean())
        rec["speed_max"].append(sp.max())
        rec["speed_p99"].append(np.percentile(sp, 99))
        for k in pose_keys:
            rec[f"inwall/{k}"].append(inwall[k])
        rec["t"].append(t_sim)
        rec["particles"].append(p.astype(np.float32))
        rec["phase"].append(phase)
        rec["beta"].append(ctrl.beta)
        rec["lip"].append(ctrl.lip.copy())
        for k in pose_keys:
            pos, q = env.pose(k)
            rec[f"{k}/pos"].append(pos)
            rec[f"{k}/quat"].append(q)
        for k in vol_keys:
            rec[f"vol/{k}"].append(vols[k])
        if env.robot is not None:
            robot_poses.append(env.robot.poses())
        if n_frame % args.render_every == 0:
            import imageio.v3 as iio
            for cn, cam in env.cams.items():
                iio.imwrite(os.path.join(out, "frames", f"{cn}_{n_frame:05d}.png"), cam.render()[0])
        if n_frame % 15 == 0:
            print(f"[{time.time() - tw:7.1f}s] t={t_sim:6.2f} phase={phase:7s} beta={np.degrees(ctrl.beta):6.1f} "
                  + " ".join(f"{k}={vols[k]:.0f}" for k in vol_keys) + f" inwall={sum(inwall.values())} "
                  f"v={sp.mean():.3f}/{sp.max():.2f} events={ctrl.events[-1:] if ctrl.events else ''}", flush=True)
        # ---- phase logic
        if phase == "settle" and n_frame >= settle_frames:
            phase = "pour"
        elif phase == "pour" and ctrl.done:
            phase = "final"
            final_frames = n_frame + int(args.final * fps)
        elif phase == "final" and n_frame >= final_frames:
            break
        if t_sim > args.max_secs:
            print("max time reached", flush=True)
            break
        # ---- step
        for _ in range(spf):
            if phase != "settle":
                ctrl.step()
            driver.update()
            env.scene.step()
        t_sim += spf * dt
        n_frame += 1

    np.savez_compressed(
        os.path.join(out, "raw.npz"),
        **{k: np.asarray(v) for k, v in rec.items() if k not in ("phase",)},
        phase=np.array(rec["phase"]),
    )
    if robot_poses:
        import pickle
        with open(os.path.join(out, "robot_poses.pkl"), "wb") as f:
            pickle.dump(robot_poses, f)
    import containers
    meta = dict(mode=args.mode, cups=containers.CUPS, events=ctrl.events, n_frames=n_frame + 1, fps=fps, dt=dt, substeps=args.substeps,
                wall_secs=time.time() - t0, n_particles=int(env.particles().shape[0]), v_particle=env.v_particle,
                capacities_ml={n: env.infos[n]["capacity"] * 1e6 for n in env.infos},
                masses_kg={n: env.infos[n]["mass"] for n in env.infos}, coupler=env.coupler_class,
                liquid=st.LIQUID, rigid_coup=st.RIGID_COUP, particle_size=st.PARTICLE_SIZE,
                success=env.success()[1], final_volumes=env.volumes())
    if env.robot is not None:
        meta["robot"] = env.robot.summary()
    with open(os.path.join(out, "events.json"), "w") as f:
        json.dump(meta, f, indent=1, default=float)
    print(json.dumps(meta["final_volumes"], indent=1), flush=True)
    print(json.dumps(meta["success"], indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
