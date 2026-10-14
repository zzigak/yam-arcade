"""Install chosen candidates as final assets and (re)write assets/sam3d/INDEX.md.
usage: python -I assemble.py            (selection table below)"""
import json, os, shutil

ROOT = os.path.expanduser("~/projects/yam-arcade/assets/sam3d")
W = f"{ROOT}/_work"
SRC = json.load(open(f"{W}/sources.json"))
FILES = ["visual.glb", "visual.obj", "visual_lo.obj", "collision_hollow.obj", "info.json", "contact.png"]

# final name: (candidate dir under _work/cand or ROOT, source key, verdict, notes)
SEL = json.load(open(f"{W}/selection.json"))


def install(name, cand, src_key, verdict, notes):
    cdir = cand if cand.startswith("/") else f"{W}/cand/{cand}"
    out = f"{ROOT}/{name}"
    if os.path.realpath(cdir) != os.path.realpath(out):
        shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
        for f in FILES:
            if os.path.exists(f"{cdir}/{f}"):
                shutil.copy(f"{cdir}/{f}", f"{out}/{f}")
        for f in os.listdir(cdir):  # obj material side files, if any
            if f.endswith((".mtl", ".png")) and f not in FILES:
                shutil.copy(f"{cdir}/{f}", f"{out}/{f}")
    info = json.load(open(f"{out}/info.json"))
    info["name"] = name
    info["source_image"] = SRC[src_key]
    info["input_png"] = f"_work/inputs/{src_key}.png (mask: _work/inputs/{src_key}_mask.png)"
    info["verdict"] = verdict
    info["notes"] = notes
    json.dump(info, open(f"{out}/info.json", "w"), indent=1)
    return info


def row(info):
    bb = info["bbox_m"]
    dims = f"{bb[0]*100:.1f} x {bb[1]*100:.1f} x {bb[2]*100:.1f}"
    cap = f"{info['capacity_l']:.2f}" if "capacity_l" in info else "-"
    col = info.get("collision")
    colt = f"yes ({col['faces']} f, wall {col['wall_m']*1000:.1f} mm, floor top {col['floor_top_z_m']*1000:.0f} mm)" if col else "-"
    s = info["source_image"]
    src = f"[{s['title'][:38]}]({s['page']}) - {s['creator'] or 'n/a'} - {s['license']}"
    return f"| `{info['name']}` | {dims} | {cap} | {colt} | {info['verdict']} | seed {info['seed']} | {src} | {info['notes']} |"


HEAD = """# SAM 3D assets (yam-arcade)

Made with SAM 3D Objects (`~/tools/sam-3d-objects`) from single real photos: photo -> object mask (rembg isnet, or
SAM-ViT-H box prompt where several objects touched) -> SAM 3D (2 seeds for the important objects, best kept) ->
post-processing in `_work/scripts/process.py` (all scripts, masks, overlays and raw outputs are under `_work/`).

## Conventions (read this)

- **Frame: Z-up, metres, base at z = 0, bounding box centred on x = y = 0.** Objects with a handle are yawed so the
  **handle points to +x** (the pitcher's spout therefore points to -x). `info.json -> collision.cavity_center_xy_m` is
  the centre of the opening (it is not at x = y = 0 when there is a handle).
- **OBJ files are Z-up as-is** (Genesis `gs.morphs.Mesh(file=...obj)` with its default `file_meshes_are_zup=True`
  loads them upright; in Blender import with `up_axis='Z', forward_axis='Y'`).
- **`visual.glb` follows the glTF spec (Y-up)**: Blender / three.js import it upright. If you load it with trimesh,
  or in Genesis, you get raw Y-up coordinates: map `(x, y, z) -> (x, -z, y)` or pass `file_meshes_are_zup=False`, or
  just use `visual.obj`.
- Colours are **per-vertex** (SAM 3D gives no UV texture), in `visual.glb` (COLOR_0) and as `v x y z r g b` in the OBJs.
- `visual.obj` ~200k faces, `visual_lo.obj` 40k faces (decimated, colours re-sampled); neither is guaranteed
  watertight. Use `collision_hollow.obj` for physics.
- **`collision_hollow.obj`** (containers): watertight, open-topped, solid bottom, solid handle, walls >= the stated
  thickness, inner cavity following the SAM 3D cavity (or, for the pitcher and urn, the outer shape offset inward),
  a few thousand faces. Built by voxelising the scaled mesh (0.6-1.5 mm voxels), thickening thin walls to the target,
  removing lids, marching cubes, light Taubin smoothing and quadric decimation. The rim is slightly ragged
  (+-1 voxel). Lids (pitcher, jar, urn) exist only in the visual; the collision mesh is open at the top.
- **Capacity** = brimful volume of the collision cavity with the object upright (the level at which water would
  spill over the lowest rim point), from the voxel grid.
- Masses are not set here (choose them per station: e.g. glass 2.5 g/cm^3, porcelain 2.4, PP/SAN plastic ~1.1).

## Assets

| name | bbox x * y * z (cm) | capacity (L) | collision_hollow | verdict | seed | source photo - author - license | notes |
|---|---|---|---|---|---|---|---|
"""


def main():
    infos = []
    import subprocess
    for name, (cand, key, verdict, notes) in SEL["assets"].items():
        install(name, cand, key, verdict, notes)
    subprocess.run([os.path.expanduser("~/simquarry/genesis-world/.venv/bin/python"), "-I", f"{W}/scripts/checks.py",
                    *SEL["assets"].keys()], check=True, capture_output=True)
    infos = [json.load(open(f"{ROOT}/{n}/info.json")) for n in SEL["assets"]]
    txt = HEAD + "\n".join(row(i) for i in infos) + "\n"
    q = ["\n## Pouring quick facts (station frame of each asset)\n",
         "| name | opening centre x, y (cm) | brim level z (cm) | floor top z (cm) | spout tip x, y, z (cm) | handle |",
         "|---|---|---|---|---|---|"]
    for i in infos:
        col = i.get("collision")
        if not col:
            continue
        ck = i.get("checks", {})
        sp = ck.get("spout_tip_m")
        q.append(f"| `{i['name']}` | {col['cavity_center_xy_m'][0]*100:.1f}, {col['cavity_center_xy_m'][1]*100:.1f} | "
                 f"{col['fill_level_z_m']*100:.1f} | {col['floor_top_z_m']*100:.1f} | "
                 + (f"{sp[0]*100:.1f}, {sp[1]*100:.1f}, {sp[2]*100:.1f}" if sp else "-") + " | "
                 + ("+x" if "handle along +x" in i["frame"] else "-") + " |")
    txt += "\n".join(q) + "\n"
    if SEL.get("not_delivered"):
        txt += "\n## Not delivered / rejected\n\n" + "\n".join(f"- {x}" for x in SEL["not_delivered"]) + "\n"
    if SEL.get("status"):
        txt += "\n## Status\n\n" + SEL["status"] + "\n"
    txt += "\n## Size choices\n\n" + "\n".join(
        f"- `{i['name']}`: {i['target']['dim']} = {i['target']['size_m']*100:.1f} cm - {i['target']['why']}"
        + (f"; depth rescaled x{i['y_rescale']['factor']}: {i['y_rescale']['why']}" if i.get("y_rescale") else "")
        for i in infos) + "\n"
    txt += ("\n## Licenses\n\nPhotos are CC0 / public domain / CC BY / CC BY-SA, except `urn` (CC BY-NC-SA 2.0: "
            "non-commercial research use only). Attribute the authors above if these meshes are shown publicly.\n")
    open(f"{ROOT}/INDEX.md", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
