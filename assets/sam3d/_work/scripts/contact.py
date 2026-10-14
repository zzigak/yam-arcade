"""Make QA contact sheets: (1) collision section meshes (genesis venv part), (2) tile Blender renders (PIL part).
usage: python -I contact.py section NAME...   |   python -I contact.py tile NAME..."""
import sys, os, json
ROOT = os.path.expanduser("~/projects/yam-arcade/assets/sam3d")
mode, names = sys.argv[1], sys.argv[2:]
if mode == "section":
    import trimesh, numpy as np
    for n in names:
        p = f"{ROOT}/{n}/collision_hollow.obj"
        if not os.path.exists(p):
            continue
        m = trimesh.load(p, force="mesh")
        info = json.load(open(f"{ROOT}/{n}/info.json"))
        cy = info["collision"]["cavity_center_xy_m"][1]
        h = m.slice_plane([0, cy, 0], [0, 1, 0], cap=True)   # keep the far half; camera looks from -y into the cut
        h.export(f"{ROOT}/_work/qa/" + n.replace("/", "__") + "_section.obj")
else:
    from PIL import Image, ImageDraw
    Q = f"{ROOT}/_work/qa"
    for n in names:
        info = json.load(open(f"{ROOT}/{n}/info.json"))
        views = [(f"{Q}/" + n.replace("/", "__") + f"_v{i}.png") for i in (1, 2, 3, 4) if os.path.exists((f"{Q}/" + n.replace("/", "__") + f"_v{i}.png"))]
        ims = [Image.open(v).convert("RGB") for v in views]
        W = sum(i.width for i in ims); H = ims[0].height + 40
        S = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(S)
        x = 0
        labels = ["3/4 above", "side (-y)", "top-in", "collision section"]
        for im, lab in zip(ims, labels):
            S.paste(im, (x, 40)); d.text((x + 6, 24), lab, fill=(0, 0, 0)); x += im.width
        bb = info["bbox_m"]
        t = f"{n}  seed {info['seed']}  bbox {bb[0]*100:.1f} x {bb[1]*100:.1f} x {bb[2]*100:.1f} cm"
        if "capacity_l" in info:
            t += f"   capacity {info['capacity_l']:.3f} L   wall {info['collision']['wall_m']*1000:.1f} mm  faces {info['collision']['faces']}  watertight {info['collision']['watertight']}"
        d.text((6, 6), t, fill=(0, 0, 0))
        S.save(f"{ROOT}/{n}/contact.png")
        print("wrote", f"{ROOT}/{n}/contact.png")
