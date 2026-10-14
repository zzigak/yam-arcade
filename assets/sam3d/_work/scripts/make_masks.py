"""Convert chosen source photos to PNG inputs and make object masks with rembg (+ optional box restriction).
usage: python -I make_masks.py NAME [NAME...]   (config in CFG below)"""
import json, os, sys
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi

SRC = os.path.expanduser("~/projects/yam-arcade/assets/sam3d/_src")
OUT = os.path.expanduser("~/projects/yam-arcade/assets/sam3d/_work/inputs")
# name: (source path, box (x0,y0,x1,y1) as fractions or None, rembg model, max side)
CFG = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs.json")))


def run(name):
    c = CFG[name]
    im = ImageOps.exif_transpose(Image.open(os.path.join(SRC, c["src"]))).convert("RGB")
    s = c.get("max_side", 1280) / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    W, H = im.size
    im.save(f"{OUT}/{name}.png")
    from rembg import remove, new_session
    sess = new_session(c.get("model", "isnet-general-use"))
    a = np.asarray(remove(im, session=sess, only_mask=True)).astype(np.float32) / 255
    m = a > c.get("thr", 0.5)
    if c.get("box"):
        x0, y0, x1, y1 = c["box"]
        bm = np.zeros_like(m); bm[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = True
        m &= bm
    for x0, y0, x1, y1 in c.get("exclude", []):
        m[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = False
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    holes = ndi.binary_fill_holes(m) & ~m          # fill only small holes (keep handle loops open)
    hl, hn = ndi.label(holes)
    if hn:
        hs = ndi.sum(holes, hl, range(1, hn + 1))
        m |= np.isin(hl, 1 + np.where(hs < 0.002 * m.sum())[0])
    m = ndi.binary_opening(m, iterations=c.get("open", 1))
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == (1 + int(np.argmax(ndi.sum(m, lab, range(1, n + 1)))))
    Image.fromarray((m * 255).astype(np.uint8)).save(f"{OUT}/{name}_mask.png")
    rgb = np.asarray(im).astype(np.float32)
    ov = rgb * 0.45 + np.array([255, 0, 0]) * 0.55 * ~m[..., None] + rgb * 0.55 * m[..., None]
    Image.fromarray(ov.clip(0, 255).astype(np.uint8)).save(f"{OUT}/{name}_overlay.jpg", quality=85)
    print(name, W, H, "mask frac %.3f" % m.mean())


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
