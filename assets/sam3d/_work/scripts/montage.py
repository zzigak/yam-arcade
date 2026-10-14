"""Tile images into one labelled montage. usage: python -I montage.py out.jpg img1 img2 ... [--cell 400]"""
import sys
from PIL import Image, ImageDraw, ImageOps
a = sys.argv[1:]
cell = 400
if "--cell" in a:
    i = a.index("--cell"); cell = int(a[i+1]); del a[i:i+2]
out, files = a[0], a[1:]
cols = min(4, len(files)); rows = (len(files) + cols - 1) // cols
M = Image.new("RGB", (cols*cell, rows*cell), (255, 255, 255))
d = ImageDraw.Draw(M)
for k, f in enumerate(files):
    try:
        im = Image.open(f); im = ImageOps.exif_transpose(im).convert("RGB"); im.thumbnail((cell, cell-16))
    except Exception:
        continue
    x, y = (k % cols)*cell, (k//cols)*cell
    M.paste(im, (x, y+16)); d.text((x+3, y+2), f.split("/")[-2] + "/" + f.split("/")[-1], fill=(255, 0, 0))
M.save(out, quality=88)
