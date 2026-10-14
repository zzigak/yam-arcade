"""Box(+points)-prompted SAM v1 mask (transformers SamModel, CPU). Overwrites inputs/NAME_mask.png and overlay.
usage: python sam_box_mask.py NAME x0 y0 x1 y1 [--neg x,y ...] [--pos x,y ...]  (fractions of image size)"""
import sys, os
import numpy as np, torch
from PIL import Image
from scipy import ndimage as ndi
from transformers import SamModel, SamProcessor
IN = os.path.expanduser("~/projects/yam-arcade/assets/sam3d/_work/inputs")
a = sys.argv[1:]; name = a[0]; box = [float(v) for v in a[1:5]]
pts, lbl = [], []
for k in range(5, len(a)):
    if a[k] in ("--neg", "--pos"):
        x, y = map(float, a[k + 1].split(",")); pts.append([x, y]); lbl.append(1 if a[k] == "--pos" else 0)
im = Image.open(f"{IN}/{name}.png").convert("RGB"); W, H = im.size
proc = SamProcessor.from_pretrained("facebook/sam-vit-huge"); model = SamModel.from_pretrained("facebook/sam-vit-huge").eval()
kw = dict(input_boxes=[[[box[0] * W, box[1] * H, box[2] * W, box[3] * H]]])
if pts:
    kw["input_points"] = [[[[p[0] * W, p[1] * H] for p in pts]]]; kw["input_labels"] = [[lbl]]
inp = proc(im, return_tensors="pt", **kw)
with torch.no_grad():
    out = model(**inp, multimask_output=True)
masks = proc.image_processor.post_process_masks(out.pred_masks, inp["original_sizes"], inp["reshaped_input_sizes"])[0][0].numpy()
scores = out.iou_scores[0, 0].numpy(); print("scores", scores)
m = masks[int(np.argmax(scores))]
lab, n = ndi.label(m)
if n > 1:
    m = lab == (1 + int(np.argmax(ndi.sum(m, lab, range(1, n + 1)))))
m = ndi.binary_fill_holes(m)
Image.fromarray((m * 255).astype(np.uint8)).save(f"{IN}/{name}_mask.png")
rgb = np.asarray(im).astype(np.float32)
ov = rgb * 0.45 + np.array([255, 0, 0]) * 0.55 * ~m[..., None] + rgb * 0.55 * m[..., None]
Image.fromarray(ov.clip(0, 255).astype(np.uint8)).save(f"{IN}/{name}_overlay.jpg", quality=85)
print(name, "mask frac %.3f" % m.mean())
