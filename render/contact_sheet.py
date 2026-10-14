"""Contact sheet of rendered frames with their video time and the station clocks.

    python3 render/contact_sheet.py <frames_dir> <out.jpg> [--frames f1,f2,...] [--cols 4] [--width 480]

Without --frames: every PNG in the directory. Needs Pillow (genesis-world venv).
"""
import argparse
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flypath  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("out")
    ap.add_argument("--frames", default=None)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--width", type=int, default=480)
    a = ap.parse_args()
    if a.frames:
        fs = [int(v) for v in a.frames.split(",")]
    else:
        fs = sorted(int(f[:5]) for f in os.listdir(a.frames_dir) if f.endswith(".png") and f[:5].isdigit())
    ims = [Image.open(os.path.join(a.frames_dir, f"{f:05d}.png")).convert("RGB") for f in fs]
    w = a.width
    h = round(ims[0].height * w / ims[0].width)
    cap = 34
    rows = (len(ims) + a.cols - 1) // a.cols
    sheet = Image.new("RGB", (a.cols * w, rows * (h + cap)), "white")
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(flypath.FONT, 13)
    for i, (f, im) in enumerate(zip(fs, ims)):
        x, y = (i % a.cols) * w, (i // a.cols) * (h + cap)
        sheet.paste(im.resize((w, h)), (x, y))
        t = f / flypath.FPS
        clocks = " ".join(f"{st[:2]}{flypath.sim_time(st, t):.0f}" for st in flypath.CLOCKS)
        d.text((x + 6, y + h + 2), f"frame {f}  t {t:.1f} s", fill=(30, 30, 30), font=font)
        d.text((x + 6, y + h + 17), f"sim s: {clocks}", fill=(110, 110, 110), font=font)
    sheet.save(a.out, quality=88)
    print("wrote", a.out, sheet.size)


if __name__ == "__main__":
    main()
