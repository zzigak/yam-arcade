"""Contact sheet of a clip's frames, each captioned with its video time (and sim time for clips).

    ~/simquarry/genesis-world/.venv/bin/python render/sheet.py <frames_dir> <out.jpg> [--clip st] [--frames ...]
        [--every N] [--cols 5] [--width 384] [--title text]
"""
import argparse
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FONT = "/usr/share/fonts/opentype/inter/Inter-Medium.otf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("out")
    ap.add_argument("--clip", default=None)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--every", type=int, default=0)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    have = sorted(int(f[:5]) for f in os.listdir(a.dir) if f.endswith(".png") and f[:5].isdigit())
    if a.frames:
        fs = [int(v) for v in a.frames.split(",")]
    elif a.every:
        fs = [f for f in have if f % a.every == 0] + ([have[-1]] if have[-1] % a.every else [])
    else:
        fs = have
    clip = None
    if a.clip:
        import clips
        clip = clips.Clip(a.clip)
    ims = [Image.open(os.path.join(a.dir, f"{f:05d}.png")).convert("RGB") for f in fs]
    w = a.width
    h = round(ims[0].height * w / ims[0].width)
    cap, top = 22, (34 if a.title else 0)
    rows = (len(ims) + a.cols - 1) // a.cols
    sheet = Image.new("RGB", (a.cols * w, top + rows * (h + cap)), "white")
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT, 13)
    if a.title:
        d.text((8, 8), a.title, fill=(20, 20, 20), font=ImageFont.truetype(FONT, 17))
    for i, (f, im) in enumerate(zip(fs, ims)):
        x, y = (i % a.cols) * w, top + (i // a.cols) * (h + cap)
        sheet.paste(im.resize((w, h)), (x, y))
        t = f / 30.0
        txt = f"frame {f}  t {t:.1f} s"
        if clip:
            txt += f"  sim {clip.sim_time(clip.station, t):.1f} s"
        d.text((x + 6, y + h + 3), txt, fill=(60, 60, 60), font=font)
    sheet.save(a.out, quality=88)
    print("wrote", a.out, sheet.size)


if __name__ == "__main__":
    main()
