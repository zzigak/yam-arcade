"""Per-station clips -> site/media/<station>_fly.mp4, and the 2x2 grid of them.

    python3 render/grid.py clip <station> <frames_dir> [--out site/media/<station>_fly.mp4]
    python3 render/grid.py grid [--suffix _fly] [--out site/media/grid.mp4] [--draft]

Grid: 2560x1440, tiles in the room's layout (mahjong | jars over cables | water), 12 px white gutters, a small label
per tile (station, task in a few words, "xN" when the clip is not real time). Clips shorter than the longest hold
their last frame. Also writes a 1920x1080 copy (<out>_1080.mp4, i.e. grid_1080.mp4). A missing clip becomes a grey
tile that says so.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import layout as L  # noqa: E402

MEDIA = os.path.join(ROOT, "site", "media")
ORDER = ["mahjong", "jars", "cables", "water"]          # top-left, top-right, bottom-left, bottom-right (as in the room)
FONT_B = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
FONT = "/usr/share/fonts/opentype/inter/Inter-Regular.otf"
W, H, G = 2560, 1440, 12
TW, TH = (W - G) // 2, (H - G) // 2                     # 1274 x 714


def run(cmd):
    subprocess.run(["nice", "-n", "10"] + cmd, check=True)


def duration(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                          "csv=p=0", p]))


def encode_clip(st, frames_dir, out=None, fps=30):
    out = out or os.path.join(MEDIA, f"{st}_fly.mp4")
    frames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png") and f[:5].isdigit())
    first = int(frames[0][:5])
    run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-start_number", str(first), "-i",
         os.path.join(frames_dir, "%05d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
         "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", out])
    print(f"encoded {out} ({len(frames)} frames)")


def esc_text(s):
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace("%", "\\%")


def grid(suffix="_fly", out=None, draft=False):
    import clips
    out = out or os.path.join(MEDIA, "grid.mp4")
    srcs = {st: os.path.join(MEDIA, f"{st}{suffix}.mp4") for st in ORDER}
    have = {st: p for st, p in srcs.items() if os.path.exists(p)}
    if not have:
        raise SystemExit("no clips")
    T = max(duration(p) for p in have.values())
    inputs, filt = [], []
    fs1, fs2 = 30, 22
    for i, st in enumerate(ORDER):
        if st in have:
            inputs += ["-i", have[st]]
            c = clips.Clip(st)
            title, task = c.label()
            src = f"[{len(inputs) // 2 - 1}:v]"
            pad = max(0.0, T - duration(have[st]))
            lab = (f"drawbox=x=18:y=ih-96:w={max(len(title), len(task)) * 13 + 40}:h=78:color=white@0.78:t=fill,"
                   f"drawtext=fontfile={FONT_B}:text='{esc_text(title)}':fontsize={fs1}:fontcolor=0x222220:x=34:y=h-88,"
                   f"drawtext=fontfile={FONT}:text='{esc_text(task)}':fontsize={fs2}:fontcolor=0x55534e:x=34:y=h-50")
            filt.append(f"{src}scale={TW}:-2,crop={TW}:{TH},tpad=stop_mode=clone:stop_duration={pad:.3f},"
                        f"fps=30,{lab}[t{i}]")
        else:
            filt.append(f"color=c=0xe4e4e4:s={TW}x{TH}:d={T:.3f}:r=30,drawtext=fontfile={FONT}:text="
                        f"'{esc_text(L.STATIONS[st]['title'])} — not rendered yet':fontsize=28:fontcolor=0x777777:"
                        f"x=(w-tw)/2:y=(h-th)/2[t{i}]")
    layout = f"0_0|{TW + G}_0|0_{TH + G}|{TW + G}_{TH + G}"
    filt.append(f"[t0][t1][t2][t3]xstack=inputs=4:layout={layout}:fill=white,trim=duration={T:.3f}[g]")
    fc = ";".join(filt)
    crf = "23" if draft else "18"
    run(["ffmpeg", "-y", "-loglevel", "error"] + inputs + ["-filter_complex", fc, "-map", "[g]", "-c:v", "libx264",
         "-crf", crf, "-preset", "slow" if not draft else "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-an", out])
    small = out.replace(".mp4", "_1080.mp4")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", out, "-vf", "scale=1920:1080:flags=lanczos", "-c:v", "libx264",
         "-crf", crf, "-preset", "slow" if not draft else "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-an", small])
    print(f"wrote {out} ({W}x{H}) and {small} (1920x1080), {T:.1f} s; tiles: "
          + ", ".join(f"{st}{'' if st in have else ' (missing)'}" for st in ORDER))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("clip", "grid"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--suffix", default="_fly")
    ap.add_argument("--draft", action="store_true")
    a = ap.parse_args()
    if a.cmd == "clip":
        encode_clip(a.args[0], a.args[1], out=a.out)
    else:
        grid(suffix=a.suffix, out=a.out, draft=a.draft)
