"""Build site/index.html for the YAM arcade from what is on disk. Served by Caddy at /arcade/.

    python3 site/build_site.py              # the four stations
    python3 site/build_site.py --with-test  # also a section for the synthetic compositor test (out/_test)

Styled like ~/projects/franka-deformables/site (one narrow Palatino column, grey tab rows, mono chips, dotted
leaders). Every number on the page is read from files: layout.py, out/<station>/{meta.json, checks.json, DONE},
stations/<station>/NOTES.md, and render/frames/<camera>/render_stats.json written by render/compose.py.
Videos are expected at site/media/<camera>.mp4 (compose.py --encode); missing ones show as "not rendered yet".
"""
import argparse
import html
import json
import os
import re
import shutil
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import layout as L  # noqa: E402

OUT = os.path.join(ROOT, "out")
MEDIA = os.path.join(HERE, "media")
SRC = os.path.join(HERE, "src")
FRAMES = os.path.join(ROOT, "render", "frames")
# A station's result directory, when it is not out/<station> (cables: the Newton run is the result; out/cables itself
# holds the Genesis comparison). Same mapping the fly-through snapshots use (render/snapshot.py).
RESULT_DIRS = {"cables": os.path.join(OUT, "cables", "newton")}


# Other versions shown only as links: the Genesis + libuipc cables run (one cable 52 % in) next to the Newton result.
COMPARISON_SRC = {"cables": [(os.path.join(OUT, "cables", "preview.mp4"), "cables_genesis_preview.mp4",
                              "Genesis comparison: preview"),
                             (os.path.join(OUT, "cables", "demo_attempt", "preview.mp4"),
                              "cables_genesis_demo_attempt.mp4", "Genesis comparison: demo attempt")]}
COMPARISONS = {s: [(dst, lab) for _, dst, lab in v] for s, v in COMPARISON_SRC.items()}


def result_dir(s):
    return RESULT_DIRS.get(s, os.path.join(OUT, s))


ARM_NAMES = {"yam": "YAM", "yam_pro": "YAM Pro", "yam_ultra": "YAM Ultra", "big_yam": "Big YAM"}
NUMWORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def esc(s):
    return html.escape(str(s), quote=True)


def words(n):
    return NUMWORDS[n] if 0 <= n < len(NUMWORDS) else str(n)


def fmt(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v != v:
            return "nan"
        if v == 0:
            return "0"
        a = abs(v)
        return f"{v:.3g}" if (a < 1e-3 or a >= 1e5) else f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return ", ".join(f"{k} {fmt(x)}" for k, x in v.items())
    return str(v)


def first(d, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return k, d[k]
    return None, None


def as_ok(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("pass", "passed", "ok", "true", "yes", "good"):
            return True
        if s in ("fail", "failed", "false", "no", "bad"):
            return False
    return None


# ------------------------------------------------------------------------------------------- readers
def load_checks(path):
    """Normalise a station's checks.json into rows; tolerant of the shapes a station is likely to write:
    {name: {value, limit, pass}}, [{name, value, limit, pass}], or {"checks": ...}, with synonyms."""
    if not os.path.exists(path):
        return None, None
    d = json.load(open(path))
    overall = None
    if isinstance(d, dict):
        _, overall = first(d, "all_pass", "all_passed", "pass", "passed", "ok")
        overall = as_ok(overall) if not isinstance(overall, dict) else None
        for k in ("checks", "results", "items"):
            if isinstance(d.get(k), (dict, list)):
                d = d[k]
                break
    if isinstance(d, dict):
        items = [(k, v) for k, v in d.items() if k not in ("all_pass", "all_passed", "pass", "passed", "ok")]
    elif isinstance(d, list):
        items = [((c.get("name") or c.get("check") or c.get("id") or f"check {i + 1}") if isinstance(c, dict) else
                  f"check {i + 1}", c) for i, c in enumerate(d)]
    else:
        return [], overall
    rows = []
    for name, c in items:
        if not isinstance(c, dict):
            c = {"value": c}
        _, value = first(c, "value", "measured", "actual", "val", "result_value")
        lk, limit = first(c, "limit", "threshold", "bound", "max", "min", "target", "tolerance", "range")
        _, ok = first(c, "pass", "passed", "ok", "status", "result")
        unit = c.get("unit") or c.get("units") or ""
        cmp = c.get("cmp") or c.get("op") or c.get("comparison") or ({"max": "≤", "min": "≥"}.get(lk, ""))
        note = c.get("note") or c.get("description") or c.get("desc") or c.get("why") or c.get("detail") or ""
        rows.append(dict(name=str(name), value=value, limit=limit, ok=as_ok(ok), unit=unit, cmp=cmp, note=note))
    if overall is None and rows and all(r["ok"] is not None for r in rows):
        overall = all(r["ok"] for r in rows)
    return rows, overall


def checks_table(rows):
    if not rows:
        return '<p class="au">No checks recorded yet.</p>'
    trs = []
    for r in rows:
        lim = (f"{esc(r['cmp'])} " if r["cmp"] else "") + (esc(fmt(r["limit"])) if r["limit"] is not None else "")
        unit = f" {esc(r['unit'])}" if r["unit"] else ""
        verdict = {True: "pass", False: "<b>FAIL</b>", None: "–"}[r["ok"]]
        note = f'<br><span class="au">{esc(r["note"])}</span>' if r["note"] else ""
        trs.append(f"<tr><td>{esc(r['name'])}{note}</td>"
                   f"<td class=\"num\">{esc(fmt(r['value'])) if r['value'] is not None else '–'}{unit}</td>"
                   f"<td>{lim}{unit if r['limit'] is not None else ''}</td><td>{verdict}</td></tr>")
    return ('<div class="tablewrap"><table class="keys wide"><thead><tr><th>check</th><th class="num">value</th>'
            '<th>limit</th><th>verdict</th></tr></thead><tbody>' + "".join(trs) + "</tbody></table></div>")


def inline_md(s):
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"<i>\1</i>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r"\1", s)  # relative links point into the repo, not the site
    return s


def md(text, hmin=3):
    """Small Markdown subset: headings, paragraphs, bullet/numbered lists, pipe tables, fenced code, inline marks."""
    out, para, lst, i = [], [], None, 0
    lines = text.splitlines()

    def flush():
        nonlocal para, lst
        if para:
            out.append("<p>" + inline_md(" ".join(para)) + "</p>")
            para = []
        if lst:
            tag, items = lst
            out.append(f"<{tag}>" + "".join(f"<li>{inline_md(x)}</li>" for x in items) + f"</{tag}>")
            lst = None
    while i < len(lines):
        ln = lines[i]
        st = ln.strip()
        if st.startswith("```"):
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            out.append("<pre><code>" + esc("\n".join(lines[i + 1:j])) + "</code></pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", st)
        if m:
            flush()
            lvl = min(6, len(m.group(1)) - 1 + hmin)
            out.append(f"<h{lvl}>{inline_md(m.group(2))}</h{lvl}>")
            i += 1
            continue
        if st.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?\s*:?-{2,}", lines[i + 1].strip()):
            flush()
            cells = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]  # noqa: E731
            head = cells(st)
            j, body = i + 2, []
            while j < len(lines) and lines[j].strip().startswith("|"):
                body.append(cells(lines[j]))
                j += 1
            out.append('<div class="tablewrap"><table class="keys wide"><thead><tr>' +
                       "".join(f"<th>{inline_md(h)}</th>" for h in head) + "</tr></thead><tbody>" +
                       "".join("<tr>" + "".join(f"<td>{inline_md(c)}</td>" for c in r) + "</tr>" for r in body) +
                       "</tbody></table></div>")
            i = j
            continue
        m = re.match(r"^\s*([-*+]|\d+[.)])\s+(.*)", ln)
        if m:
            if para:
                flush()
            tag = "ol" if m.group(1)[0].isdigit() else "ul"
            if lst is None or lst[0] != tag:
                flush()
                lst = (tag, [])
            lst[1].append(m.group(2))
            i += 1
            continue
        if not st:
            flush()
            i += 1
            continue
        if lst and ln.startswith((" ", "\t")):  # continuation of a list item
            lst[1][-1] += " " + st
        else:
            if lst:
                flush()
            para.append(st)
        i += 1
    flush()
    return "\n".join(out)


def notes_parts(path):
    """(highlight html, full html). Highlight: a section titled like highlights/summary/result, else the first
    paragraph after the title."""
    if not os.path.exists(path):
        return None, None
    text = open(path, encoding="utf-8", errors="replace").read()
    full = md(text, hmin=4)
    secs = re.split(r"(?m)^(#{1,6}\s+.*)$", text)
    for k in range(1, len(secs) - 1, 2):
        if re.search(r"highlight|summary|result|tl;?dr|status", secs[k], re.I):
            return md(secs[k + 1].strip(), hmin=4), full
    body = re.sub(r"(?m)^#{1,6}\s+.*$", "", text).strip()
    para = body.split("\n\n")[0] if body else ""
    return (md(para, hmin=4) if para else None), full


def render_stats(cam):
    p = os.path.join(FRAMES, cam, "render_stats.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p))
    t = list(s.get("frames", {}).values())
    s["n"] = len(t)
    s["median_s"] = statistics.median(t) if t else None
    s["total_min"] = sum(t) / 60 if t else None
    return s


def poster(mp4, jpg, frac=0.5):
    if os.path.exists(jpg) and os.path.getmtime(jpg) > os.path.getmtime(mp4):
        return
    try:
        d = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                           "csv=p=0", mp4]))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{d * frac:.2f}", "-i", mp4, "-frames:v", "1",
                        "-q:v", "3", jpg], check=True)
    except Exception as e:  # a broken clip should not stop the page
        print("poster failed", mp4, e)


def video_block(cam, title, chips, caption=""):
    mp4 = os.path.join(MEDIA, f"{cam}.mp4")
    tags = "".join(f'<span class="tag {c}">{esc(t)}</span>' for t, c in chips)
    head = (f'<div class="top"><span class="t">{esc(title)}</span><span class="lead"></span>'
            f'<span class="tags">{tags}</span></div>')
    if os.path.exists(mp4):
        jpg = os.path.join(MEDIA, f"{cam}.jpg")
        poster(mp4, jpg)
        v = (f'<video src="media/{cam}.mp4" poster="media/{cam}.jpg" autoplay muted loop playsinline controls '
             f'preload="metadata"></video>')
    else:
        v = '<div class="novideo">not rendered yet</div>'
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure class="clip">{head}{v}{cap}</figure>'


def stats_line(st):
    if not st or not st.get("n"):
        return ""
    return (f"Cycles {esc(st.get('device', ''))}, {esc(st.get('res', ''))}, {fmt(st.get('samples'))} samples, "
            f"{fmt(st['n'])} frames at {fmt(st['median_s'])} s each (median), {fmt(st['total_min'])} min in all.")


FLY_FRAMES = os.path.join(FRAMES, "fly_final")


def fly_stats(frames_dir=FLY_FRAMES):
    """Merge the per-chunk render_stats_*.json of a chunked render (fly-through or a station clip)."""
    import glob
    files = sorted(glob.glob(os.path.join(frames_dir, "render_stats*.json")))
    if not files:
        return None
    out, frames, hosts, gpus = None, {}, set(), set()
    for f in files:
        d = json.load(open(f))
        out = out or d
        frames.update(d.get("frames", {}))
        hosts.add(d.get("host", "?"))
        gpus.add(d.get("gpu", "?"))
    t = list(frames.values())
    out = dict(out, frames=frames, n=len(t), median_s=statistics.median(t) if t else None,
               total_min=sum(t) / 60 if t else None, hosts=sorted(hosts), gpus=sorted(gpus), chunks=len(files))
    return out


def clip_info(s):
    """Caption facts for a station's clip, from render/clips.py and the clip render's stats."""
    sys.path.insert(0, os.path.join(ROOT, "render"))
    try:
        import clips
        c = clips.Clip(s)
    except Exception:
        return None, ""
    st = fly_stats(os.path.join(FRAMES, f"fly_{s}_final"))
    speed = "real time" if abs(c.speed - 1) < 1e-6 else f"×{fmt(c.speed)}"
    cap = (f"One camera move around the table, {fmt(c.DURATION)} s, {speed} (sim {fmt(c.sim_0)}–{fmt(c.sim_T)} s). "
           f"The station is rendered alone at the room centre.")
    if st:
        src = st.get("sources", {}).get(s, {})
        cap += (f" Blender {st.get('blender', '')} Cycles on {', '.join(st['gpus'])} ({fmt(st['chunks'])} chunks): "
                f"{st.get('res', '')}, {fmt(st.get('samples'))} samples, {fmt(st['n'])} frames at {fmt(st['median_s'])} s "
                f"each (median). Export as rendered: {src.get('source', '')}, written {src.get('source_meta_mtime_iso', '')}.")
    return c, esc(cap)


def clips_how():
    """'How it was built' item for the per-station clips and the grid, numbers from the clip render stats."""
    rows = []
    for s in L.STATIONS:
        st = fly_stats(os.path.join(FRAMES, f"fly_{s}_final"))
        if st:
            rows.append((s, st))
    if not rows:
        return ""
    gpus = sorted({g for _, st in rows for g in st["gpus"]})
    n = sum(st["n"] for _, st in rows)
    gpu_min = sum(st["total_min"] for _, st in rows)
    chunks = sum(st["chunks"] for _, st in rows)
    res = sorted({st.get("res", "") for _, st in rows})
    spp = sorted({fmt(st.get("samples")) for _, st in rows})
    return f"""
        <li class="p"><div class="top"><span class="t">Clips and grid</span><span class="lead"></span><span class="tags">
          <span class="tag">clips.py</span><span class="tag">sc cluster</span><span class="tag">ffmpeg xstack</span></span></div>
          <p class="au">Each clip renders one station alone at the room centre, so every tile has the same light. The camera
          follows one eased spline per station (<a href="src/clips.py">clips.py</a>), keeping clear of the cart's mast.
          Each clip was drafted at low resolution, checked frame by frame, then rendered at {esc(', '.join(res))} with
          {esc(', '.join(spp))} samples: {fmt(n)} frames in {fmt(chunks)} resumable chunks on the Stanford sc cluster
          ({esc(', '.join(gpus))}), {fmt(gpu_min)} GPU-min in all, from frozen copies of each station's export.
          <a href="src/grid.py">grid.py</a> stacks the four clips with ffmpeg.</p></li>"""


def grid_block():
    if not os.path.exists(os.path.join(MEDIA, "grid_1080.mp4")):
        return ""
    sys.path.insert(0, os.path.join(ROOT, "render"))
    import clips
    parts = []
    for s in L.STATIONS:
        try:
            c = clips.Clip(s)
        except Exception:
            continue
        sp = "" if abs(c.speed - 1) < 1e-6 else f" at ×{fmt(c.speed)}"
        parts.append(f"{L.STATIONS[s]['title']} {fmt(c.DURATION)} s{sp}")
    cap = (f"{words(len(parts)).capitalize()} clips, one per table, each a single camera move around its own table. "
           "The simulations play in real time unless a tile says ×N; shorter clips hold their last frame. "
           + esc("; ".join(parts)) + '. <a href="media/grid.mp4">[2560×1440]</a>')
    tiles = []
    for st in L.STATIONS:
        if not os.path.exists(os.path.join(MEDIA, f"{st}_fly.mp4")):
            continue
        c, _ = clip_info(st)
        chip = [(f"{fmt(c.DURATION)} s", ""), (("×" + fmt(c.speed)) if abs(c.speed - 1) > 1e-6 else "1×", "")] if c else []
        tiles.append(video_block(f"{st}_fly", L.STATIONS[st]["title"], chip,
                                 f'<a href="#{st}">checks and notes</a>'))
    quad = f'<div class="quad">{"".join(tiles)}</div>' if tiles else ""
    return f"""
      {video_block("grid_1080", "All four stations", [], cap)}
      {quad}"""


def flythrough_block():
    mp4 = os.path.join(MEDIA, "flythrough.mp4")
    if not os.path.exists(mp4):
        return ""
    sys.path.insert(0, os.path.join(ROOT, "render"))
    import flypath
    st = fly_stats()
    clocks = []
    for s, pieces in flypath.CLOCKS.items():
        title = L.STATIONS[s]["title"]
        parts = [f"sim {fmt(a)}–{fmt(b)} s at ×{fmt(k)} from {fmt(v0)} s" for v0, k, a, b in pieces]
        clocks.append(f"{title}: " + "; ".join(parts))
    cap = (f"One continuous camera move, {fmt(flypath.DURATION)} s. Each station keeps its own simulation clock; "
           "where it is sped up the frame says so (×N), and outside its clip a station holds its first or last frame. "
           + esc(". ".join(clocks)) + ".")
    if st:
        src = "; ".join(f"{s} {v.get('source', '')} ({v.get('source_meta_mtime_iso', '')})"
                        for s, v in sorted(st.get("sources", {}).items()))
        cap += (f" Rendered with Blender {esc(st.get('blender', ''))} Cycles on {esc(', '.join(st['gpus']))} "
                f"({fmt(st['chunks'])} chunks), {esc(st.get('res', ''))}, {fmt(st.get('samples'))} samples: "
                f"{fmt(st['n'])} frames at {fmt(st['median_s'])} s each (median), {fmt(st['total_min'])} GPU-min. "
                f"Exports as rendered: {esc(src)}.")
    return f"""
      {video_block("flythrough", "Fly-through", [(f"{fmt(flypath.DURATION)} s", "")], cap)}"""


# ------------------------------------------------------------------------------------------- page
def station_section(s, info, st_dir, notes_dir):
    meta_p = os.path.join(st_dir, "meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else None
    done = os.path.exists(os.path.join(st_dir, "DONE"))
    rows, overall = load_checks(os.path.join(st_dir, "checks.json"))
    hi, full = notes_parts(os.path.join(notes_dir, "NOTES.md"))
    if hi is None:
        hi, full = notes_parts(os.path.join(st_dir, "NOTES.md"))
    arm = ARM_NAMES.get(info.get("arm"), info.get("arm", ""))
    small = [arm] if arm else []
    if meta:
        n, fps = meta.get("n_frames", 0), meta.get("fps", L.RENDER_FPS)
        small.append(f"{fmt(n)} frames · {fmt(n / fps if fps else 0)} s")
    chips = [("done", "ok") if done else ("in progress", "wip")]
    if rows:
        npass = sum(1 for r in rows if r["ok"])
        chips.append((f"{npass}/{len(rows)} checks pass", "" if npass == len(rows) else "fail"))
    if meta:
        kinds = {}
        for k, o in meta["objects"].items():
            if not k.startswith("robot/"):
                kinds[o["kind"]] = kinds.get(o["kind"], 0) + 1
        robot_links = sum(1 for k in meta["objects"] if k.startswith("robot/"))
        objs = sum(1 for k in meta["objects"] if not k.startswith("robot/"))
        parts = [f"{fmt(v)} {k}" for k, v in sorted(kinds.items())]
        kind_line = (f"Export: {fmt(objs)} objects ({', '.join(parts)})" if parts else "Export: no objects") + \
            (f", plus the robot ({fmt(robot_links)} link meshes)" if robot_links > 1 else
             (", plus one robot link mesh" if robot_links == 1 else "")) + "."
        extra = {k: v for k, v in meta.items() if k not in ("station", "fps", "n_frames", "objects", "camera")
                 and isinstance(v, (int, float, str, bool)) and len(str(v)) < 200}
        if extra:
            kind_line += " " + " · ".join(f"{k}: {fmt(v)}" for k, v in extra.items())
    else:
        kind_line = "No export yet."
    st_render = render_stats(s)
    cap = esc(kind_line) + (" " + stats_line(st_render) if st_render else "")
    vid, vtitle = s, "Close-up"
    if os.path.exists(os.path.join(MEDIA, f"{s}_fly.mp4")):
        _, ccap = clip_info(s)
        vid, vtitle, cap = f"{s}_fly", "Around the table", ccap + " " + esc(kind_line)
    over = os.path.exists(os.path.join(MEDIA, f"{s}_over.mp4"))
    links = []
    if over:
        links.append(f'<a href="media/{s}_over.mp4">[over-the-shoulder view]</a>')
    if os.path.exists(os.path.join(SRC, s, "NOTES.md")):
        links.append(f'<a href="src/{s}/NOTES.md">[NOTES.md]</a>')
    if os.path.exists(os.path.join(SRC, s, "checks.json")):
        links.append(f'<a href="src/{s}/checks.json">[checks.json]</a>')
    if os.path.exists(os.path.join(SRC, s, "station.py")):
        links.append(f'<a href="src/{s}/station.py">[station.py]</a>')
    for f, lab in COMPARISONS.get(s, []):
        if os.path.exists(os.path.join(MEDIA, f)):
            links.append(f'<a href="media/{f}">[{esc(lab)}]</a>')
    title = info.get("title", s)
    task = info.get("task", "")
    return f"""
    <section class="task" id="{esc(s)}" data-task="{esc(s)}">
      <h2>{esc(title)} <small>{esc(' · '.join(small))}</small></h2>
      <p class="desc">{esc(task[:1].upper() + task[1:])}.</p>
      {('<div class="notes">' + hi + '</div>') if hi else ''}
      {video_block(vid, vtitle, chips, cap)}
      <h3>Checks{'' if overall is None else (' <small>all pass</small>' if overall else ' <small class="bad">not all pass</small>')}</h3>
      {checks_table(rows) if rows is not None else '<p class="au">checks.json not written yet.</p>'}
      {('<details><summary>NOTES.md</summary><div class="notes full">' + full + '</div></details>') if full else ''}
      <p class="links">{' '.join(links)}</p>
    </section>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-test", action="store_true", help="add the synthetic compositor test (out/_test)")
    args = ap.parse_args()
    os.makedirs(MEDIA, exist_ok=True)
    os.makedirs(SRC, exist_ok=True)
    for v in COMPARISON_SRC.values():
        for src, dst, _ in v:
            if os.path.exists(src):
                shutil.copy(src, os.path.join(MEDIA, dst))
    stations = dict(L.STATIONS)
    if args.with_test:
        stations["_test"] = dict(arm="", title="Compositor test", task="synthetic export exercising every kind and "
                                 "material; nothing here is simulated")

    # copies of sources, served as text
    for f in ("render/compose.py", "render/make_test_export.py", "render/flypath.py", "render/clips.py",
              "render/grid.py", "render/sc_render.sh",
              "render/snapshot.py", "site/build_site.py", "export.py", "layout.py", "SPEC.md", "robots/README.md"):
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            dst = os.path.join(SRC, f.replace("/", "_") if f.startswith("robots/") else os.path.basename(f))
            shutil.copy(p, dst)
    for s in stations:
        for f, base in (("NOTES.md", os.path.join(ROOT, "stations", s)), ("station.py", os.path.join(ROOT, "stations", s)),
                        ("checks.json", result_dir(s)), ("NOTES.md", result_dir(s))):
            p = os.path.join(base, f)
            if os.path.exists(p):
                os.makedirs(os.path.join(SRC, s), exist_ok=True)
                shutil.copy(p, os.path.join(SRC, s, f))

    sections = "".join(station_section(s, info, result_dir(s), os.path.join(ROOT, "stations", s))
                       for s, info in stations.items())
    n_done = sum(os.path.exists(os.path.join(result_dir(s), "DONE")) for s in L.STATIONS)
    ov_stats = render_stats("overview")
    overview = (video_block("overview", "Overview", [(f"{n_done}/{len(L.STATIONS)} stations done",
                                                      "ok" if n_done == len(L.STATIONS) else "wip")],
                            stats_line(ov_stats)) if os.path.exists(os.path.join(MEDIA, "overview.mp4")) else "")
    fly = grid_block() + flythrough_block()

    arms = ", ".join(f"{ARM_NAMES.get(v['arm'], v['arm'])} at {v['title'].lower()}" for v in L.STATIONS.values())
    T = L.TABLE
    tabs = "".join(f'<span class="sep">|</span><button type="button" data-v="{esc(s)}" aria-pressed="false">'
                   f'{esc(v.get("title", s))}</button>' for s, v in stations.items())
    kinds_used = sorted({o["material"].get("kind", "?") for s in stations
                         if os.path.exists(os.path.join(result_dir(s), "meta.json"))
                         for o in json.load(open(os.path.join(result_dir(s), "meta.json")))["objects"].values()
                         if isinstance(o.get("material"), dict)})
    how = f"""
      <ol class="papers">
        <li class="p"><div class="top"><span class="t">Layout</span><span class="lead"></span><span class="tags">
          <span class="tag">layout.py</span></span></div><p class="au">One file fixes the room: {words(len(L.STATIONS))}
          tables of {fmt(T['size_x'])} × {fmt(T['size_y'])} m with the top at {fmt(T['height'])} m, each station in its
          own frame (origin on the floor under the table centre, z up) and placed in the room by a world pose. The cart
          parks on the near side; its arm bases sit {fmt(L.ROBOT_MOUNT['arm_spacing'])} m apart, level with the top.</p></li>
        <li class="p"><div class="top"><span class="t">Simulation, one station at a time</span><span class="lead"></span>
          <span class="tags"><span class="tag">Genesis</span><span class="tag">libuipc</span><span class="tag">SPH</span></span></div>
          <p class="au">Each station runs in its own Genesis scene with its own table top as a fixed body: the rigid solver
          for tiles, jars and the cart, libuipc (IPC) through Genesis's coupler for rods and screw threads, SPH for water.
          The robot is the same wheeled bimanual cart everywhere ({esc(arms)}), driven only through PD joint targets.
          Each station writes its own numeric checks; they are the tables above.</p></li>
        <li class="p"><div class="top"><span class="t">Export</span><span class="lead"></span><span class="tags">
          <span class="tag">export.py</span></span></div><p class="au">A station records every frame at
          {fmt(L.RENDER_FPS)} fps in its own frame: rigid meshes with poses, rods as polylines with a radius, particles,
          and per-frame meshes, each with a material kind. The robot's links are rigids too. Kinds in use:
          {esc(', '.join(kinds_used) if kinds_used else 'none yet')}.</p></li>
        <li class="p"><div class="top"><span class="t">Compositing and rendering</span><span class="lead"></span>
          <span class="tags"><span class="tag">Blender {esc((ov_stats or {}).get('blender', ''))}</span><span class="tag">Cycles</span></span></div>
          <p class="au"><a href="src/compose.py">compose.py</a> builds the studio (a seamless light-grey cove, a large
          overhead soft box, a key and a fill, AgX view), draws the furniture from layout.py, and places every export
          at its station's pose. Rods become smooth tubes rebuilt every frame, particles are rendered as spheres, and
          meshes are swapped per frame. Dark panels that only reflections can see give glass and water their edges on
          the white background. Nothing is re-simulated or edited at this stage.
          {stats_line(ov_stats)}</p></li>{clips_how()}
      </ol>"""
    test_clip = ""
    if os.path.exists(os.path.join(MEDIA, "_test.mp4")):
        test_clip = ("<details><summary>compositor test scene (synthetic, not simulated)</summary>" +
                     video_block("_test", "Material and kind test", [("synthetic", "wip")],
                                 "Every material kind on a shelf, a sliding cube, swaying rods, falling beads and drops, "
                                 "and per-frame water meshes. Used to tune the look; not part of any result.") +
                     "</details>")
    src_links = " ".join(f'<a href="src/{f}">[{f}]</a>' for f in
                         ("compose.py", "clips.py", "grid.py", "sc_render.sh", "build_site.py", "export.py", "layout.py",
                          "SPEC.md", "robots_README.md", "make_test_export.py") if os.path.exists(os.path.join(SRC, f)))
    page = TEMPLATE.format(
        n_tables=words(len(L.STATIONS)).capitalize(), sections=sections, overview=fly + overview, how=how, tabs=tabs,
        test_clip=test_clip, src_links=src_links, n_done=n_done, n_st=len(L.STATIONS),
        tasks="; ".join(f"{esc(v['title'].lower())}: {esc(v['task'])}" for v in L.STATIONS.values()))
    open(os.path.join(HERE, "index.html"), "w").write(page)
    print(f"wrote {HERE}/index.html ({n_done}/{len(L.STATIONS)} stations done)")


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>YAM Arcade</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Same approach as /franka/: one narrow reading column, grey tab rows, mono chips, dotted leaders. */
:root{{
  --bg:#ffffff; --fg:#191918; --muted:#6b6862; --faint:#b9b5ac; --rule:#dcd8cf;
  --link:#3f55b5; --link-h:#2b3d8c; --ok:#2f7d4f; --wip:#a16207; --bad:#c2410c; --chip:#efece5;
  --serif:Palatino,"Palatino Linotype","Palatino LT STD","Book Antiqua",Georgia,serif;
  --mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{
  --bg:#18181a; --fg:#ebe8e1; --muted:#a39f97; --faint:#6a675f; --rule:#34332f;
  --link:#9fb0ee; --link-h:#c4cff5; --ok:#7cc79a; --wip:#e0b04f; --bad:#f5934f; --chip:#242427; color-scheme:dark}}}}
:root[data-theme="dark"]{{
  --bg:#18181a; --fg:#ebe8e1; --muted:#a39f97; --faint:#6a675f; --rule:#34332f;
  --link:#9fb0ee; --link-h:#c4cff5; --ok:#7cc79a; --wip:#e0b04f; --bad:#f5934f; --chip:#242427; color-scheme:dark}}

*{{box-sizing:border-box}}
html{{font-size:16.7px;-webkit-text-size-adjust:100%}}
body{{margin:0;background:var(--bg);color:var(--fg);font-family:var(--serif);font-size:1rem;line-height:1.5;
  padding-inline:1.25rem;padding-block:2.5rem 5rem}}
.page{{max-width:54rem;margin:0 auto}}
a{{color:var(--link);text-decoration:underline dotted 1px;text-underline-offset:3px}}
a:hover{{color:var(--link-h)}}
:focus-visible{{outline:2px solid var(--link);outline-offset:2px;border-radius:2px}}
code,.mono{{font-family:var(--mono);font-size:.82em}}
h1{{margin:0;font-size:2.2rem;font-weight:600;line-height:1.2;text-wrap:balance}}
.lede{{margin:.9rem 0 0;font-weight:300;max-width:42rem}}
.controls{{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:var(--bg);
  margin:2rem -1.25rem 0;padding:.9rem 1.25rem .8rem;border-bottom:1px solid var(--rule);display:grid;gap:.55rem}}
.tabs{{display:flex;align-items:baseline;flex-wrap:wrap;gap:.25rem .55rem;font-size:1.45rem;font-weight:500;line-height:1.25}}
.tabs button{{font:inherit;background:none;border:0;padding:0;color:var(--faint);cursor:pointer}}
.tabs button[aria-pressed="true"]{{color:var(--fg);cursor:default}}
.tabs button:not([aria-pressed="true"]):hover{{color:var(--muted)}}
.tabs .sep{{color:var(--rule);font-weight:300}}
.count{{font-family:var(--mono);font-size:.72rem;color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}}
.task,.ov{{margin-top:2.4rem}}
h2{{display:flex;align-items:baseline;flex-wrap:wrap;gap:.2rem .7rem;margin:0 0 .6rem;font-size:1.7rem;font-weight:500;line-height:1.2}}
h2 small,h3 small{{font-family:var(--mono);font-size:.72rem;font-weight:400;color:var(--muted)}}
h3{{margin:1.3rem 0 .2rem;font-size:1.1rem;font-weight:500}}
h3 small.bad{{color:var(--bad)}}
.desc{{margin:0 0 .8rem;font-weight:300;max-width:46rem}}
.notes{{font-weight:300;max-width:46rem;font-size:.95rem}}
.notes p{{margin:.2rem 0 .7rem}}
.notes ul,.notes ol{{margin:.2rem 0 .7rem;padding-left:1.2rem}}
.notes h4,.notes h5,.notes h6{{margin:1rem 0 .3rem;font-size:1rem;font-weight:500}}
.notes pre{{overflow-x:auto;background:var(--chip);padding:.6rem .8rem;border-radius:2px;font-size:.8rem}}
.notes.full{{font-size:.9rem}}
.clip{{margin:0;min-width:0}}
.quad{{display:grid;grid-template-columns:1fr 1fr;gap:1rem 1.1rem;margin-top:1.4rem}}
.clip video,.novideo{{display:block;width:100%;aspect-ratio:16/9;margin-top:.35rem;background:var(--chip);border-radius:2px}}
.novideo{{display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:.8rem;color:var(--muted)}}
.clip figcaption{{margin:.35rem 0 0;font-size:.88rem;font-weight:300;color:var(--muted)}}
.top{{display:flex;align-items:baseline;gap:.5rem;min-width:0}}
.t{{font-size:1.02rem;font-weight:500;color:var(--fg);min-width:0;overflow-wrap:anywhere}}
.lead{{flex:1 1 2rem;min-width:1.5rem;border-bottom:1px dotted var(--faint);transform:translateY(-.3em)}}
.tags{{display:flex;gap:.3rem;flex:0 0 auto;flex-wrap:wrap}}
.tag{{font-family:var(--mono);font-size:.66rem;line-height:1;padding:.22rem .32rem;border-radius:2px;
  background:var(--chip);color:var(--muted);white-space:nowrap;letter-spacing:.02em}}
.tag.ok{{color:var(--ok)}} .tag.wip{{color:var(--wip)}} .tag.fail{{color:var(--bad);font-weight:500}}
.tablewrap{{overflow-x:auto}}
table.keys{{margin:.6rem 0 .3rem;border-collapse:collapse;font-family:var(--mono);font-size:.74rem;font-variant-numeric:tabular-nums}}
table.keys.wide{{width:100%}}
table.keys th{{text-align:left;font-weight:400;color:var(--muted);padding:.15rem 1.4rem .25rem 0;border-bottom:1px solid var(--rule)}}
table.keys td{{padding:.12rem 1.4rem .12rem 0;border-bottom:1px dotted var(--rule);vertical-align:top}}
table.keys td.num,table.keys th.num{{text-align:right}}
table.keys td .au{{font-family:var(--serif);font-size:.8rem}}
.links{{margin:.7rem 0 0;font-size:.85rem;display:flex;flex-wrap:wrap;gap:0 .7rem}}
.links a{{color:var(--fg)}} .links a:hover{{color:var(--link)}}
details{{margin-top:.6rem}}
summary{{font-family:var(--mono);font-size:.74rem;color:var(--muted);cursor:pointer;width:max-content}}
summary:hover{{color:var(--link)}}
.part{{margin-top:3.2rem}}
ol.papers{{list-style:none;margin:0;padding:0;display:grid;gap:.85rem}}
.p{{min-width:0}}
.au{{margin:.12rem 0 0;font-size:.88rem;font-weight:300;color:var(--muted)}}
footer{{margin-top:4rem;font-size:.82rem;font-weight:300;color:var(--muted);max-width:42rem}}
@media (max-width:47rem){{
  html{{font-size:15.7px}}
  body{{padding-block:1.5rem 3rem;padding-inline:1.1rem}}
  h1{{font-size:1.6rem}}
  .controls{{margin-inline:-1.1rem;padding-inline:1.1rem}}
  .tabs{{font-size:1.2rem}}
  .lead{{display:none}}
  .quad{{grid-template-columns:1fr}}
  .top{{flex-wrap:wrap}}
  table.keys th,table.keys td{{padding-right:.8rem}}
}}
</style>
</head>
<body>
<div class="page">
  <header>
    <h1>YAM Arcade</h1>
    <p class="links" style="margin:.5rem 0 0"><a href="viewer/">[3D viewer: orbit and scrub every simulation run, live from out/]</a></p>
    <p class="lede">{n_tables} tables in one white room, each with its own task for a wheeled bimanual I2RT YAM cart.
    Every station is simulated on its own, checked with numbers, exported, and composited into the same room for
    rendering.</p>
  </header>

  <nav class="controls" aria-label="Stations">
    <div class="tabs" data-group="task">
      <button type="button" data-v="all" aria-pressed="true">All</button>{tabs}
    </div>
    <span class="count">{n_done} of {n_st} stations done</span>
  </nav>

  <main>
    <section class="ov" data-task="all">
      {overview}
    </section>
    {sections}

    <section class="part" id="how" data-task="all">
      <h2>How it was built</h2>
      {how}
      {test_clip}
    </section>
  </main>

  <footer>
    <p class="links" style="margin:0 0 .6rem">{src_links}</p>
    Source and outputs live in <code>~/projects/yam-arcade</code> on peefton. This page is rebuilt from the
    station outputs by <code>site/build_site.py</code>.
  </footer>
</div>
<script>
const state = {{task: "all"}};
function apply() {{
  for (const s of document.querySelectorAll("section[data-task]")) {{
    const t = s.dataset.task;
    s.hidden = !(state.task === "all" || t === state.task || (t === "all" && state.task === "all"));
  }}
  for (const b of document.querySelectorAll(".tabs button"))
    b.setAttribute("aria-pressed", String(b.dataset.v === state.task));
  history.replaceState(null, "", state.task !== "all" ? "#" + state.task : location.pathname);
  for (const v of document.querySelectorAll("section:not([hidden]) video")) {{ v.currentTime = 0; v.play().catch(() => {{}}); }}
}}
document.querySelector(".tabs").addEventListener("click", e => {{
  const b = e.target.closest("button"); if (!b) return; state.task = b.dataset.v; apply();
}});
const h = location.hash.slice(1);
if (h && document.getElementById(h) && document.getElementById(h).dataset.task === h) {{ state.task = h; apply(); }}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
