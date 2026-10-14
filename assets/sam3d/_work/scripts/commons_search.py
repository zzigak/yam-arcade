"""Search Wikimedia Commons (full-text or a category) for photos; list them and optionally download thumbnails.

usage: python -I commons_search.py "query" [--n 20] [--cat] [--dl OUTDIR] [--w 512] [--idx 0,3]
"""
import json, os, sys, time, urllib.parse, urllib.request, urllib.error

API = "https://commons.wikimedia.org/w/api.php"
UA = "yam-arcade-assets/0.1 (research simulation asset fetch; contact ziga)"


def fetch(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and k < tries - 1:
                time.sleep(5 * (k + 1)); continue
            raise


def api(params):
    return json.loads(fetch(API + "?" + urllib.parse.urlencode(dict(params, format="json"))))


def search(q, n=20, width=512, cat=False):
    if cat:
        gen = dict(generator="categorymembers", gcmtitle=q if q.startswith("Category:") else "Category:" + q,
                   gcmtype="file", gcmlimit=n)
    else:
        gen = dict(generator="search", gsrsearch=f"filetype:bitmap {q}", gsrnamespace=6, gsrlimit=n)
    d = api(dict(action="query", prop="imageinfo", iiprop="url|size|extmetadata|mime", iiurlwidth=width, **gen))
    out = []
    for p in sorted(d.get("query", {}).get("pages", {}).values(), key=lambda p: (p.get("index", 0), p["title"])):
        ii = p["imageinfo"][0]
        em = ii.get("extmetadata", {})
        out.append(dict(title=p["title"], w=ii["width"], h=ii["height"], mime=ii.get("mime"),
                        license=em.get("LicenseShortName", {}).get("value"),
                        artist=em.get("Artist", {}).get("value", "")[:200],
                        page=ii.get("descriptionurl"), thumb=ii.get("thumburl"), url=ii.get("url")))
    return out


if __name__ == "__main__":
    a = sys.argv
    q = a[1]
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    res = search(q, int(opt("--n", 20)), int(opt("--w", 512)), "--cat" in a)
    for i, r in enumerate(res):
        print(i, r["title"][:90], r["w"], r["h"], r["license"], r["mime"])
    if "--dl" in a:
        out = opt("--dl", None); os.makedirs(out, exist_ok=True)
        old = json.load(open(os.path.join(out, "results.json"))) if os.path.exists(os.path.join(out, "results.json")) else {}
        idx = [int(x) for x in opt("--idx", "").split(",")] if "--idx" in a else range(len(res))
        for i in idx:
            r = res[i]
            if not r["thumb"] or r["mime"] not in ("image/jpeg", "image/png"):
                continue
            fn = f"{i:02d}.jpg" if r["mime"] == "image/jpeg" else f"{i:02d}.png"
            try:
                open(os.path.join(out, fn), "wb").write(fetch(r["thumb"]))
                old[fn] = r
            except Exception as e:
                print("fail", i, e)
            time.sleep(1.0)
        json.dump(old, open(os.path.join(out, "results.json"), "w"), indent=1)
