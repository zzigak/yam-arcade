"""Openverse search; downloads thumbnails of all hits (for previewing) or full images (--full --idx ...).
usage: python -I ov.py "query" OUTDIR [--n 16] [--full --idx 0,3]"""
import json, os, sys, time, urllib.parse, urllib.request
UA = "yam-arcade-assets/0.1 (research simulation)"
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read()
a = sys.argv; q, out = a[1], a[2]
opt = lambda k, d: a[a.index(k)+1] if k in a else d
os.makedirs(out, exist_ok=True)
rj = os.path.join(out, "results.json")
if "--full" in a and os.path.exists(rj):
    res = json.load(open(rj))
else:
    res = json.loads(get("https://api.openverse.org/v1/images/?" + urllib.parse.urlencode(dict(q=q, page_size=int(opt("--n", 16))))))["results"]
    json.dump(res, open(rj, "w"), indent=1)
for i, r in enumerate(res):
    print(i, r["title"][:60], r.get("width"), r.get("height"), r["license"], r["license_version"], r["provider"])
if "--full" in a:
    for i in [int(x) for x in opt("--idx", "").split(",")]:
        try: open(os.path.join(out, f"full{i:02d}.jpg"), "wb").write(get(res[i]["url"]))
        except Exception as e: print("fail full", i, e)
else:
    for i, r in enumerate(res):
        try: open(os.path.join(out, f"t{i:02d}.jpg"), "wb").write(get(r["thumbnail"]))
        except Exception as e: print("fail", i, e)
