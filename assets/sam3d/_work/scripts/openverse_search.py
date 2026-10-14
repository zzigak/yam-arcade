"""Search Openverse images; print id, title, size, license, source url. --dl OUTDIR downloads the listed indices.
usage: python -I openverse_search.py "query" [--n 20] [--dl OUTDIR --idx 0,3,5]"""
import json, os, sys, urllib.parse, urllib.request
UA = "yam-arcade-assets/0.1 (research sim)"
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read()
a = sys.argv
q = a[1]; n = int(a[a.index("--n")+1]) if "--n" in a else 20
res = json.loads(get("https://api.openverse.org/v1/images/?" + urllib.parse.urlencode(dict(q=q, page_size=n, license_type="commercial,modification"))))["results"]
for i, r in enumerate(res):
    print(i, r["title"][:60], r.get("width"), r.get("height"), r["license"], r["license_version"], r["provider"], r["foreign_landing_url"])
if "--dl" in a:
    out = a[a.index("--dl")+1]; os.makedirs(out, exist_ok=True)
    idx = [int(x) for x in a[a.index("--idx")+1].split(",")] if "--idx" in a else range(len(res))
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    for i in idx:
        r = res[i]
        try:
            data = get(r["url"])
            open(os.path.join(out, f"ov{i:02d}.jpg"), "wb").write(data)
        except Exception as e:
            try:
                data = get(r["thumbnail"]); open(os.path.join(out, f"ov{i:02d}_thumb.jpg"), "wb").write(data)
            except Exception as e2:
                print("fail", i, e, e2)
