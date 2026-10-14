"""Load SAM 3D once and reconstruct every job in jobs.json. Run with the sam-3d-objects venv.
usage: python run_sam3d.py jobs.json OUTDIR"""
import json, os, sys, time, traceback
sys.path.insert(0, os.path.expanduser("~/tools/sam-3d-objects"))
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
jobs = json.load(open(sys.argv[1])); out = sys.argv[2]
IN = os.path.expanduser("~/projects/yam-arcade/assets/sam3d/_work/inputs")
from sam3d_infer import load_model, reconstruct
import torch
t0 = time.time(); model = load_model(); print(f"model loaded {time.time()-t0:.1f}s", flush=True)
for name, seeds in jobs:
    for seed in seeds:
        pre = f"{out}/{name}_s{seed}"
        if os.path.exists(pre + ".glb"):
            print("skip", pre); continue
        t = time.time()
        try:
            res, files = reconstruct(model, f"{IN}/{name}.png", f"{IN}/{name}_mask.png", pre, seed=seed)
            print(f"done {name} seed {seed} {time.time()-t:.1f}s peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
        except Exception:
            traceback.print_exc(); print(f"FAILED {name} {seed}", flush=True)
        os.path.exists(pre + ".ply") and os.remove(pre + ".ply")  # splats are not needed downstream (~200 MB each)
del model; torch.cuda.empty_cache(); print("all done", flush=True)
