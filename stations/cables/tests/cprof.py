"""cProfile of station steps: 20 steps with every cable in the air (no contacts), then 20 after landing."""
import cProfile, io, os, pstats, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import station as S
import genesis as gs
gs.init(backend=gs.gpu, logging_level="warning")
env = S.build(robot="--robot" in sys.argv, krb_only="--krb-only" in sys.argv)
import subprocess
print("other GPU jobs:", len(subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                             capture_output=True, text=True).stdout.split()), flush=True)
env.step()
def prof(n, label):
    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    for _ in range(n):
        env.step()
    pr.disable()
    print(f"== {label}: {(time.time() - t0) / n:.3f} s/step", flush=True)
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(28)
    print(s.getvalue()[:6000], flush=True)
prof(20, "in the air")
for _ in range(int(1.5 / S.DT)):
    env.step()
if "--air-only" not in sys.argv:
    prof(20, "landed")
