"""
Compute TEDn for ONE page in its own process, then exit (so all memory goes back to the OS).

    python worker.py --gt GT.musicxml --pred PRED.musicxml --out RESULT.json
                     [--impl lean|stock] [--mem-cap-gb 20]

Scoring is byte-for-byte the original pipeline (legato/scripts/compute_TEDn.py compute_score ->
TEDn_xml_xml(flavor='lmx'), LIMIT=6000); see ted_core.py. The only optional change is --impl lean,
which swaps zss.distance for the memory-lean version in lean_zss.py (same distance, no edit script).

Safety: --mem-cap-gb sets RLIMIT_AS on this process. If the page would exceed it, Python raises
MemoryError inside this process instead of dragging the whole machine down; the result file then
records status="oom" so the scheduler can retry it later with more headroom.
"""
import argparse
import json
import os
import resource
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def write_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--impl", choices=["lean", "stock"], default="lean")
    ap.add_argument("--mem-cap-gb", type=float, default=0.0, help="RLIMIT_AS for this process; 0 = none")
    ap.add_argument("--meta", default="{}", help="JSON copied verbatim into the result")
    args = ap.parse_args()

    if args.mem_cap_gb > 0:
        cap = int(args.mem_cap_gb * (1024 ** 3))
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    result = dict(json.loads(args.meta), gt=args.gt, pred=args.pred, impl=args.impl,
                  mem_cap_gb=args.mem_cap_gb, pid=os.getpid())
    t0 = time.time()
    try:
        import lean_zss
        if args.impl == "lean":
            lean_zss.install()
        from ted_core import compute_score

        with open(args.pred, "r", encoding="utf-8") as f:
            pred_xml = f.read()
        with open(args.gt, "r", encoding="utf-8") as f:
            gold_xml = f.read()

        _, score = compute_score((0, pred_xml, gold_xml))
        result.update(status="ok", TEDn=score)
        code = 0
    except MemoryError:
        result.update(status="oom", error="MemoryError (hit --mem-cap-gb)")
        code = 3
    except Exception as e:
        result.update(status="error", error=f"{type(e).__name__}: {e}",
                      traceback=traceback.format_exc()[-2000:])
        code = 2
    result["seconds"] = round(time.time() - t0, 3)
    result["peak_rss_gb_self"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2), 3)
    write_atomic(args.out, result)
    sys.exit(code)


if __name__ == "__main__":
    main()
