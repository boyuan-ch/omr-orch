#!/usr/bin/env python3
"""
OMR-NED between predicted and ground-truth MusicXML, pairing files on page number.

    python run_omrned.py --gt-dir ../../data/sample/gt/musicxml \
                         --pred-dir ../../work/orch_dataset/Bee_1_challenge/xmls \
                         --out results_omrned.json

Metric: `python -m musicdiff --ml_training_evaluation`, score = last column of its output.csv,
OMR-NED = OMR-ED / total numsyms (lower is better) -- the same invocation used for the paper.
Install with `pip install musicdiff`; set $MUSICDIFF_PY to use a different interpreter.
"""
import argparse, csv, json, math, os, re, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

MUSICDIFF_PY = os.environ.get("MUSICDIFF_PY", sys.executable)


def page_of(path):
    m = re.search(r"_(\d+)\.musicxml$", os.path.basename(path))
    return int(m.group(1)) if m else None


def run_batch(gt_dir, pred_dir, out_dir):
    subprocess.run([MUSICDIFF_PY, "-m", "musicdiff", "--ml_training_evaluation",
                    "--ground_truth_folder", gt_dir, "--predicted_folder", pred_dir,
                    "--output_folder", out_dir], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", default="results_omrned.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-csv", default=None, help="directory to keep musicdiff's per-batch CSVs in")
    a = ap.parse_args()

    gt = {page_of(p): os.path.join(a.gt_dir, p) for p in os.listdir(a.gt_dir) if page_of(p)}
    pred = {page_of(p): os.path.join(a.pred_dir, p) for p in os.listdir(a.pred_dir) if page_of(p)}
    pages = sorted(set(gt) & set(pred))
    if not pages:
        raise SystemExit("no shared page numbers between the two folders")
    print(f"scoring {len(pages)} pages with {MUSICDIFF_PY}")

    workers = max(1, min(a.workers, len(pages)))
    chunk = math.ceil(len(pages) / workers)
    chunks = [pages[i:i + chunk] for i in range(0, len(pages), chunk)]
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        batches = []
        for b, ch in enumerate(chunks):
            g, p, o = (os.path.join(tmp, f"{k}_{b}") for k in ("gt", "pred", "out"))
            for d in (g, p, o):
                os.makedirs(d)
            for n in ch:
                shutil.copyfile(gt[n], os.path.join(g, f"{n}.xml"))
                shutil.copyfile(pred[n], os.path.join(p, f"{n}.xml"))
            batches.append((b, g, p, o))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_batch, g, p, o) for _, g, p, o in batches]
            for f in as_completed(futs):
                f.result()
        for b, _, _, o in batches:
            oc = os.path.join(o, "output.csv")
            if not os.path.exists(oc):
                continue
            if a.keep_csv:
                os.makedirs(a.keep_csv, exist_ok=True)
                shutil.copy(oc, os.path.join(a.keep_csv, f"batch_{b}.csv"))
            with open(oc, encoding="utf-8") as f:
                rd = csv.reader(f, skipinitialspace=True); next(rd)
                for row in rd:
                    if not row or row[0].strip() == "Total:":
                        continue
                    stem = os.path.splitext(os.path.basename(row[2].strip()))[0]
                    if stem.isdigit():
                        results[int(stem)] = dict(**{"OMR-NED": float(row[-1])},
                                                  OMR_ED=float(row[-2]), total_numsyms=float(row[-3]))

    vals = [v["OMR-NED"] for v in results.values()]
    summary = dict(pages_scored=len(vals), mean_OMR_NED=(sum(vals) / len(vals)) if vals else None,
                   per_page={str(k): v for k, v in sorted(results.items())})
    with open(a.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{'page':>6} {'OMR-NED':>10}")
    for n in sorted(results):
        print(f"{n:>6} {results[n]['OMR-NED'] * 100:>10.2f}")
    if vals:
        print(f"{'MEAN':>6} {sum(vals) / len(vals) * 100:>10.2f}   (x100, lower is better)")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
