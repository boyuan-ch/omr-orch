#!/usr/bin/env python3
"""
Stage-1 metadata metrics: staff-count accuracy, staff accuracy, instrument F1 (paper Table 1).

    python metadata_metrics.py --pred-dir work/orch_dataset/Bee_1_challenge/csv \
                               --gt-dir data/sample/gt/metadata --sheet Bee_1_challenge

Definitions (Section 3.1 of the paper):
  staff-count accuracy  proportion of pages whose predicted number of staves is exact
  staff accuracy        proportion of ground-truth staves whose predicted instrument set matches
                        exactly at the aligned page and staff position (a position mismatch is an error)
  instrument F1         micro-averaged F1 over (page, staff, instrument) triples
"""
import argparse, glob, json, os, re
import pandas as pd

COLS = ("ins1", "ins2", "ins3")


def page_of(p):
    """Page number from either naming convention: `{sheet}_007.csv` (after the bridge) or
    `{sheet}_007_trans_gpt_final.csv` (straight out of stage 1)."""
    name = os.path.basename(p)
    m = re.search(r"_(\d+)(?:_trans_gpt_final)?\.csv$", name)
    return int(m.group(1)) if m else None


def load(d, sheet):
    out = {}
    for p in glob.glob(os.path.join(d, f"{sheet}_*.csv")) + glob.glob(os.path.join(d, "*_trans_gpt_final.csv")):
        n = page_of(p)
        if n is None:
            continue
        try:
            out[n] = pd.read_csv(p)
        except Exception as e:
            print(f"  WARNING: cannot read {p}: {e}")
    return out


def staff_sets(df, with_part_tone=False):
    rows = []
    for _, r in df.iterrows():
        s = set()
        for i, c in enumerate(COLS, start=1):
            v = r.get(c)
            if pd.isna(v) or str(v).strip() == "":
                continue
            lab = str(v).strip()
            if with_part_tone:
                part, tone = r.get(f"part{i}"), r.get(f"tone{i}")
                if pd.notna(part):
                    lab += f"__{int(float(part))}"
                if pd.notna(tone):
                    lab += f":{str(tone).strip()}"
            s.add(lab)
        rows.append(s)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--sheet", default="Bee_1_challenge")
    ap.add_argument("--with-part-tone", action="store_true",
                    help="also require part number and transposition to match")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    gt, pred = load(a.gt_dir, a.sheet), load(a.pred_dir, a.sheet)
    pages = sorted(set(gt) & set(pred))
    if not pages:
        raise SystemExit(f"no shared pages between {a.gt_dir} and {a.pred_dir}")
    missing = sorted(set(gt) - set(pred))

    n_pages_exact = 0
    staff_total = staff_exact = 0
    tp = n_gt_tot = n_pred_tot = 0
    per_page = {}
    for n in pages:
        g, p = staff_sets(gt[n], a.with_part_tone), staff_sets(pred[n], a.with_part_tone)
        exact_cnt = sum(1 for i in range(len(g)) if i < len(p) and g[i] == p[i])
        n_pages_exact += (len(g) == len(p))
        staff_total += len(g); staff_exact += exact_cnt
        for i in range(len(g)):
            q = p[i] if i < len(p) else set()
            tp += len(g[i] & q); n_gt_tot += len(g[i]); n_pred_tot += len(q)
        for i in range(len(g), len(p)):          # extra predicted staves still cost precision
            n_pred_tot += len(p[i])
        per_page[n] = dict(gt_staves=len(g), pred_staves=len(p), staff_exact=exact_cnt)

    prec = tp / n_pred_tot if n_pred_tot else 0.0
    rec = tp / n_gt_tot if n_gt_tot else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    res = dict(pages=len(pages), pages_missing_prediction=missing,
               staffcount_accuracy=n_pages_exact / len(pages),
               staff_accuracy=staff_exact / staff_total if staff_total else 0.0,
               instrument_precision=prec, instrument_recall=rec, instrument_f1=f1,
               per_page=per_page)

    print(f"pages compared           {len(pages)}" + (f"   (missing predictions: {missing})" if missing else ""))
    print(f"staff-count accuracy     {res['staffcount_accuracy'] * 100:6.2f}%   ({n_pages_exact}/{len(pages)} pages)")
    print(f"staff accuracy           {res['staff_accuracy'] * 100:6.2f}%   ({staff_exact}/{staff_total} staves)")
    print(f"instrument precision     {prec * 100:6.2f}%")
    print(f"instrument recall        {rec * 100:6.2f}%")
    print(f"instrument F1            {f1 * 100:6.2f}%")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
