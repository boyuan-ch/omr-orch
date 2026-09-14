#!/usr/bin/env python3
"""
Build manifest.json for run_all.py by pairing ground-truth and predicted MusicXML on page number.

    python make_manifest.py --gt-dir ../../data/sample/gt/musicxml \
                            --pred-dir ../../work/orch_dataset/Bee_1_challenge/xmls \
                            --sheet Bee_1_challenge --out manifest.json

Records per page:
  cells           size of the zss matrix -> drives run_all.py's RAM admission control
  est_gb          predicted peak RSS (~3x measured)
  gold_raw_nodes  reference tree size after cut_xml; --min-gold-nodes filters on this
                  (the paper excludes pages below 1000)
"""
import argparse, copy, json, os, re, sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ted_core import cut_xml, LIMIT
from utils.TEDn_eval.symbolic.Pruner import Pruner
from utils.TEDn_eval.symbolic.actual_durations_to_fractional import actual_durations_to_fractional
from utils.TEDn_eval.evaluation.TEDn import Xml4ZSS_Levenshtein, encode_notes, NoteContentCoder


def page_of(path):
    m = re.search(r"_(\d+)\.musicxml$", os.path.basename(path))
    return int(m.group(1)) if m else None


def _count(e):
    return 1 + sum(_count(c) for c in Xml4ZSS_Levenshtein.get_children(e))


def sizes(xml_str):
    cut, _ = cut_xml(xml_str, LIMIT)
    raw = sum(1 for _ in ET.fromstring(cut).iter())
    root = ET.fromstring(ET.canonicalize(cut, strip_text=True))
    for ch in [c for c in root if c.tag != "part"]:
        root.remove(ch)
    pr = Pruner(prune_durations=False, prune_measure_attributes=False, prune_prints=True,
                prune_slur_numbering=True, prune_directions=True, prune_barlines=True,
                prune_harmony=True)
    for p in root.findall("part"):
        actual_durations_to_fractional(p); pr.process_part(p)
    enc = encode_notes(copy.deepcopy(root), NoteContentCoder())
    return raw, [_count(p) for p in enc.findall("part")]


def job(args):
    page, gt, pred = args
    g_raw, g_parts = sizes(open(gt, encoding="utf-8").read())
    _, p_parts = sizes(open(pred, encoding="utf-8").read())
    if len(p_parts) >= len(g_parts):                       # partwise comparison
        cells = sum(a * b for a, b in zip(p_parts[:len(g_parts)], g_parts))
    else:                                                  # whole-score comparison
        cells = (1 + sum(p_parts)) * (1 + sum(g_parts))
    return page, g_raw, cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--sheet", default="piece")
    ap.add_argument("--sym", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "manifest.json"))
    a = ap.parse_args()

    gt = {page_of(p): p for p in sorted(os.listdir(a.gt_dir)) if page_of(p)}
    gt = {n: os.path.join(a.gt_dir, os.path.basename(p)) for n, p in gt.items()}
    pred = {page_of(p): os.path.join(a.pred_dir, p) for p in sorted(os.listdir(a.pred_dir)) if page_of(p)}
    common = sorted(set(gt) & set(pred))
    if not common:
        raise SystemExit(f"no page numbers shared between {a.gt_dir} and {a.pred_dir}")
    missing = sorted(set(gt) - set(pred))
    if missing:
        print(f"  note: {len(missing)} GT pages have no prediction: {missing}")

    with Pool(min(8, len(common))) as pool:
        res = pool.map(job, [(n, gt[n], pred[n]) for n in common])

    pages = [dict(id=f"{a.sheet}_p{n:03d}", sym=a.sym, page=n, mvt=1, page_in_mvt=n,
                  gt=os.path.abspath(gt[n]), pred=os.path.abspath(pred[n]),
                  cells=cells, gold_raw_nodes=g_raw,
                  est_gb=round(0.15 + 0.25e-6 * cells, 3))
             for n, g_raw, cells in res]
    with open(a.out, "w") as f:
        json.dump(dict(description=f"TEDn manifest for {a.sheet}", pages=pages), f, indent=1)
    print(f"{len(pages)} pages -> {a.out}")
    print(f"  reference nodes: min {min(p['gold_raw_nodes'] for p in pages)}, "
          f"max {max(p['gold_raw_nodes'] for p in pages)}; "
          f"{sum(1 for p in pages if p['gold_raw_nodes'] < 1000)} below the paper's 1000-node cutoff")


if __name__ == "__main__":
    main()
