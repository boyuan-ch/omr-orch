#!/usr/bin/env python3
"""
Bridge stage 1 (layout metadata) to stage 2 (OMR).

Stage 1 writes one `<page>_trans_gpt_final.csv` per page folder. Stage 2 expects a different tree,
and the two disagree on zero-padding, which is the easiest thing to get wrong:

    <out>/<sheet>/csv/<sheet>_<NNN>.csv          3-digit zero padded
    <out>/<sheet>/csv/<sheet>.json               ordered instrument list for the piece
    <out>/<sheet>/imgs/<sheet>_<N>/<sheet>_<N>.png   NOT padded

Usage
    # from stage-1 output
    python bridge_metadata_to_omr.py --metadata-dir work/pages --images-dir data/sample/images \
        --sheet Bee_1_challenge --out work/orch_dataset
    # or straight from the shipped ground-truth metadata (skips stage 1 entirely)
    python bridge_metadata_to_omr.py --metadata-dir data/sample/gt/metadata --images-dir data/sample/images \
        --sheet Bee_1_challenge --out work/orch_dataset
"""
import argparse
import glob
import json
import os
import re
import shutil

import pandas as pd


def page_number(name):
    m = re.search(r"_(\d+)(?:_trans_gpt_final)?\.(?:csv|png|jpg|jpeg)$", os.path.basename(name))
    return int(m.group(1)) if m else None


def instrument_list(df):
    """Replicate ScoreMetaData's instrument naming: '<ins>[__<part>][:<tone>]', system-1 rows only."""
    d = df.copy()
    for i in (1, 2, 3):
        d[f"c{i}"] = (d[f"ins{i}"].astype("object")
                      + d[f"part{i}"].apply(lambda x: "" if pd.isna(x) else f"__{int(float(x))}")
                      + d[f"tone{i}"].apply(lambda x: "" if pd.isna(x) else f":{x}"))
    mask = d["ins1"] == "timpani"                       # timpani is always a single unnumbered staff
    d.loc[mask, "c1"] = "timpani"
    d.loc[mask, ["c2", "c3"]] = None
    if "system" in d.columns:
        d = d[d["system"] == d["system"].dropna().iloc[0]]
    return [str(x) for x in d[["c1", "c2", "c3"]].to_numpy().flatten() if pd.notna(x)]


def copy_images(args, pages):
    """Stage 2 wants imgs/<sheet>_<N>/<sheet>_<N>.png with the page number NOT zero padded."""
    for n in pages:
        d = os.path.join(args.out, args.sheet, "imgs", f"{args.sheet}_{n}")
        os.makedirs(d, exist_ok=True)
        hit = [p for p in glob.glob(os.path.join(args.images_dir, "*")) if page_number(p) == n]
        if not hit:
            print(f"  WARNING: no image for page {n}")
            continue
        shutil.copyfile(hit[0], os.path.join(d, f"{args.sheet}_{n}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-dir", required=True,
                    help="dir of per-page CSVs, or the stage-1 work dir holding <page>/ folders")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rebuild-instruments", action="store_true",
                    help="derive the instrument list from the CSVs even if the dataset ships one")
    ap.add_argument("--union-instruments", action="store_true",
                    help="build the instrument list from every page instead of only the first")
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.metadata_dir, "*_trans_gpt_final.csv")) +
                  glob.glob(os.path.join(args.metadata_dir, "*", "*_trans_gpt_final.csv")))
    if not csvs:      # fall back to plain per-page CSVs (e.g. the shipped ground truth)
        csvs = sorted(p for p in glob.glob(os.path.join(args.metadata_dir, f"{args.sheet}_*.csv"))
                      if page_number(p) is not None)
    if not csvs:
        raise SystemExit(f"no per-page CSV found under {args.metadata_dir}")

    csv_out = os.path.join(args.out, args.sheet, "csv")
    os.makedirs(csv_out, exist_ok=True)
    pages = []
    for src in csvs:
        n = page_number(src)
        if n is None:
            continue
        shutil.copyfile(src, os.path.join(csv_out, f"{args.sheet}_{n:03d}.csv"))
        pages.append(n)
    pages.sort()

    # The piece-level instrument list is authoritative for stage 2: it fixes the part order of the
    # output score. If the dataset ships one, copy it rather than re-deriving it -- the naming
    # ("flute" vs "flute__1"/"flute__2") depends on readCsvAndEditViolin's part-number cleanup, and a
    # mismatch silently drops staves downstream.
    shipped = os.path.join(args.metadata_dir, f"{args.sheet}.json")
    if os.path.exists(shipped) and not args.rebuild_instruments:
        shutil.copyfile(shipped, os.path.join(csv_out, f"{args.sheet}.json"))
        ins = json.load(open(shipped, encoding="utf-8"))
        print(f"instruments ({len(ins)}): {ins}   [copied from {shipped}]")
        copy_images(args, pages)
        print(f"pages: {pages}")
        print(f"-> {os.path.join(args.out, args.sheet)}")
        return

    frames = [pd.read_csv(os.path.join(csv_out, f"{args.sheet}_{n:03d}.csv")) for n in pages]
    if args.union_instruments:
        seen, ins = set(), []
        for f in frames:
            for x in instrument_list(f):
                if x not in seen:
                    seen.add(x); ins.append(x)
    else:
        ins = instrument_list(frames[0])
    with open(os.path.join(csv_out, f"{args.sheet}.json"), "w", encoding="utf-8") as f:
        json.dump(ins, f, ensure_ascii=False, indent=2)

    copy_images(args, pages)

    print(f"pages: {pages}")
    print(f"instruments ({len(ins)}): {ins}")
    print(f"-> {os.path.join(args.out, args.sheet)}")


if __name__ == "__main__":
    main()
