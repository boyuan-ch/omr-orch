# env: intern
import argparse
import difflib
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


# Standard symphony orchestra instrument order (per staff line)
SYMPHONY_ORCHESTRA = [
    'flute',    'flute',
    'oboe',     'oboe',
    'clarinet', 'clarinet',
    'bassoon',  'bassoon',
    'horn',     'horn',     'horn', 'horn',
    'trumpet',  'trumpet',
    'trombone', 'trombone', 'trombone',
    'tuba',
    'timpani',
    'violin',   'violin',
    'viola',
    'cello',
    'bass',
]

ORCH_COUNTS = {}
for _i in SYMPHONY_ORCHESTRA:
    ORCH_COUNTS[_i] = ORCH_COUNTS.get(_i, 0) + 1


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def find_nearest_staff(y, staff_y_centers):
    best_idx, best_dist = None, float("inf")
    for i, sy in enumerate(staff_y_centers):
        d = abs(y - sy)
        if d < best_dist:
            best_idx, best_dist = i, d
    return best_idx


def find_nearest_staff_upper(y, staff_y_centers):
    best_idx, best_dist = None, float("inf")
    for i, sy in enumerate(staff_y_centers):
        if sy < y:
            d = abs(y - sy)
            if d < best_dist:
                best_idx, best_dist = i, d
    return best_idx


def find_nearest_staff_lower(y, staff_y_centers):
    best_idx, best_dist = None, float("inf")
    for i, sy in enumerate(staff_y_centers):
        if sy > y:
            d = abs(y - sy)
            if d < best_dist:
                best_idx, best_dist = i, d
    return best_idx


def get_current_staff_list(n_staves, staff_to_tokens, classified_tokens):
    results = []
    for s in range(n_staves):
        found = []
        for idx in sorted(staff_to_tokens[s]):
            name = (classified_tokens[idx].get("Instrument") or "").strip()
            if name and name not in found:
                found.append(name)
        results.append(found)
    return results


# ---------------------------------------------------------------------------
# Core matching
# ---------------------------------------------------------------------------

def match_and_process(staff_gt_json_path, filtered_ocr_json_path, classified_path):
    with open(staff_gt_json_path, "r", encoding="utf-8") as f:
        staff_info = json.load(f)
    with open(filtered_ocr_json_path, "r", encoding="utf-8") as f:
        ocr_info = json.load(f)
    with open(classified_path, "r", encoding="utf-8") as f:
        cls_info = json.load(f)

    staff_y_centers   = staff_info["staff_y_centers"]
    n_staves          = staff_info["n_staves"]
    filtered_text     = ocr_info["filtered_text"]
    filtered_bbox     = ocr_info["filtered_yolo_bbox"]
    classified_tokens = cls_info["tokens"]

    tokens = []
    for i, (txt, bbox, cls_item) in enumerate(zip(filtered_text, filtered_bbox, classified_tokens)):
        y = bbox[1]
        tokens.append({
            "idx":                 i,
            "text":                txt,
            "cls":                 cls_item["class_id"],
            "y":                   y,
            "nearest_staff":       find_nearest_staff(y, staff_y_centers),
            "nearest_staff_upper": find_nearest_staff_upper(y, staff_y_centers),
            "nearest_staff_lower": find_nearest_staff_lower(y, staff_y_centers),
        })

    staff_to_tokens   = {i: set() for i in range(n_staves)}
    inst_to_staff_set = defaultdict(set)

    parts       = [t for t in tokens if t["cls"] == 1]
    instruments = [t for t in tokens if t["cls"] in (0, 5)]

    # Step 1: instrument → nearest staff (snapshot for init_list)
    for inst in instruments:
        s = inst["nearest_staff"]
        inst_to_staff_set[inst["idx"]].add(s)
        staff_to_tokens[s].add(inst["idx"])

    init_list = get_current_staff_list(n_staves, staff_to_tokens, classified_tokens)

    # Step 2: part → nearest staff
    for p in parts:
        s = p["nearest_staff"]
        p["assigned_staff"] = s
        staff_to_tokens[s].add(p["idx"])

    # Step 3: part proximity extends instrument's staff coverage
    for p in parts:
        best_inst_idx, best_dist = None, float("inf")
        for inst in instruments:
            d = abs(p["y"] - inst["y"])
            if d < best_dist:
                best_dist, best_inst_idx = d, inst["idx"]
        if best_inst_idx is not None:
            target_staff = p["assigned_staff"]
            inst_to_staff_set[best_inst_idx].add(target_staff)
            staff_to_tokens[target_staff].add(best_inst_idx)

    # Step 4: ensemble instructions — divisi → staff above, others → staff below
    for t in tokens:
        if t["cls"] == 3:
            s = t["nearest_staff_upper"] if "v" in t["text"].lower() else t["nearest_staff_lower"]
            if s is not None:
                staff_to_tokens[s].add(t["idx"])

    # Step 5: tone → follows nearest instrument's staffs
    for t in tokens:
        if t["cls"] == 2:
            best_inst_idx, best_dist = None, float("inf")
            for inst in instruments:
                d = abs(t["y"] - inst["y"])
                if d < best_dist:
                    best_dist, best_inst_idx = d, inst["idx"]
            if best_inst_idx is not None:
                for s in inst_to_staff_set[best_inst_idx]:
                    staff_to_tokens[s].add(t["idx"])

    final_list = get_current_staff_list(n_staves, staff_to_tokens, classified_tokens)
    return init_list, final_list


# ---------------------------------------------------------------------------
# CSV alignment helpers
# ---------------------------------------------------------------------------

def align_csv_to_predict(compare_list, csv_list):
    def simplify(x):
        return str(x[0]).lower() if (isinstance(x, list) and x) else ""

    matcher = difflib.SequenceMatcher(
        None,
        [simplify(x) for x in compare_list],
        [simplify(x) for x in csv_list],
    )
    pred_to_csv = [None] * len(compare_list)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("equal", "replace"):
            for i, j in zip(range(i1, i2), range(j1, j2)):
                pred_to_csv[i] = j
    return pred_to_csv


def _orch_pos(inst_list):
    for inst in (inst_list if isinstance(inst_list, list) else [inst_list]):
        name = str(inst).strip().lower()
        for k, v in enumerate(SYMPHONY_ORCHESTRA):
            if v == name:
                return k
    return -1


def _split_by_order(seq):
    splits, prev = [], -1
    for i, item in enumerate(seq):
        pos = _orch_pos(item)
        if pos >= 0:
            if pos < prev:
                splits.append(i)
            prev = pos
    return splits


def _split_seq(seq, splits):
    parts, prev = [], 0
    for s in splits:
        parts.append(seq[prev:s])
        prev = s
    parts.append(seq[prev:])
    return parts


def find_bridge_instruments(prev_ins, next_ins):
    p = prev_ins.strip().lower() if prev_ins else None
    n = next_ins.strip().lower() if next_ins else None
    if not p or not n:
        return []
    last_p = max((k for k, v in enumerate(SYMPHONY_ORCHESTRA) if v == p), default=None)
    if last_p is None:
        return []
    first_n = next((k for k in range(last_p + 1, len(SYMPHONY_ORCHESTRA)) if SYMPHONY_ORCHESTRA[k] == n), None)
    if first_n is None:
        return []
    return SYMPHONY_ORCHESTRA[last_p + 1 : first_n]


def postprocess_aligned(df_aligned, page_int):
    df_post = df_aligned.copy().reset_index(drop=True)
    seen_counts      = {}
    fill_counter     = {}
    bridge_queue     = []
    pre_fill_counter = {}
    current_system   = None

    for i in range(len(df_post)):
        ins_val    = df_post.at[i, 'ins1']
        row_is_nan = pd.isna(ins_val) or (isinstance(ins_val, str) and not ins_val.strip())

        if not row_is_nan:
            row_system = df_post.at[i, 'system'] if 'system' in df_post.columns else None
            if pd.notna(row_system) and row_system != current_system:
                current_system   = row_system
                seen_counts      = {}
                fill_counter     = {}
                bridge_queue     = []
                pre_fill_counter = {}

        if row_is_nan:
            prev_ins = df_post.at[i - 1, 'ins1'] if i > 0 else None
            if pd.isna(prev_ins) or (isinstance(prev_ins, str) and not str(prev_ins).strip()):
                prev_ins = None

            next_ins = None
            for j in range(i + 1, len(df_post)):
                v = df_post.at[j, 'ins1']
                if pd.notna(v) and isinstance(v, str) and v.strip():
                    next_ins = v
                    break
            next_key = next_ins.strip().lower() if next_ins else None

            filled = False

            # Priority 1: pre-fill from next instrument if it needs more staves
            if next_key and ORCH_COUNTS.get(next_key, 1) > 1:
                next_seen    = seen_counts.get(next_key, 0)
                next_prefill = pre_fill_counter.get(next_key, 0)
                if next_seen + next_prefill < ORCH_COUNTS[next_key]:
                    df_post.at[i, 'ins1'] = next_ins
                    pre_fill_counter[next_key] = next_prefill + 1
                    filled = True

            # Priority 2: forward-fill from prev instrument (within its stave count)
            if not filled and prev_ins and isinstance(prev_ins, str):
                prev_key  = prev_ins.strip().lower()
                already   = fill_counter.get(prev_key, 0)
                seen      = seen_counts.get(prev_key, 0)
                max_total = ORCH_COUNTS.get(prev_key, 1)
                if seen + already < max_total:
                    df_post.at[i, 'ins1'] = prev_ins
                    fill_counter[prev_key] = already + 1
                    filled = True

            # Priority 2b: nearest under-counted instrument preceding next in orch order
            if not filled:
                first_next_pos = len(SYMPHONY_ORCHESTRA)
                if next_key:
                    first_next_pos = next(
                        (k for k, v in enumerate(SYMPHONY_ORCHESTRA) if v == next_key),
                        len(SYMPHONY_ORCHESTRA),
                    )
                candidate = None
                for k in range(first_next_pos - 1, -1, -1):
                    inst = SYMPHONY_ORCHESTRA[k]
                    total = (seen_counts.get(inst, 0)
                             + pre_fill_counter.get(inst, 0)
                             + fill_counter.get(inst, 0))
                    if seen_counts.get(inst, 0) > 0 and total < ORCH_COUNTS.get(inst, 1):
                        candidate = inst
                        break
                if candidate:
                    df_post.at[i, 'ins1'] = candidate
                    pre_fill_counter[candidate] = pre_fill_counter.get(candidate, 0) + 1
                    filled = True

            # Priority 3: bridge instruments between prev and next in orch order
            if not filled:
                if not bridge_queue:
                    bridge_queue = find_bridge_instruments(prev_ins, next_ins)
                if bridge_queue:
                    df_post.at[i, 'ins1'] = bridge_queue.pop(0)

            if i > 0:
                for col in ['system', 'staffgroup']:
                    if col in df_post.columns and pd.isna(df_post.at[i, col]):
                        df_post.at[i, col] = df_post.at[i - 1, col]
        else:
            ins_key = ins_val.strip().lower() if isinstance(ins_val, str) and ins_val.strip() else ''
            swapped = False
            if ins_key:
                first_current = next((k for k, v in enumerate(SYMPHONY_ORCHESTRA) if v == ins_key), None)
                if first_current is not None and first_current > 0:
                    pred_key  = SYMPHONY_ORCHESTRA[first_current - 1]
                    pred_seen = seen_counts.get(pred_key, 0)
                    pred_max  = ORCH_COUNTS.get(pred_key, 1)
                    next_is_nan = False
                    if i + 1 < len(df_post):
                        nv = df_post.at[i + 1, 'ins1']
                        next_is_nan = pd.isna(nv) or (isinstance(nv, str) and not nv.strip())
                    cur_predicted = ''
                    if 'predicted' in df_post.columns:
                        cur_predicted = str(df_post.at[i, 'predicted']).strip().lower()
                        if cur_predicted == 'nan':
                            cur_predicted = ''
                    predicted_ok = cur_predicted == '' or cur_predicted == pred_key
                    if pred_seen > 0 and pred_seen < pred_max and pred_max > 1 and next_is_nan and predicted_ok:
                        df_post.at[i, 'ins1'] = pred_key
                        bridge_queue = [ins_val] + bridge_queue
                        seen_counts[pred_key] = pred_seen + 1
                        swapped = True

            fill_counter     = {}
            pre_fill_counter = {}
            if not swapped:
                bridge_queue = []
                if ins_key:
                    seen_counts[ins_key] = seen_counts.get(ins_key, 0) + 1

    df_post['staff'] = range(1, len(df_post) + 1)
    df_post['page']  = page_int
    for col in ['page', 'staff', 'system', 'staffgroup']:
        if col in df_post.columns:
            df_post[col] = pd.to_numeric(df_post[col], errors='coerce').astype('Int64')
    return df_post


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_staff_gt_json(staff_gt_json_path, csv_path=None, page_int=1):
    folder = os.path.dirname(staff_gt_json_path)
    stem   = os.path.basename(staff_gt_json_path).replace("_yolostaff_gt.json", "")

    # Auto-discover *_trans_gpt.csv from the same folder if not given explicitly
    if csv_path is None:
        candidates = sorted(glob.glob(os.path.join(folder, "*_trans_gpt.csv")))
        if candidates:
            csv_path = candidates[0]
            print(f"[CSV] Auto-discovered: {csv_path}")

    filtered_ocr_path = os.path.join(folder, f"{stem}_ocr_filtered.json")
    classified_path   = os.path.join(folder, f"{stem}_ocr_filtered_classified_normalized.json")

    if not os.path.exists(filtered_ocr_path):
        print(f"[SKIP] OCR filtered JSON not found: {filtered_ocr_path}")
        return
    if not os.path.exists(classified_path):
        print(f"[SKIP] Classified JSON not found: {classified_path}")
        return

    init_list, final_list = match_and_process(staff_gt_json_path, filtered_ocr_path, classified_path)

    # Always save instrument-per-staff JSON
    output_path = os.path.join(folder, f"{stem}_staff_instruments.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "filename":          stem,
            "n_staves":          len(init_list),
            "init_instruments":  init_list,
            "final_instruments": final_list,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}  ({len(init_list)} staves)")

    if csv_path is None:
        print("[CSV] No *_trans_gpt.csv found; skipping CSV management.")
        return

    # --- Optional: compare against GT CSV and write back ---
    try:
        df_all = pd.read_csv(csv_path)
        df_all['page_processed'] = df_all['page'].ffill().astype(int)
        df_gt = df_all[df_all['page_processed'] == page_int].sort_values('staff').copy()

        has_instrument = df_gt[['ins1', 'ins2', 'ins3']].apply(
            lambda col: col.notna() & (col.astype(str).str.strip() != '') & (col.astype(str).str.lower() != 'nan')
        ).any(axis=1)
        df_gt = df_gt[has_instrument].copy()

        csv_list = []
        for _, row in df_gt.iterrows():
            st_ins = []
            for col in ['ins1', 'ins2', 'ins3']:
                val = row.get(col)
                if pd.notna(val) and str(val).strip().lower() != "nan" and str(val).strip():
                    st_ins.append(str(val).strip())
            csv_list.append(st_ins)
    except Exception as e:
        print(f"[CSV] Failed to load GT: {e}")
        return

    compare_list = init_list
    print(f"\n===== [Results] {stem} =====")
    for i in range(max(len(compare_list), len(csv_list))):
        f_val  = compare_list[i] if i < len(compare_list) else "N/A"
        c_val  = csv_list[i]     if i < len(csv_list)     else "N/A"
        status = "OK" if str(f_val).lower() == str(c_val).lower() else "!!"
        print(f"  Staff {i+1}: Predict {f_val} | CSV {c_val} [{status}]")

    pred_splits = _split_by_order(compare_list)
    csv_splits  = _split_by_order(csv_list)

    df_gt_reset  = df_gt.reset_index(drop=True)
    gt_cols      = [c for c in df_gt_reset.columns if c != 'page_processed']
    nan_row      = {col: np.nan for col in gt_cols}
    aligned_rows = []

    if pred_splits and len(pred_splits) == len(csv_splits):
        compare_parts = _split_seq(compare_list, pred_splits)
        csv_parts     = _split_seq(csv_list,     csv_splits)
        df_row_splits = _split_seq(list(range(len(df_gt_reset))), csv_splits)
        for cpart, csv_part, df_idxs in zip(compare_parts, csv_parts, df_row_splits):
            df_sys = df_gt_reset.iloc[df_idxs].reset_index(drop=True)
            p2c = align_csv_to_predict(cpart, csv_part)
            for i, csv_idx in enumerate(p2c):
                r = df_sys.iloc[csv_idx][gt_cols].to_dict() if csv_idx is not None else nan_row.copy()
                r['predicted'] = ", ".join(cpart[i]) if (isinstance(cpart[i], list) and cpart[i]) else ""
                aligned_rows.append(r)
    else:
        p2c = align_csv_to_predict(compare_list, csv_list)
        for i, csv_idx in enumerate(p2c):
            r = df_gt_reset.iloc[csv_idx][gt_cols].to_dict() if csv_idx is not None else nan_row.copy()
            r['predicted'] = ", ".join(compare_list[i]) if (isinstance(compare_list[i], list) and compare_list[i]) else ""
            aligned_rows.append(r)

    df_aligned = pd.DataFrame(aligned_rows, columns=gt_cols + ['predicted'])
    df_post    = postprocess_aligned(df_aligned, page_int)

    print(f"\n===== [Post-processed] {stem} =====")
    display_cols = [c for c in gt_cols + ['predicted'] if c in df_post.columns]
    print(df_post[display_cols].to_string(index=False))

    # Save as _modify.csv (input for f.py), leaving the original _trans_gpt.csv untouched
    modify_path = os.path.join(folder, f"{stem}_trans_gpt_modify.csv")
    df_full = pd.read_csv(csv_path)
    df_full['page_processed'] = df_full['page'].ffill().astype(int)
    df_other  = df_full[df_full['page_processed'] != page_int].drop(columns='page_processed')
    save_cols = [c for c in df_full.columns if c != 'page_processed']
    df_save   = df_post[[c for c in save_cols if c in df_post.columns]].copy()
    df_out    = pd.concat([df_other, df_save], ignore_index=True)
    df_out['page_processed'] = df_out['page'].ffill().astype(int)
    df_out    = df_out.sort_values(['page_processed', 'staff']).drop(columns='page_processed')
    df_out['page'] = df_out['page'].where(~df_out['page'].duplicated(), other=pd.NA)
    df_out.to_csv(modify_path, index=False)
    print(f"Saved: {modify_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(input_folder, csv_path=None, page_int=1):
    if not os.path.isdir(input_folder):
        raise ValueError(f"Not a directory: {input_folder}")

    staff_files = sorted(
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith("_yolostaff_gt.json")
    )

    if not staff_files:
        print(f"No *_yolostaff_gt.json files found in {input_folder}")
        return

    for staff_gt_path in staff_files:
        print(f"Processing: {staff_gt_path}")
        try:
            process_staff_gt_json(staff_gt_path, csv_path, page_int)
        except Exception as e:
            print(f"Error: {e}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Match OCR tokens to staves; optionally align against a GT CSV."
    )
    parser.add_argument(
        "input_folder",
        help="Folder containing *_yolostaff_gt.json, *_ocr_filtered.json, and *_ocr_filtered_classified_normalized.json",
    )
    parser.add_argument(
        "--csv",
        default=None,
        metavar="CSV_PATH",
        help="Path to *_trans_gpt.csv for alignment and update (auto-discovered from folder if omitted)",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        metavar="PAGE_INT",
        help="Page number in the CSV to align against (default: 1)",
    )
    args = parser.parse_args()
    main(args.input_folder, args.csv, args.page)
