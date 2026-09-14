import glob
import os
import sys

import pandas as pd

# ============================================================
# Instrument family sets
# ============================================================
WOODWIND   = {"flute", "oboe", "clarinet", "bassoon", "contrabassoon", "piccolo", "english horn"}
BRASS      = {"horn", "trumpet", "trombone", "tuba", "cornet"}
PERCUSSION = {"timpani", "percussion", "drum", "cymbal"}
STRINGS    = {"violin", "viola", "cello", "bass", "contrabass", "double bass"}


def instrument_family(ins):
    if pd.isna(ins):
        return "Unknown"
    s = str(ins).strip().lower()
    if s in WOODWIND:   return "Woodwind"
    if s in BRASS:      return "Brass"
    if s in PERCUSSION: return "Percussion"
    if s in STRINGS:    return "Strings"
    return "Other"


def infer_family(row):
    for col in ["ins1", "ins2", "ins3"]:
        val = row.get(col)
        if pd.notna(val) and str(val).strip().lower() not in ("nan", ""):
            return instrument_family(val)
    return "Unknown"


def is_single_instrument(row):
    """True when only ins1 is filled (ins2 and ins3 are empty)."""
    ins1 = row["ins1"]
    ins2 = row["ins2"]
    ins3 = row["ins3"]
    empty2  = pd.isna(ins2) or str(ins2).strip().lower() in ("", "nan")
    empty3  = pd.isna(ins3) or str(ins3).strip().lower() in ("", "nan")
    filled1 = pd.notna(ins1) and str(ins1).strip().lower() not in ("", "nan")
    return filled1 and empty2 and empty3


def assign_adjacent_parts(df):
    """
    Within each page, find runs of consecutive rows that share the same
    single instrument and assign part1 = 1, 2, 3 ... to any row whose
    part1 is currently empty.
    """
    df = df.copy()
    for _, grp in df.groupby("page_norm", sort=False):
        idxs = grp.index.tolist()
        i = 0
        while i < len(idxs):
            idx  = idxs[i]
            row  = df.loc[idx]
            if not is_single_instrument(row):
                i += 1
                continue
            ins_name = str(row["ins1"]).strip().lower()
            run = [idx]
            j = i + 1
            while j < len(idxs):
                jdx  = idxs[j]
                jrow = df.loc[jdx]
                if is_single_instrument(jrow) and str(jrow["ins1"]).strip().lower() == ins_name:
                    run.append(jdx)
                    j += 1
                else:
                    break
            if len(run) >= 2:
                for part_num, ridx in enumerate(run, start=1):
                    current = df.at[ridx, "part1"]
                    if pd.isna(current) or str(current).strip().lower() in ("", "nan"):
                        df.at[ridx, "part1"] = float(part_num)
            i += len(run)
    return df


def assign_violin_parts(df):
    df = df.copy()
    for _, grp in df.groupby("page_norm", sort=False):
        idxs    = grp.index.tolist()
        counter = 0
        for idx in idxs:
            row  = df.loc[idx]
            ins1 = row["ins1"]
            is_violin = (
                pd.notna(ins1)
                and str(ins1).strip().lower() == "violin"
                and is_single_instrument(row)
            )
            if is_violin:
                counter += 1
                current = df.at[idx, "part1"]
                if pd.isna(current) or str(current).strip().lower() in ("", "nan"):
                    df.at[idx, "part1"] = float(counter)
            else:
                counter = 0
    return df


def main(input_folder):
    input_folder = input_folder.rstrip("/\\")

    matches = sorted(glob.glob(os.path.join(input_folder, "*_trans_gpt_modify.csv")))
    if not matches:
        print(f"No *_trans_gpt_modify.csv found in {input_folder}")
        sys.exit(1)
    csv_input = matches[0]

    stem       = os.path.basename(csv_input)[:-len("_trans_gpt_modify.csv")]
    csv_output = os.path.join(input_folder, f"{stem}_trans_gpt_final.csv")

    df = pd.read_csv(csv_input)

    # Step 1b: remove hyphens (e.g. "contra-bassoon" → "contrabassoon")
    df = df.apply(
        lambda col: col.str.replace("-", "", regex=False) if col.dtype == object else col
    )

    # Step 2: page_norm — forward-filled page as 3-digit string
    page_num        = pd.to_numeric(df["page"], errors="coerce")
    page_num_ffill  = page_num.ffill().astype("Int64")
    df["page_norm"] = page_num_ffill.apply(
        lambda x: f"{int(x):03d}" if pd.notna(x) else pd.NA
    )

    # Step 3: fill page column for every row
    df["page"] = df["page_norm"]

    # Step 4: instrument family
    df["family"] = df.apply(infer_family, axis=1)

    # Step 5: assign part numbers for adjacent same-instrument runs
    df = assign_adjacent_parts(df)

    # Step 5b: assign ordered part numbers to consecutive violin rows
    df = assign_violin_parts(df)

    # Step 6: save
    df.to_csv(csv_output, index=False)
    print(f"Saved: {csv_output}  ({len(df)} rows)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python f.py <input_folder>")
        sys.exit(1)
    main(sys.argv[1])
