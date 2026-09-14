import json
import csv
import os
import glob
import re
import sys


def process_staff(staff):
    instrs = staff.get("instrument", [])
    parts = staff.get("part", [])
    tones = staff.get("tone", [])

    instrs = [str(i).strip().lower() for i in instrs if i and str(i).lower() != "none"]
    parts = [str(p).strip() for p in parts if p is not None and str(p).lower() != "none"]
    tones = [str(t).strip() for t in tones if t and str(t).lower() != "none"]

    try:
        parts_sorted = sorted(parts, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 999)
    except Exception:
        parts_sorted = sorted(parts)

    n = max(len(instrs), len(parts_sorted), len(tones), 0)

    triplets = []
    if n > 0:
        norm_instrs = instrs * n if len(instrs) == 1 and n > 1 else instrs
        norm_tones  = tones  * n if len(tones)  == 1 and n > 1 else tones

        for i in range(n):
            ins = norm_instrs[i] if i < len(norm_instrs) else ""
            prt = parts_sorted[i] if i < len(parts_sorted) else ""
            tne = norm_tones[i] if i < len(norm_tones) else ""
            triplets.append((ins, prt, tne))

    return triplets


def main(input_folder):
    input_folder = input_folder.rstrip("/\\")

    matches = glob.glob(os.path.join(input_folder, "*_staffgroup.json"))
    if not matches:
        print(f"No *_staffgroup.json found in {input_folder}")
        sys.exit(1)
    json_file = matches[0]

    stem = os.path.basename(json_file)[:-len("_staffgroup.json")]
    output_csv = os.path.join(input_folder, f"{stem}_trans_gpt.csv")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "pages" not in data or not data["pages"]:
        print("No pages found in JSON.")
        sys.exit(1)

    staves = data["pages"][0].get("staves", [])

    fieldnames = [
        "page", "staff", "system", "staffgroup", "num_ins", "ens",
        "ins1", "part1", "tone1",
        "ins2", "part2", "tone2",
        "ins3", "part3", "tone3",
        "ocr"
    ]

    all_rows = []
    for i, staff in enumerate(staves):
        triplets = process_staff(staff)

        row = {
            "page": 1 if i == 0 else "",
            "staff": staff.get("staff_id", ""),
            "system": staff.get("system_id", ""),
            "staffgroup": staff.get("staff_group_id", ""),
            "num_ins": len(triplets) if len(triplets) > 1 else "",
            "ens": "",
            "ocr": staff.get("ocr", "")
        }

        for j in range(1, 4):
            if j <= len(triplets):
                ins, prt, tne = triplets[j - 1]
                row[f"ins{j}"] = ins
                row[f"part{j}"] = prt
                row[f"tone{j}"] = tne
            else:
                row[f"ins{j}"] = ""
                row[f"part{j}"] = ""
                row[f"tone{j}"] = ""

        all_rows.append(row)

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved: {output_csv}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python b_trans_gpt.py <input_folder>")
        sys.exit(1)
    main(sys.argv[1])
