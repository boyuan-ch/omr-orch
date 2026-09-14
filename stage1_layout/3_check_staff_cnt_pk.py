"""
3_check_staff_cnt_pk.py - Convert _stafflist.pkl files to _yolostaff.json.

Reads *_stafflist.pkl files from the input folder, finds the matching image
in the same folder, and writes {stem}_yolostaff.json alongside them.

Run AFTER 2_cnt_staff_omr.py.

Usage:
    python 3_check_staff_cnt_pk.py <input_folder>
"""

import argparse
import json
import os
import pickle
from typing import Tuple

from PIL import Image

# Inline Staff class and RESIZE_RATIO so this file is self-contained
# (mirrors the definitions in 2_cnt_staff_omr.py)
RESIZE_RATIO = 2
LOWER, UPPER = 3_000_000, 4_350_000

SUPPORTED_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tiff")


class Staff:
    def __init__(self, left, right, ys, minMaxDiff=0):
        self.left = left
        self.right = right
        self.ys = ys
        self.minDiff = minMaxDiff


def _compute_effective_h(img_w: int, img_h: int) -> int:
    """Mirrors _resize_image + RESIZE_RATIO to get the height the model worked in."""
    pix = img_w * img_h
    if LOWER <= pix <= UPPER:
        resized_h = img_h
    else:
        ratio = ((LOWER / pix + UPPER / pix) / 2) ** 0.5
        resized_h = round(ratio * img_h)
    return resized_h * RESIZE_RATIO


def _find_image(folder: str, stem: str) -> str | None:
    for ext in SUPPORTED_IMG_EXTS:
        candidate = os.path.join(folder, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def process_pkl(pkl_path: str) -> None:
    folder = os.path.dirname(pkl_path)
    stem = os.path.basename(pkl_path).replace("_stafflist.pkl", "")

    img_path = _find_image(folder, stem)
    if img_path is None:
        print(f"[SKIP] no image found for {stem} in {folder}")
        return

    with open(pkl_path, "rb") as f:
        staff_list = pickle.load(f)

    img_w, img_h = Image.open(img_path).size
    effective_h = _compute_effective_h(img_w, img_h)

    staff_y_centers = [
        (s.ys[0] + s.ys[4]) / 2 / effective_h
        for s in staff_list
    ]

    out = {
        "filename": stem,
        "n_staves": len(staff_list),
        "staff_y_centers": staff_y_centers,
    }

    out_path = os.path.join(folder, f"{stem}_yolostaff_gt.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=4)

    print(f"{stem}  ->  n_staves={len(staff_list)}  saved: {os.path.basename(out_path)}")


def main(input_folder: str) -> None:
    if not os.path.isdir(input_folder):
        raise ValueError(f"Not a directory: {input_folder}")

    pkl_files = sorted(
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith("_stafflist.pkl")
    )

    if not pkl_files:
        print(f"No *_stafflist.pkl files found in {input_folder}")
        return

    for pkl_path in pkl_files:
        try:
            process_pkl(pkl_path)
        except Exception as e:
            print(f"Error processing {pkl_path}: {e}")

    print(f"\nDone. {len(pkl_files)} file(s) processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert _stafflist.pkl files to _yolostaff.json in the same folder."
    )
    parser.add_argument("input_folder", help="Folder containing *_stafflist.pkl and images")
    args = parser.parse_args()
    main(args.input_folder)
