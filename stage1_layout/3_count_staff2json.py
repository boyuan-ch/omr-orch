# env: stave
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
from count_staff_yolo import count_staves_from_yolo

SUPPORTED_IMG_EXTS = (".png", ".jpg", ".jpeg")


def _find_image(folder: str, stem: str) -> str | None:
    for ext in SUPPORTED_IMG_EXTS:
        candidate = os.path.join(folder, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def process_txt(txt_path: str) -> None:
    folder = os.path.dirname(txt_path)
    stem   = os.path.splitext(os.path.basename(txt_path))[0]

    img_path = _find_image(folder, stem)
    if img_path is None:
        print(f"[SKIP] no image found for {stem}")
        return

    (
        n_staves,
        staff_measure_counts,
        staff_height,
        avg_start_x,
        staff_y_centers,
        yolo_boxes,
    ) = count_staves_from_yolo(
        yolo_txt_path=txt_path,
        img_path=img_path,
        target_label=1,
        black_ratio_threshold=0.02,
    )

    output_dict = {
        "filename": stem,
        "n_staves": n_staves,
        "staff_measure_counts": staff_measure_counts,
        "staff_height": float(staff_height),
        "avg_start_x": float(avg_start_x),
        "staff_y_centers": [float(y) for y in staff_y_centers],
        "yolo_boxes": yolo_boxes,
    }

    json_path = os.path.join(folder, f"{stem}_yolostaff.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(output_dict, jf, indent=4)

    print(f"Saved: {json_path}  (n_staves={n_staves})")


def main(input_folder: str) -> None:
    if not os.path.isdir(input_folder):
        raise ValueError(f"Not a directory: {input_folder}")

    stem     = os.path.basename(input_folder.rstrip("/\\"))
    txt_path = os.path.join(input_folder, f"{stem}.txt")

    if not os.path.exists(txt_path):
        print(f"No '{stem}.txt' found in {input_folder}")
        return

    print(f"Processing: {txt_path}")
    try:
        process_txt(txt_path)
    except Exception as e:
        print(f"Error: {e}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count staves from YOLO txt files in a folder.")
    parser.add_argument("input_folder", help="Path to folder containing .txt and image files")
    args = parser.parse_args()
    main(args.input_folder)
