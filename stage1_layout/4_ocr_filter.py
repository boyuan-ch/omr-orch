# env: stave
import argparse
import json
import os


def doctr_bbox_to_yolo(bbox):
    """Convert DocTR minmax bbox → YOLO (xc, yc, w, h)."""
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width    = x_max - x_min
    height   = y_max - y_min
    return (x_center, y_center, width, height)


def run_filter_ocr(
    staff_json_path: str,
    staff_json_gt_path: str,
    ocr_json_path: str,
    output_json_path: str,
    top_param: float = 1.5,
    bottom_param: float = 0.5,
    conf_threshold: float = 0.2,
) -> None:
    with open(staff_json_path, "r") as f:
        staff_data = json.load(f)
    with open(staff_json_gt_path, "r") as f:
        staff_data_gt = json.load(f)

    n_staves            = staff_data_gt["n_staves"]
    staff_y_centers     = staff_data_gt["staff_y_centers"]
    staff_height        = staff_data["staff_height"]
    avg_start_x         = staff_data["avg_start_x"]
    staff_measure_counts = staff_data.get("staff_measure_counts", [])

    if staff_measure_counts and n_staves > 0:
        avg_measures_per_staff = sum(staff_measure_counts) / n_staves
        avg_measure_width = (1.0 - avg_start_x) / avg_measures_per_staff
    else:
        avg_measure_width = 0.2

    limit_top    = staff_y_centers[0] - staff_height * top_param
    limit_bottom = staff_y_centers[-1] + staff_height * bottom_param
    limit_left   = 0.0
    limit_right  = avg_start_x + avg_measure_width

    with open(ocr_json_path, "r") as f:
        ocr_data = json.load(f)

    filename       = ocr_data["filename"]
    ocr_texts      = ocr_data["ocr_text"]
    ocr_boxes      = ocr_data["ocr_box_minmax"]
    ocr_confidence = ocr_data["ocr_confidence"]

    yolo_bboxes = [doctr_bbox_to_yolo(b) for b in ocr_boxes]

    filtered_texts      = []
    filtered_yolo_bboxes = []
    filtered_confidence = []

    for text, yolo, conf in zip(ocr_texts, yolo_bboxes, ocr_confidence):
        x_center, y_center, _, _ = yolo
        if (limit_left <= x_center <= limit_right) and (limit_top <= y_center <= limit_bottom):
            if conf > conf_threshold or any(char in text.lower() for char in ["l", "1", "i"]):
                filtered_texts.append(text)
                filtered_yolo_bboxes.append([float(v) for v in yolo])
                filtered_confidence.append(conf)

    output_dict = {
        "filename": filename,
        "filtered_text": filtered_texts,
        "filtered_yolo_bbox": filtered_yolo_bboxes,
        "filtered_confidence": filtered_confidence,
        "debug_info": {
            "limit_right": limit_right,
            "avg_measure_width": avg_measure_width,
        },
    }

    with open(output_json_path, "w") as f:
        json.dump(output_dict, f, indent=4)

    print(f"Saved: {output_json_path}  ({len(filtered_texts)}/{len(ocr_texts)} tokens kept, width={avg_measure_width:.3f})")


def process_staff_json(staff_json_path: str) -> None:
    folder = os.path.dirname(staff_json_path)
    stem   = os.path.basename(staff_json_path).replace("_yolostaff.json", "")

    staff_json_gt_path = os.path.join(folder, f"{stem}_yolostaff_gt.json")
    ocr_json_path      = os.path.join(folder, f"{stem}_ocr.json")
    output_json_path   = os.path.join(folder, f"{stem}_ocr_filtered.json")

    if not os.path.exists(staff_json_gt_path):
        print(f"[SKIP] GT staff JSON not found: {staff_json_gt_path}")
        return
    if not os.path.exists(ocr_json_path):
        print(f"[SKIP] OCR JSON not found: {ocr_json_path}")
        return

    run_filter_ocr(staff_json_path, staff_json_gt_path, ocr_json_path, output_json_path)


def main(input_folder: str) -> None:
    if not os.path.isdir(input_folder):
        raise ValueError(f"Not a directory: {input_folder}")

    staff_files = sorted(
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith("_yolostaff.json") and not f.endswith("_gt.json")
    )

    if not staff_files:
        print(f"No *_yolostaff.json files found in {input_folder}")
        return

    for staff_json_path in staff_files:
        print(f"Processing: {staff_json_path}")
        try:
            process_staff_json(staff_json_path)
        except Exception as e:
            print(f"Error: {e}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter OCR tokens by staff spatial bounds.")
    parser.add_argument("input_folder", help="Path to folder containing *_yolostaff.json, *_yolostaff_gt.json, and *_ocr.json")
    args = parser.parse_args()
    main(args.input_folder)
