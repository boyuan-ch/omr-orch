# env: stave
import argparse
import json
import os
import cv2

SUPPORTED_IMG_EXTS = (".png", ".jpg", ".jpeg")


def draw_yolo_box(img, box, color=(0, 255, 0), thickness=2):
    """box = [cx, cy, w, h] normalized (0–1)."""
    H, W = img.shape[:2]
    cx, cy, w, h = box
    x1 = int((cx - w / 2) * W)
    y1 = int((cy - h / 2) * H)
    x2 = int((cx + w / 2) * W)
    y2 = int((cy + h / 2) * H)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def _find_image(folder: str, stem: str) -> str | None:
    for ext in SUPPORTED_IMG_EXTS:
        candidate = os.path.join(folder, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def process_json(json_path: str) -> None:
    folder = os.path.dirname(json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    filename   = data["filename"]
    yolo_boxes = data.get("filtered_yolo_bbox", [])

    img_path = _find_image(folder, filename)
    if img_path is None:
        print(f"[SKIP] no image found for {filename}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] cannot read image: {img_path}")
        return

    for box in yolo_boxes:
        draw_yolo_box(img, box)

    save_path = os.path.join(folder, f"{filename}_ocr_filtered.png")
    cv2.imwrite(save_path, img)
    print(f"Saved: {save_path}")


def main(input_folder: str) -> None:
    if not os.path.isdir(input_folder):
        raise ValueError(f"Not a directory: {input_folder}")

    json_files = sorted(
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith("_ocr_filtered.json")
    )

    if not json_files:
        print(f"No *_ocr_filtered.json files found in {input_folder}")
        return

    for json_path in json_files:
        print(f"Processing: {json_path}")
        try:
            process_json(json_path)
        except Exception as e:
            print(f"Error: {e}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot filtered OCR boxes onto images.")
    parser.add_argument("input_folder", help="Path to folder containing *_ocr_filtered.json and images")
    args = parser.parse_args()
    main(args.input_folder)
