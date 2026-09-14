# env: stave
import argparse
import os
import cv2

SUPPORTED_IMG_EXTS = (".png", ".jpg", ".jpeg")


def draw_boxes(img_path: str, txt_path: str, save_path: str) -> None:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read image: {img_path}")
        return

    h, w = img.shape[:2]

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        if int(parts[0]) != 1:
            continue  # only draw label == 1

        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(save_path, img)
    print(f"Saved: {save_path}")


def _find_image(folder: str, stem: str) -> str | None:
    for ext in SUPPORTED_IMG_EXTS:
        candidate = os.path.join(folder, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def main(input_folder: str) -> None:
    if not os.path.isdir(input_folder):
        raise ValueError(f"Not a directory: {input_folder}")

    stem     = os.path.basename(input_folder.rstrip("/\\"))
    txt_path = os.path.join(input_folder, f"{stem}.txt")

    if not os.path.exists(txt_path):
        print(f"No '{stem}.txt' found in {input_folder}")
        return

    img_path = _find_image(input_folder, stem)
    if img_path is None:
        print(f"[SKIP] no image named '{stem}.*' found in {input_folder}")
        return

    save_path = os.path.join(input_folder, f"{stem}_box_plot.png")
    try:
        draw_boxes(img_path, txt_path, save_path)
    except Exception as e:
        print(f"Error: {e}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw YOLO staff boxes on images in a folder.")
    parser.add_argument("input_folder", help="Path to folder containing images and .txt files")
    args = parser.parse_args()
    main(args.input_folder)
