# env: stave
import argparse
import os
import cv2
import glob
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights",
                          "ola-layout-analysis-2.0-2025-03-09.pt")
CONF_THRESHOLD = 0.7
SUPPORTED_EXTS = (".png", ".jpg", ".jpeg")


def process_image(model, img_path: str) -> None:
    results = model.predict(
        source=img_path,
        conf=CONF_THRESHOLD,
        imgsz=1280,
        device="cpu",
        save=False,
        save_txt=False,
    )

    r = results[0]
    boxes   = r.boxes.xyxy.cpu().numpy()
    confs   = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()

    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    folder       = os.path.dirname(img_path)
    stem         = os.path.splitext(os.path.basename(img_path))[0]
    txt_path     = os.path.join(folder, f"{stem}.txt")

    with open(txt_path, "w") as f:
        for (x1, y1, x2, y2), cls, conf in zip(boxes, classes, confs):
            w  = x2 - x1
            h  = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            f.write(f"{int(cls)} {cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f} {conf:.6f}\n")

    print(f"Saved: {txt_path}")


def _find_target_image(input_folder: str) -> str | None:
    stem = os.path.basename(input_folder.rstrip("/\\"))
    for ext in SUPPORTED_EXTS:
        candidate = os.path.join(input_folder, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def main(input_folder: str) -> None:
    if not os.path.isdir(input_folder):
        raise ValueError(f"Not a directory: {input_folder}")

    img_path = _find_target_image(input_folder)
    if img_path is None:
        stem = os.path.basename(input_folder.rstrip("/\\"))
        print(f"No image named '{stem}.*' found in {input_folder}")
        return

    model = YOLO(MODEL_PATH)
    print(f"Processing: {img_path}")
    try:
        process_image(model, img_path)
    except Exception as e:
        print(f"Error: {e}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO staff detection on images in a folder.")
    parser.add_argument("input_folder", help="Path to the folder containing image(s)")
    args = parser.parse_args()
    main(args.input_folder)
