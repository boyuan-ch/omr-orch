"""
2_cnt_staff_omr.py - Run OMR staff detection on images in a folder.

For each image found in the input folder, runs the unet_big model to detect
staves and saves a {stem}_stafflist.pkl to the same folder.

Usage:
    python 2_cnt_staff_omr.py <input_folder> [--batch-size N]

Speed tips:
    - Install onnxruntime-gpu for CUDA acceleration (replaces onnxruntime).
    - Increase --batch-size (e.g. 32 or 64) when running on GPU.
"""

import argparse
import os
import pickle
import statistics
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import onnxruntime as rt

from omr import MODULE_PATH
from omr.staffline_extraction import staff_extract_staffobj

MIN_BAR_HEIGHT = 8
RESIZE_RATIO = 2

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".tiff")


# ---------- minimal Staff class (mirrors pdf2musicXML_yolo.Staff) ----------

class Staff:
    def __init__(self, left: int, right: int, ys: Tuple[int, int, int, int, int], minMaxDiff: int = 0):
        self.left = left
        self.right = right
        self.ys = ys
        self.minDiff = minMaxDiff

    def get_yOne_float(self) -> float:
        x = [self.ys[i + 1] - self.ys[i] for i in range(len(self.ys) - 1)]
        return sum(x) / len(x)


# ---------- staff list helpers ------------------------------------------------

def _staffs_to_omrStaffList(staffs) -> Tuple[List[Staff], list]:
    left = min([sf.x_left for sf in staffs[0, :]])
    right = max([sf.x_right for sf in staffs[-1, :]])
    omrstaff_list: List[Staff] = []
    staffRange = []
    for i in range(staffs.shape[1]):
        yuppers = [sf.y_upper for sf in staffs[:, i]]
        ybottoms = [sf.y_lower for sf in staffs[:, i]]
        top = statistics.median(yuppers)
        bottom = statistics.median(ybottoms)
        unit_size = statistics.median([sf.unit_size for sf in staffs[:, i]])
        minMaxDiff = min(max(yuppers) - min(yuppers), max(ybottoms) - min(ybottoms))
        ys = [int(top), int(top + unit_size), int((top + bottom) / 2), int(bottom - unit_size), int(bottom)]
        omrstaff_list.append(Staff(int(left), int(right), ys, minMaxDiff))
        staffRange.append(range(int(top), int(bottom)))
    return omrstaff_list, staffRange


# ---------- single-model inference -------------------------------------------

def _resize_image(image: Image.Image, lower: int = 3_000_000, upper: int = 4_350_000) -> Image.Image:
    w, h = image.size
    pix = w * h
    if lower <= pix <= upper:
        return image
    ratio = ((lower / pix + upper / pix) / 2) ** 0.5
    return image.resize((round(ratio * w), round(ratio * h)))


def _run_first_model(img_path: str, batch_size: int = 16) -> np.ndarray:
    """Run only the unet_big model and return the staff prediction map."""
    model_path = os.path.join(MODULE_PATH, "checkpoints/unet_big")
    onnx_path = os.path.join(model_path, "model.onnx")
    metadata = pickle.load(open(os.path.join(model_path, "metadata.pkl"), "rb"))

    providers = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    try:
        rt.set_default_logger_severity(3)
    except AttributeError:
        pass  # removed in onnxruntime-gpu >= 1.18
    sess = rt.InferenceSession(onnx_path, providers=providers)
    print("Providers in use:", sess.get_providers())

    output_names = metadata['output_names']
    input_shape  = metadata['input_shape']
    output_shape = metadata['output_shape']

    image_cv = cv2.imread(img_path)
    image_pil = Image.fromarray(image_cv).convert("RGB")
    image_pil = _resize_image(image_pil)
    image = np.array(image_pil)

    win_size  = input_shape[1]
    step_size = 128
    patches = []
    coords  = []
    for y in range(0, image.shape[0], step_size):
        y = min(y, image.shape[0] - win_size)
        for x in range(0, image.shape[1], step_size):
            x = min(x, image.shape[1] - win_size)
            patches.append(image[y:y + win_size, x:x + win_size])
            coords.append((y, x))

    pred_patches = []
    total = len(patches)
    for i in range(0, total, batch_size):
        print(f"{i+1}/{total} (batch {batch_size})", end="\r")
        batch = np.array(patches[i:i + batch_size])
        out = sess.run(output_names, {'input': batch})[0]
        pred_patches.append(out)
    print()

    full_out  = np.zeros(image.shape[:2] + (output_shape[-1],), dtype=np.float32)
    full_mask = np.zeros(image.shape[:2] + (output_shape[-1],), dtype=np.float32)
    for idx, (y, x) in enumerate(coords):
        bi, ri = divmod(idx, batch_size)
        hop = pred_patches[bi][ri]
        full_out [y:y + win_size, x:x + win_size] += hop
        full_mask[y:y + win_size, x:x + win_size] += 1

    full_out /= full_mask
    class_map = np.argmax(full_out, axis=-1)
    staff = np.where(class_map == 1, 1, 0)

    staff = cv2.resize(staff.astype(np.uint8), None, fx=RESIZE_RATIO, fy=RESIZE_RATIO,
                       interpolation=cv2.INTER_NEAREST)
    return staff


# ---------- public API --------------------------------------------------------

def extract_staff(img_path: str, batch_size: int = 16) -> Tuple[int, List[Staff]]:
    """Return (staff_count, staff_list) for one image."""
    staff_map = _run_first_model(img_path, batch_size=batch_size)
    staffs, _ = staff_extract_staffobj(staff_map, MIN_BAR_HEIGHT)
    staff_list, _ = _staffs_to_omrStaffList(staffs)
    return len(staff_list), staff_list


def process_image(img_path: str, batch_size: int = 16) -> None:
    folder = os.path.dirname(img_path)
    stem = os.path.splitext(os.path.basename(img_path))[0]

    count, staff_list = extract_staff(img_path, batch_size=batch_size)

    pkl_path = os.path.join(folder, f"{stem}_stafflist.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(staff_list, f)

    print(f"n_staves={count}  saved: {pkl_path}")


def _find_target_image(input_folder: str) -> str | None:
    stem = os.path.basename(input_folder.rstrip("/\\"))
    for ext in SUPPORTED_EXTS:
        candidate = os.path.join(input_folder, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def main(input_folder: str, batch_size: int = 16) -> None:
    if not os.path.isdir(input_folder):
        raise ValueError(f"Not a directory: {input_folder}")

    img_path = _find_target_image(input_folder)
    if img_path is None:
        stem = os.path.basename(input_folder.rstrip("/\\"))
        print(f"No image named '{stem}.*' found in {input_folder}")
        return

    print(f"\nProcessing: {img_path}")
    try:
        process_image(img_path, batch_size=batch_size)
    except Exception as e:
        print(f"Error: {e}")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OMR staff detection on images in a folder.")
    parser.add_argument("input_folder", help="Path to the folder containing image(s)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Inference batch size (increase for GPU, e.g. 32 or 64)")
    args = parser.parse_args()
    main(args.input_folder, batch_size=args.batch_size)
