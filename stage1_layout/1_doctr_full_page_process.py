# env: ocrt
import os
import json
import argparse
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg")

def run_doctr_ocr(model, image_path):
    """
    Run DocTR OCR on an image and save results to a JSON file.

    Args:
        model: The OCR model to use.
        image_path (str): Path to input image.
        output_json_path (str): The full output JSON file path.

    Returns:
        output_json_path (str): Path to saved JSON file.
    """

    # 讀取圖片
    doc = DocumentFile.from_images(image_path)

    # OCR 推論
    result = model(doc)
    pages = result.pages

    ocr_text_list = []
    ocr_box_minmax_list = []
    ocr_conf_list = []

    # 解析 OCR 結果
    for page in pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:

                    text = word.value

                    # DocTR geometry format:
                    # geometry = ((xmin, ymin), (xmax, ymax))
                    (xmin, ymin), (xmax, ymax) = word.geometry

                    ocr_text_list.append(text)
                    ocr_conf_list.append(word.confidence)

                    ocr_box_minmax_list.append([
                        float(xmin),
                        float(ymin),
                        float(xmax),
                        float(ymax)
                    ])

    return ocr_text_list, ocr_box_minmax_list, ocr_conf_list

def process_image(model, img_path: str) -> None:
    folder = os.path.dirname(img_path)
    base = os.path.splitext(os.path.basename(img_path))[0]

    ocr_text_list, ocr_box_minmax_list, ocr_conf_list = run_doctr_ocr(model, img_path)

    output_dict = {
        "filename": base,
        "ocr_text": ocr_text_list,
        "ocr_box_minmax": ocr_box_minmax_list,
        "ocr_confidence": ocr_conf_list,
    }

    json_path = os.path.join(folder, f"{base}_ocr.json")
    with open(json_path, "w") as jf:
        json.dump(output_dict, jf, indent=4)

    print(f"Saved: {json_path}")


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

    model = ocr_predictor(pretrained=True)
    print(f"Processing: {img_path}")
    process_image(model, img_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run docTR OCR on images in a folder.")
    parser.add_argument("input_folder", help="Path to the folder containing image(s)")
    args = parser.parse_args()
    main(args.input_folder)
