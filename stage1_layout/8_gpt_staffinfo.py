import base64
import json
import os
import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg")

TEXT_PROMPT = """
You are an expert Musicologist and OMR (Optical Music Recognition) specialist.
Your task is to analyze orchestral score images and extract structured data about the staves, including their layout hierarchy (System and Group).

**Definitions for Layout Analysis:**
1.  **System (`system_id`):** A "System" is a distinct block of music consisting of multiple staves playing the same measures simultaneously.
    -   Systems are separated by significant horizontal whitespace.
    -   If a page contains two blocks of music (e.g., measures 1-10 at the top, 11-20 at the bottom), the top block is `system_id: 1`, and the bottom block is `system_id: 2`.
2.  **Staff Group (`staff_group_id`):** A "Group" consists of adjacent staves connected by a continuous vertical barline or bracket on the left margin (e.g., Woodwinds, Brass, Percussion, Strings).
    -   **Logic:** Start with `staff_group_id: 1`. When the vertical connecting line on the left breaks, increment the `staff_group_id`.
    -   **Reset:** `staff_group_id` must reset to 1 at the beginning of a new `system_id`.
3.  **Staff ID (`staff_id`):** A unique identifier for each staff line on the page, incrementing continuously from top to bottom (does NOT reset between systems).

**Extraction Rules:**
1.  **Identify Instruments:** Read the text at the left margin (OCR). Translate to English (e.g., 'Corni' -> 'Horn').
2.  **Tonality:** Extract key/transposition info (e.g., 'in B' -> 'B flat', 'Es' -> 'E flat').
3.  **Parts:** Extract part numbers (e.g., '1-2' -> [1, 2]).
4.  **Handling "None":** If a staff has no label (common in subsequent systems), infer it if possible or mark as "None".
5.  **Multi-Instrument Staves:** If a staff contains multiple instruments (e.g., "Violoncello e Basso"), list both in `instrument`.

**Output Format:**
Return a SINGLE valid JSON object containing a "pages" list.
The JSON structure must match the example below EXACTLY.
If no staves are detected (like cover page), return an empty "staves" list.

**Example JSON Output:**
{
"pages": [
    {
    "image_index": 1,
    "filename": "Tchai_4_1.png",
    "staves": [
        {"staff_id": 1, "staff_group_id": 1, "system_id": 1, "ocr": "Flauti", "instrument": ["Flute"], "tone": [], "part": []},
        {"staff_id": 2, "staff_group_id": 1, "system_id": 1, "ocr": "Oboi", "instrument": ["Oboe"], "tone": [], "part": []},
        {"staff_id": 3, "staff_group_id": 1, "system_id": 1, "ocr": "Clarinett i in B.", "instrument": ["Clarinet"], "tone": ["B flat"], "part": []},
        {"staff_id": 4, "staff_group_id": 1, "system_id": 1, "ocr": "Fagotti", "instrument": ["Bassoon"], "tone": [], "part": []},
        {"staff_id": 5, "staff_group_id": 1, "system_id": 1, "ocr": "Corni in Es.", "instrument": ["Horn"], "tone": ["E flat"], "part": []},
        {"staff_id": 6, "staff_group_id": 1, "system_id": 1, "ocr": "Corno 3º in Es.", "instrument": ["Horn"], "tone": ["E flat"], "part": [3]},
        {"staff_id": 7, "staff_group_id": 1, "system_id": 1, "ocr": "Trombe in Es.", "instrument": ["Trumpet"], "tone": ["E flat"], "part": []},
        {"staff_id": 8, "staff_group_id": 2, "system_id": 1, "ocr": "Timpani in Es. B.", "instrument": ["Timpani"], "tone": ["E flat", "B flat"], "part": []},
        {"staff_id": 9, "staff_group_id": 3, "system_id": 1, "ocr": "Violino I.", "instrument": ["Violin"], "tone": [], "part": [1]},
        {"staff_id": 10, "staff_group_id": 3, "system_id": 1, "ocr": "Violino II.", "instrument": ["Violin"], "tone": [], "part": [2]},
        {"staff_id": 11, "staff_group_id": 3, "system_id": 1, "ocr": "Viola.", "instrument": ["Viola"], "tone": [], "part": []},
        {"staff_id": 12, "staff_group_id": 3, "system_id": 1, "ocr": "Violoncello e Basso.", "instrument": ["Cello", "Bass"], "tone": [], "part": []},
        {"staff_id": 13, "staff_group_id": 1, "system_id": 2, "ocr": "Flauti", "instrument": ["Flute"], "tone": [], "part": []},
        {"staff_id": 14, "staff_group_id": 1, "system_id": 2, "ocr": "Oboi", "instrument": ["Oboe"], "tone": [], "part": []},
        {"staff_id": 15, "staff_group_id": 1, "system_id": 2, "ocr": "Clarinett i in B.", "instrument": ["Clarinet"], "tone": ["B flat"], "part": []},
        {"staff_id": 16, "staff_group_id": 1, "system_id": 2, "ocr": "Fagotti", "instrument": ["Bassoon"], "tone": [], "part": []},
        {"staff_id": 17, "staff_group_id": 1, "system_id": 2, "ocr": "Corni in Es.", "instrument": ["Horn"], "tone": ["E flat"], "part": []},
        {"staff_id": 18, "staff_group_id": 1, "system_id": 2, "ocr": "Corno 3º in Es.", "instrument": ["Horn"], "tone": ["E flat"], "part": [3]},
        {"staff_id": 19, "staff_group_id": 1, "system_id": 2, "ocr": "Trombe in Es.", "instrument": ["Trumpet"], "tone": ["E flat"], "part": []},
        {"staff_id": 20, "staff_group_id": 2, "system_id": 2, "ocr": "Timpani in Es. B.", "instrument": ["Timpani"], "tone": ["E flat", "B flat"], "part": []},
        {"staff_id": 21, "staff_group_id": 3, "system_id": 2, "ocr": "Violino I.", "instrument": ["Violin"], "tone": [], "part": [1]},
        {"staff_id": 22, "staff_group_id": 3, "system_id": 2, "ocr": "Violino II.", "instrument": ["Violin"], "tone": [], "part": [2]},
        {"staff_id": 23, "staff_group_id": 3, "system_id": 2, "ocr": "None", "instrument": ["None"], "tone": [], "part": []},
        {"staff_id": 24, "staff_group_id": 3, "system_id": 2, "ocr": "Violoncello e Basso.", "instrument": ["Cello", "Bass"], "tone": [], "part": []}
    ]
    }
]
}
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "image_index": {"type": "integer"},
                    "filename": {"type": "string"},
                    "staves": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "staff_id": {"type": "integer"},
                                "staff_group_id": {"type": "integer"},
                                "system_id": {"type": "integer"},
                                "ocr": {"type": "string"},
                                "instrument": {"type": "array", "items": {"type": "string"}},
                                "tone": {"type": "array", "items": {"type": "string"}},
                                "part": {"type": "array", "items": {"type": "integer"}},
                            },
                            "required": ["staff_id", "staff_group_id", "system_id", "ocr", "instrument", "tone", "part"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["image_index", "filename", "staves"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["pages"],
    "additionalProperties": False,
}


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def process_image(img_path: str) -> None:
    folder = os.path.dirname(img_path)
    img_filename = os.path.basename(img_path)
    base = os.path.splitext(img_filename)[0]

    base64_image = encode_image(img_path)

    response = client.responses.create(
        model="gpt-5.1",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": TEXT_PROMPT + f"\nProcess the image: {img_filename}"},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "omr_staff_schema",
                "schema": JSON_SCHEMA,
                "strict": True,
            }
        },
    )

    omr_result = json.loads(response.output_text)

    output_json_path = os.path.join(folder, f"{base}_staffgroup.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(omr_result, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_json_path}")


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

    print(f"Processing: {img_path}")
    try:
        process_image(img_path)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT staff info extraction on images in a folder.")
    parser.add_argument("input_folder", help="Path to the folder containing image(s)")
    args = parser.parse_args()
    main(args.input_folder)
