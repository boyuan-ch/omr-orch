# env: intern
import argparse
import base64
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

TEXT_PROMPT = """
You are an expert Musicologist and an Optical Music Recognition (OMR) post-processing agent.
Your task is to classify a list of OCR text tokens extracted from the left margin of an orchestral score.
After classification, you must perform additional post-processing to normalize and complete the musical information.

You will receive a list of strings (tokens). You must analyze each token in the context of musical instrumentation and classify it into one of the following 6 categories.

### Categories
0: **Instrument**
   - Names of instruments (e.g., "Violino", "Flauti", "Cor", "Hrn", "Trombe", "C.B.").
   - Can be in Italian, German, French, or English.
   - Includes standard abbreviations.
   - After classification, convert to a standardized English instrument name (e.g., "Flauti" → "Flute", "Fg." → "Bassoon").
   - The list is the standardized English instrument names you must use if is one of the following, For those not in list, you should write the most common English instrument name.
        "Violin", "Viola", "Cello", "Bass", "Flute", "Oboe", "Clarinet", "Bassoon", "Horn", "Trumpet", "Trombone", "Tuba", "Timpani", "Percussion", "Piano", "Harp", "Organ", "Saxophone", "Guitar", "Choir"
   - If multiple instruments are indicated (e.g., "tromboni e tuba"), separate them into individual instruments with comma.

1: **Part of Instrument**
   - Numbers or Roman numerals indicating players or voices.
   - Examples: "1", "2", "I", "II", "1-2", "III".
   - Normalize to numeric form after classification:
     - Roman numerals → integers
     - Ranges ("1-2") → list of integers (e.g., [1,2]).

2: **Tone of Instrument**
   - Key signatures, solfège labels, transposition indicators:
     Examples: "F", "Fa", "B", "Sib", "Es", "Mib", "in A".
   - Normalize to a standard tone label **using English text for accidentals**:
     - "Sib" → "B flat"
     - "Mib" → "E flat"
     - "Es" → "E flat"
     - "Fis" → "F sharp"
     - "A♭" → "A flat"
   - Always output "flat"/"sharp" (never musical symbols).

3: **Ensemble Instruction**
   - Words indicating divisi, unison, solo, grouping:
     Examples: "divisi", "div.", "unis.", "tutti", "solo", "a 2", "a 3".
   - Normalize to English canonical form:
     - "div." → "divisi"
     - "unis." → "unison"
     - "a 2" → "a2"

4: **None of the above**
   - Garbage tokens, punctuation, connectors, or OCR errors.
   - Includes: "in", "e", "and", ".", ":", "-".
   - No further processing required.

5: **Mixture of above**
   - Tokens that contain multiple pieces of information fused together.
     Examples:
     - "Clarinetti in B" (Instrument + Tone)
     - "Ob. 1" (Instrument + Part)
     - "Vln. div." (Instrument + Instruction)
   - You must split these tokens into meaningful sub-tokens, re-classify each sub-token,
     and then output the merged normalized information:
     Example:
       "Clarinetti in B"
         → Instrument: "Clarinet"
         → Tone: "B flat" (if applicable)

### Further Processing After Classification (CRITICAL)

For each token, depending on the class_id, perform:

#### A) Instrument normalization (class 0)
   - Expand abbreviations ("Cl." → "Clarinet").
   - Fix OCR issues ("Vla." → "Viola").
   - Output field name: `"Instrument"`.

#### B) Part normalization (class 1)
   - Convert Roman numerals to numbers.
   - Convert ranges (1-2 → [1,2]).
   - Output field name: `"Parts"`.

#### C) Tone normalization (class 2)
   - Convert European tonal labels to English letter names.
   - Convert accidentals to English text ("flat", "sharp").
   - Output field name: `"Tone"`.

#### D) Ensemble instruction normalization (class 3)
   - Normalize abbreviations.
   - Output field name: `"Instruction"`.

#### E) No processing (class 4)
   - No added fields.

#### F) Mixed token processing (class 5)
   - Split the token into smaller components.
   - Re-classify each sub-token.
   - Apply all normalizations recursively.
   - Combine into a single dictionary with any or all of:
     `"Instrument"`, `"Parts"`, `"Tone"`, `"Instruction"`.

### Output Requirements (CRITICAL)
1. You must classify EVERY single token in the input list.
2. For every token, you must always output all of the following fields: "token", "class_id", "Instrument", "Tone", "Instruction", "Parts". If a field is not applicable to this token, set it to null. For example, if the token has no Parts information, use "Parts": null.
3. The output must include classification AND post-processing info.
4. Output Format:
   {"tokens": [
      { "token": "token1", "class_id": classID1, ... },
      { "token": "token2", "class_id": classID2, ... }
   ]}
5. Final output must be valid JSON.

### Example
Input: ['Clarinetti', 'in', 'B', '1-2','div.','Clarinetti in B']
Output:
{
"tokens": [
    { "token": "Clarinetti", "class_id": 0, "Instrument": "Clarinet", "Tone": null, "Instruction": null, "Parts": null },
    { "token": "in",         "class_id": 4, "Instrument": null,      "Tone": null, "Instruction": null, "Parts": null },
    { "token": "B",          "class_id": 2, "Instrument": null,      "Tone": "B flat", "Instruction": null, "Parts": null },
    { "token": "1-2",        "class_id": 1, "Instrument": null,      "Tone": null, "Instruction": null, "Parts": [1,2] },
    { "token": "div.",       "class_id": 3, "Instrument": null,      "Tone": null, "Instruction": "divisi", "Parts": null },
    { "token": "Clarinetti in B", "class_id": 5, "Instrument": "Clarinet", "Tone": "B flat", "Instruction": null, "Parts": null }
]
}

### Input Data
List:
"""

CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "tokens": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "token":       {"type": "string"},
                    "class_id":    {"type": "integer", "minimum": 0, "maximum": 5},
                    "Instrument":  {"type": "string", "nullable": True},
                    "Tone":        {"type": "string", "nullable": True},
                    "Instruction": {"type": "string", "nullable": True},
                    "Parts":       {"type": "array", "items": {"type": "integer"}, "nullable": True},
                },
                "required": ["token", "class_id", "Instrument", "Tone", "Instruction", "Parts"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tokens"],
    "additionalProperties": False,
}


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def process_json(json_path: str) -> None:
    folder = os.path.dirname(json_path)
    base   = os.path.splitext(os.path.basename(json_path))[0]  # e.g. stem_ocr_filtered

    with open(json_path, "r") as f:
        ocr_data = json.load(f)
    filtered_texts = ocr_data["filtered_text"]

    response = client.responses.create(
        model="gpt-5.1",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": TEXT_PROMPT + str(filtered_texts)},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "omr_token_classification",
                "schema": CLASSIFICATION_SCHEMA,
                "strict": True,
            }
        },
    )

    omr_result = json.loads(response.output_text)

    output_json_path = os.path.join(folder, f"{base}_classified_normalized.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(omr_result, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_json_path}")


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
    parser = argparse.ArgumentParser(description="Classify filtered OCR tokens via GPT.")
    parser.add_argument("input_folder", help="Path to folder containing *_ocr_filtered.json files")
    args = parser.parse_args()
    main(args.input_folder)
