"""用開源 VLM（預設 Qwen3-VL-8B-Instruct, FP16）取代 GPT-5.1，
跑 stage1_layout/6_1_gpt_classify_reformat.py 的 OCR token 分類（GPT 版）的開源替代。

輸入：OCR 過濾後的 token 字串 list + 該頁樂譜影像
輸出：每個 token 的 class_id / Instrument / Tone / Instruction / Parts，
      JSON 結構與 GPT 版逐鍵一致。

【與 GPT baseline 的關係】
兩邊都是「圖片 + OCR token 字串」一起送進模型，條件對等。

  註：6_1_gpt_classify_reformat.py 目前的檔案狀態是 input_image 被註解掉的，
  但那是針對某一頁有問題而做的單頁重測時留下的狀態，實際產生絕大多數
  classified 結果的版本是有送圖的（已由本專案作者確認）。
  交接文件 transfer.md §5.4 曾據檔案現狀推論「這一步是純文字」，該推論已更正。

因為有圖片參與，兩種解析度（--res api / high）在這一步都有意義。

輸入資料夾不會被寫入任何東西；輸出全部在 --out-root 底下。

用法：
  python classify_tokens_vlm.py --task Bee_2_challenge --dry-run
  python classify_tokens_vlm.py --task Bee_1_challenge --model internvl3_5 --res api --resume

輸出：
  out/{model_tag}/{res}/{task}_ocr_filtered_classified_normalized/
      {piece}_NNN_ocr_filtered_classified_normalized.json   <- 檔名與 GPT 版逐字一致
      run_manifest.json
      run_log.jsonl
"""
import argparse
import gc
import json
import os
import sys
import time

# 跟 run_staffinfo_vlm.py 同樣的理由：長跑數百頁、輸出長度不斷變動，
# 提早設好可擴展 segment，緩解 CUDA allocator 碎片化導致的 OOM。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Step 6 (token classification) and step 8 (staff groups) use different JSON schemas, so each keeps
# its own parser: classify_json_utils.py here, json_utils.py for run_staffinfo_vlm.py.
from classify_json_utils import empty_result, normalize_tokens, parse_model_json  # noqa: E402
from backends import backend_tag, build_backend, resolve_model_id  # noqa: E402
from gpu_utils import is_oom, load_with_oom_retry, retry_on_oom  # noqa: E402
from image_prep import prepare_image  # noqa: E402

DATASET_ROOT = os.environ.get("OMR_DATASET_ROOT", "data")  # override with --dataset-root
OUT_ROOT = os.path.join(_HERE, "out")

DEFAULT_MODEL = "qwen3"

# Prompt copied verbatim from the GPT version of this step (6_1_gpt_classify_reformat.py).
# 刻意不改寫措辭去提「附圖」——GPT 版本身就是用這段 prompt 搭配圖片送出的，
# 一字不動才能保證兩邊條件對等。
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

# 本地沒有 vLLM / outlines / xgrammar，無法做 constrained decoding（GPT 版用
# json_schema strict）。多加這句提高輸出 JSON 的機率，跟 vlm_staffinfo 的
# TEXT_PROMPT 補充句是同樣的理由。
_JSON_ONLY_SUFFIX = "\n\nOutput ONLY the JSON object. No markdown fences, no explanation, no extra text."


def assert_manifest(out_dir, model_id, res, force=False):
    """跟 run_staffinfo_vlm.py 同一套防呆：這個資料夾先前若是別的 model_id/res
    產生的，直接中止，不要讓 --resume 把不同組合的結果混在一起。"""
    manifest_path = os.path.join(out_dir, "run_manifest.json")
    current = {"model_id": model_id, "res": res}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            prev = json.load(f)
        if (prev.get("model_id"), prev.get("res")) != (model_id, res) and not force:
            raise SystemExit(
                f"[abort] {out_dir} 先前是 {prev.get('model_id')} / res={prev.get('res')} "
                f"產生的，現在要用 {model_id} / res={res} 寫入同一個資料夾。\n"
                f"    1. 換一個 --out-root，或\n"
                f"    2. 確定要覆蓋才加 --force-overwrite\n")
    with open(manifest_path, "w") as f:
        json.dump(current, f, indent=2)


def process_file(backend, ocr_json_path, image_path, res_mode, max_new_tokens, label="",
                 oom_retries=6, oom_wait=60):
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)
    filtered_texts = ocr_data["filtered_text"]

    image, img_meta = prepare_image(image_path, res_mode)
    prompt = TEXT_PROMPT + str(filtered_texts) + _JSON_ONLY_SUFFIX

    # 包一層 OOM 重試：GPU 是跟別人共用的，撞到別人的記憶體尖峰時等一下再試，
    # 不要把本來跑得動的頁面記成 error（見 gpu_utils.retry_on_oom）。
    raw_text, meta = retry_on_oom(
        lambda: backend.generate(image, prompt, max_new_tokens=max_new_tokens),
        max_retries=oom_retries, base_wait=oom_wait, label=label)
    data, status = parse_model_json(raw_text)
    retried = False

    if status == "failed":
        # do_sample=False 是 greedy，同一個 prompt+圖重打一次只會拿到一樣的結果，
        # 所以 retry 時把 prompt 再收緊一次（跟 vlm_staffinfo 的 retry 邏輯同理）。
        retried = True
        raw_text2, meta2 = retry_on_oom(
            lambda: backend.generate(
                image,
                prompt + "\nYour previous output was not valid JSON. "
                         "Reply with the JSON object only.",
                max_new_tokens=max_new_tokens),
            max_retries=oom_retries, base_wait=oom_wait, label=f"{label} (retry)")
        data2, status2 = parse_model_json(raw_text2)
        if status2 != "failed":
            raw_text, meta, data, status = raw_text2, meta2, data2, status2

    if data is None:
        result = empty_result()
        status = "parse_error"
    else:
        result = normalize_tokens(data)

    record = {
        **meta,
        **img_meta,
        "parse_status": status,
        "retried": retried,
        "n_tokens_in": len(filtered_texts),
        "n_tokens_out": len(result["tokens"]),
    }
    return result, raw_text, record


def main():
    global DATASET_ROOT
    parser = argparse.ArgumentParser(
        description="Classify filtered OCR tokens via a local open-source VLM (image + text).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", nargs="+", required=True,
                        help="例如 Bee_2_challenge（可給多個）。OCR token 讀 "
                             f"{DATASET_ROOT}/{{task}}_ocr_filtered/*.json，"
                             f"對應頁面圖讀 {DATASET_ROOT}/{{task}}/*.png")
    parser.add_argument("--model", choices=["qwen3", "internvl3_5"],
                        default=DEFAULT_MODEL,
                        help="模型簡稱（用 --model-id 可覆寫成任何 HF repo）")
    parser.add_argument("--model-id", default=None,
                        help="覆寫 HF repo id。輸出資料夾名稱一律跟著 model_id 走")
    parser.add_argument("--res", choices=["api"], default="api",
                        help="high = 原圖高解析度（預設）；api = 縮成 OpenAI 等效解析度")
    parser.add_argument("--resume", action="store_true", help="跳過已存在的輸出檔")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-pixels", type=int, default=None, help="Qwen 專用，覆寫預設值")
    parser.add_argument("--max-tiles", type=int, default=None, help="InternVL 專用，覆寫預設值")
    parser.add_argument("--dry-run", action="store_true",
                        help="只讀檔、印出每頁 token 數與影像前處理結果，不載入模型")
    parser.add_argument("--out-root", default=OUT_ROOT)
    parser.add_argument("--dataset-root", default=DATASET_ROOT,
                        help="root holding {task}/ page images (and {task}_ocr_filtered/ for step 6)")
    parser.add_argument("--oom-retries", type=int, default=6,
                        help="單頁遇到 GPU OOM 時的重試次數。預設 6（耐心等別人放開 GPU）；"
                             "單卡模式建議設 1，讓塞不下的頁快速失敗、留給雙卡那趟收尾")
    parser.add_argument("--max-consecutive-oom", type=int, default=0,
                        help="連續幾頁都是 OOM 就判定「這個 GPU 配置根本塞不下」並以 exit code 75 "
                             "結束（0 = 關閉）。單卡並行階段設 3：與其花好幾小時把每一頁都試一遍，"
                             "不如早點認賠、把卡讓給下一個 stage，"
                             "塞不下的頁自然會在雙卡收尾階段補齊")
    parser.add_argument("--load-oom-retries", type=int, default=20,
                        help="模型【載入】階段的 OOM 重試次數。預設 20（最久約 4 小時，"
                             "耐心等別人放開 GPU）；單卡模式建議設 1，塞不下就快速放棄")
    parser.add_argument("--oom-wait", type=int, default=60,
                        help="OOM 重試的起始等待秒數（之後指數加倍）")
    parser.add_argument("--force-overwrite", action="store_true",
                        help="忽略 run_manifest.json 的模型/解析度不符檢查（會覆蓋既有結果，慎用）")
    args = parser.parse_args()

    DATASET_ROOT = args.dataset_root

    model_id = resolve_model_id(args.model, args.model_id)
    model_tag = backend_tag(args.model, args.model_id)

    backend = None
    if not args.dry_run:
        print(f"[load] {model_id} -> out/{model_tag}/{args.res}/ fp16, res={args.res} ...",
              flush=True)
        t0 = time.time()
        # 載入權重要一次吃下十幾 GB，最容易撞到別人的 GPU 尖峰 → 耐心等待重試，
        # 而不是讓整個 stage 直接失敗。
        backend = load_with_oom_retry(
            lambda: build_backend(args.model, args.res, max_pixels=args.max_pixels,
                                  max_tiles=args.max_tiles, model_id=model_id).load(),
            max_retries=args.load_oom_retries,
            label=f"{model_tag}/{args.res}")
        print(f"[load] done in {time.time() - t0:.1f}s", flush=True)

    for task in args.task:
        ocr_dir = os.path.join(DATASET_ROOT, f"{task}_ocr_filtered")
        img_dir = os.path.join(DATASET_ROOT, task)
        out_dir = os.path.join(args.out_root, model_tag, args.res,
                                f"{task}_ocr_filtered_classified_normalized")
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "run_log.jsonl")

        if not args.dry_run:
            assert_manifest(out_dir, model_id, args.res, force=args.force_overwrite)

        if not os.path.isdir(ocr_dir):
            print(f"[warn] not a directory, skip: {ocr_dir}", flush=True)
            continue

        json_files = sorted(
            f for f in os.listdir(ocr_dir) if f.endswith("_ocr_filtered.json"))
        if not json_files:
            print(f"[warn] no *_ocr_filtered.json in {ocr_dir}", flush=True)
            continue

        print(f"\n=== {task} | {model_tag} | res={args.res} | {len(json_files)} files ===",
              flush=True)

        consecutive_oom = 0

        for filename in json_files:
            base = filename[:-len(".json")]  # "..._ocr_filtered"
            page_stem = base[:-len("_ocr_filtered")]  # "..._NNN"
            ocr_json_path = os.path.join(ocr_dir, filename)
            image_path = os.path.join(img_dir, f"{page_stem}.png")
            out_path = os.path.join(out_dir, f"{base}_classified_normalized.json")

            if args.resume and os.path.exists(out_path):
                print(f"[skip] {base} (exists)", flush=True)
                continue

            if not os.path.exists(image_path):
                print(f"[warn] image not found, skip: {image_path}", flush=True)
                continue

            if args.dry_run:
                with open(ocr_json_path, "r", encoding="utf-8") as f:
                    n = len(json.load(f)["filtered_text"])
                _, img_meta = prepare_image(image_path, args.res)
                print(f"[dry] {base}: {n} tokens, {img_meta}", flush=True)
                continue

            t0 = time.time()
            try:
                result, raw_text, record = process_file(
                    backend, ocr_json_path, image_path, args.res, args.max_new_tokens,
                    label=base, oom_retries=args.oom_retries, oom_wait=args.oom_wait)
            except Exception as e:
                # 單頁救不回來不該讓已經跑了很多頁的整批任務中斷 → 記錄後跳過。
                #
                # 關鍵：這裡【不寫出結果檔】。以前會寫一個空結果，結果 --resume
                # 下次看到「檔案已存在」就跳過，把一次暫時性失敗（例如 OOM、
                # 檔案還沒同步好）變成永久遺失，而且毫無徵兆。不寫檔的話，
                # 下一輪 --resume 會自動重跑這頁。
                print(f"[error] {base}: {type(e).__name__}: {e} (不寫檔，下次 resume 會重試)",
                      flush=True)
                err = {"file": base, "parse_status": "error", "error": f"{type(e).__name__}: {e}",
                       "total_seconds": round(time.time() - t0, 2)}
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(err, ensure_ascii=False) + "\n")
                gc.collect()
                torch.cuda.empty_cache()

                if is_oom(e):
                    consecutive_oom += 1
                    if 0 < args.max_consecutive_oom <= consecutive_oom:
                        print(f"[bail] 連續 {consecutive_oom} 頁 OOM → 這個 GPU 配置塞不下 "
                              f"{model_tag}/{args.res}，提早結束（exit 75），"
                              f"未完成的頁留給雙卡階段", flush=True)
                        raise SystemExit(75)
                else:
                    consecutive_oom = 0
                continue

            consecutive_oom = 0
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            record["file"] = base
            record["total_seconds"] = round(time.time() - t0, 2)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"[ok] {base}: tokens {record['n_tokens_in']}->{record['n_tokens_out']} "
                  f"parse={record['parse_status']} retried={record.get('retried', False)} "
                  f"{record['total_seconds']}s", flush=True)

            gc.collect()
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
