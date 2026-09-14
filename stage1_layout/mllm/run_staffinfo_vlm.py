"""用開源 VLM（Qwen3-VL-8B / InternVL3.5-8B, FP16）跑 8_gpt_staffinfo.py 的任務。

輸入：{dataset_root}/{task}/{page}.png
輸出：out/{model_tag}/{res}/{task}/{page}_staffgroup.json（結構與 GPT 版完全一致）

範例：
  python run_staffinfo_vlm.py --task Bee_2_challenge --model qwen --res api --start 42 --num 5
  python run_staffinfo_vlm.py --task Bee_2_challenge --model internvl --res api --resume
  python run_staffinfo_vlm.py --task Bee_2_challenge --model qwen --res api --dry-run
"""
import argparse
import gc
import json
import os
import sys
import time

# 讓 CUDA caching allocator 用可擴展的 segment，緩解長跑（數百頁、tile 形狀不斷變動）
# 累積碎片化導致的 OOM；必須在 torch 被 import／初始化 CUDA context 之前設定。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")  # torch>=2.7 改用的新名稱

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_utils import is_oom, load_with_oom_retry, retry_on_oom  # noqa: E402
from image_prep import prepare_image  # noqa: E402
from json_utils import empty_result, normalize_to_gpt_schema, parse_model_json  # noqa: E402

DATASET_ROOT = os.environ.get("OMR_DATASET_ROOT", "data")  # override with --dataset-root
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# Prompt copied verbatim from the GPT version of this step (8_gpt_staffinfo.py).
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

# GPT 版是用 json_schema strict 強制格式，本地模型沒有 constrained decoding，
# 因此額外加這一句（這是與 GPT 版唯一的 prompt 差異）
JSON_ONLY_SUFFIX = "\n\n    Output ONLY the JSON object. No markdown fences, no explanation, no extra text."

RETRY_SUFFIX = ("\n\n    IMPORTANT: your previous answer was not valid JSON. "
                "Respond with ONLY the JSON object, starting with { and ending with }.")


def build_prompt(filename, retry=False):
    prompt = TEXT_PROMPT + f"\nProcess the image: {filename}" + JSON_ONLY_SUFFIX
    if retry:
        prompt += RETRY_SUFFIX
    return prompt


def process_page(backend, image_path, filename, image_index, args):
    image, img_meta = prepare_image(image_path, args.res)

    record = {"filename": filename, "image": img_meta, "retries": 0}

    raw_text = ""
    data = None
    status = "failed"
    gen_meta = {}
    for attempt in range(2):
        # 包一層 OOM 重試：GPU 跟別人共用，撞到別人的記憶體尖峰時等一下再試，
        # 不要把本來跑得動的頁面直接判死（見 gpu_utils.retry_on_oom）。
        raw_text, gen_meta = retry_on_oom(
            lambda a=attempt: backend.generate(
                image, build_prompt(filename, retry=a > 0), args.max_new_tokens),
            max_retries=args.oom_retries, base_wait=args.oom_wait, label=filename)
        data, status = parse_model_json(raw_text)
        if status != "failed":
            break
        record["retries"] = attempt + 1

    if data is None:
        result = empty_result(filename, image_index)
    else:
        result = normalize_to_gpt_schema(
            data, filename, image_index, renumber=not args.no_renumber)

    record.update(gen_meta)
    record["parse_status"] = status
    record["n_staves"] = len(result["pages"][0]["staves"])
    return result, raw_text, record


MANIFEST_NAME = "run_manifest.json"


def assert_manifest(out_dir, model_id, res, task, force=False):
    """在輸出資料夾放一張「身分證」，並確認沒有跟別的模型混用同一個資料夾。

    這是防止「換了 model_id 卻寫進舊資料夾」的最後一道防線：
    tag 已經跟著 model_id 走，但萬一兩個 model_id 被 slug 成同名，這裡會擋下來。
    """
    path = os.path.join(out_dir, MANIFEST_NAME)
    current = {"model_id": model_id, "res": res, "task": task}

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        mismatch = {k: (prev.get(k), v) for k, v in current.items() if prev.get(k) != v}
        if mismatch and not force:
            lines = "\n".join(f"    {k}: 既有={old!r}  這次={new!r}"
                              for k, (old, new) in mismatch.items())
            raise SystemExit(
                f"\n[ABORT] 輸出資料夾已被別的設定使用過，停止以免混入/覆蓋既有結果：\n"
                f"  {out_dir}\n{lines}\n"
                f"  既有 manifest：{path}\n\n"
                f"  處理方式（擇一）：\n"
                f"    1. 去 backends.py 的 MODEL_TAGS_BY_ID 幫新 model_id 登記一個不同的資料夾名稱（建議）\n"
                f"    2. 改用 --out-root 指向另一個目錄\n"
                f"    3. 真的要覆蓋就加 --force-overwrite（會毀掉既有結果，請先備份）\n")
        if mismatch and force:
            print(f"[warn] --force-overwrite：忽略 manifest 不符 {mismatch}", flush=True)

    current["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2, ensure_ascii=False)


def main():
    global DATASET_ROOT
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", nargs="+", required=True,
                        help="例如 Bee_2_challenge（可給多個）")
    parser.add_argument("--model",
                        choices=["qwen3", "internvl3_5"],
                        required=True,
                        help="模型簡稱（用 --model-id 可覆寫成任何 HF repo）")
    parser.add_argument("--model-id", default=None,
                        help="覆寫 HF repo id（例如 Qwen/Qwen3-VL-8B-Instruct）。"
                             "輸出資料夾名稱一律跟著 model_id 走，不會蓋到別的模型")
    parser.add_argument("--res", choices=["api"], default="api",
                        help="api = 縮成 OpenAI 等效解析度（短邊 768）；high = 原圖高解析度")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num", type=int, default=None, help="處理幾頁（預設全部）")
    parser.add_argument("--resume", action="store_true", help="跳過已存在的輸出檔")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-pixels", type=int, default=None, help="Qwen 專用")
    parser.add_argument("--max-tiles", type=int, default=None, help="InternVL 專用")
    parser.add_argument("--no-renumber", action="store_true",
                        help="不要重編 staff_id（預設會依序重編）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只做影像前處理與 prompt 組裝，不載入模型")
    parser.add_argument("--out-root", default=OUT_ROOT)
    parser.add_argument("--dataset-root", default=DATASET_ROOT,
                        help="root holding {task}/ page images (and {task}_ocr_filtered/ for step 6)")
    parser.add_argument("--oom-retries", type=int, default=6,
                        help="單頁遇到 GPU OOM 時的重試次數。預設 6（耐心等別人放開 GPU）；"
                             "單卡模式建議設 1，讓塞不下的頁快速失敗、留給雙卡那趟收尾")
    parser.add_argument("--max-consecutive-oom", type=int, default=0,
                        help="連續幾頁都是 OOM 就判定「這個 GPU 配置根本塞不下」並以 exit code 75 "
                             "結束（0 = 關閉）。單卡並行階段設 3：與其花好幾小時把每一頁都試一遍"
                             "（每頁還要等重試），不如早點認賠、把卡讓給下一個 stage，"
                             "塞不下的頁自然會在雙卡收尾階段補齊")
    parser.add_argument("--load-oom-retries", type=int, default=20,
                        help="模型【載入】階段的 OOM 重試次數。預設 20（最久約 4 小時，"
                             "耐心等別人放開 GPU）；單卡模式建議設 1，塞不下就快速放棄")
    parser.add_argument("--oom-wait", type=int, default=60,
                        help="OOM 重試的起始等待秒數（之後指數加倍）")
    parser.add_argument("--force-overwrite", action="store_true",
                        help="忽略 run_manifest.json 的模型不符檢查（會覆蓋既有結果，慎用）")
    args = parser.parse_args()

    DATASET_ROOT = args.dataset_root

    from backends import backend_tag, build_backend, resolve_model_id

    model_id = resolve_model_id(args.model, args.model_id)
    model_tag = backend_tag(args.model, args.model_id)

    backend = None
    if not args.dry_run:
        print(f"[load] {model_id} -> out/{model_tag}/ fp16, res={args.res} ...", flush=True)
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
        image_folder = os.path.join(DATASET_ROOT, task)
        out_dir = os.path.join(args.out_root, model_tag, args.res, task)
        raw_dir = os.path.join(out_dir, "_raw")
        os.makedirs(raw_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "run_log.jsonl")

        # 硬性防呆：若這個資料夾先前是別的 model_id / res 產生的，直接中止。
        # 否則 --resume 會把兩個模型的結果混在同一個資料夾裡，而且悄無聲息。
        assert_manifest(out_dir, model_id, args.res, task, force=args.force_overwrite)

        image_files = sorted(f for f in os.listdir(image_folder) if f.endswith(".png"))
        end = len(image_files) if args.num is None else min(args.start + args.num, len(image_files))

        print(f"\n=== {task} | {model_tag} | res={args.res} | pages {args.start}..{end - 1} "
              f"(total {len(image_files)}) ===", flush=True)

        consecutive_oom = 0

        for i in range(args.start, end):
            filename = image_files[i]
            base = filename[:-4]
            out_path = os.path.join(out_dir, f"{base}_staffgroup.json")

            if args.resume and os.path.exists(out_path):
                print(f"[skip] {base} (exists)", flush=True)
                continue

            image_path = os.path.join(image_folder, filename)

            if args.dry_run:
                _, img_meta = prepare_image(image_path, args.res)
                print(f"[dry] {base}: {img_meta}", flush=True)
                continue

            t0 = time.time()
            try:
                result, raw_text, record = process_page(backend, image_path, filename, i + 1, args)
            except Exception as e:
                # 單頁救不回來（例如 OOM 重試全部失敗）不該讓已經跑了幾百頁的整批任務
                # 中斷。記下這頁失敗、清乾淨 GPU 狀態，繼續跑下一頁。
                #
                # 關鍵：這裡【不寫出結果檔】。以前會寫一個空結果，結果 --resume
                # 下次看到「檔案已存在」就跳過，把一次暫時性失敗變成永久遺失，
                # 而且毫無徵兆。不寫檔的話，下一輪 --resume 會自動重跑這頁。
                print(f"[error] {base}: {type(e).__name__}: {e} (不寫檔，下次 resume 會重試)",
                      flush=True)
                err = {"filename": filename, "task": task, "res": args.res,
                       "page_index": i, "parse_status": "error",
                       "error": f"{type(e).__name__}: {e}",
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
                f.write(json.dumps(result, indent=2, ensure_ascii=False))
            with open(os.path.join(raw_dir, f"{base}.txt"), "w", encoding="utf-8") as f:
                f.write(raw_text)

            record["task"] = task
            record["res"] = args.res
            record["page_index"] = i
            record["total_seconds"] = round(time.time() - t0, 2)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"[ok] {base}: staves={record['n_staves']} parse={record['parse_status']} "
                  f"retries={record['retries']} {record['total_seconds']}s", flush=True)

            # high 模式下每頁 tile 數隨長寬比變動，張量形狀反覆改變會讓 CUDA
            # caching allocator 碎片化、用量緩慢爬升，長跑數百頁後會 OOM
            # （即使瞬時用量遠低於顯卡容量）。每頁跑完主動釋放快取。
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
