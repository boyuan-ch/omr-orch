"""把本專案的 classify 輸出轉成「GPT-5.1 版逐鍵等價」的檔案，好餵給既有 pipeline。

為什麼需要這一層（兩個實測到的落差）：

1. **空值表示法不同**：GPT 版把沒有的欄位寫成 `""` / `[]`，本專案寫成 `null`。
   下游 `7_matching_reformat_standard_group.py:47` 是
   `cls_item.get("Instrument", "").strip()` —— 拿到 `None` 會直接
   `AttributeError: 'NoneType' object has no attribute 'strip'`。

2. **token 數可能對不上**：下游用
   `zip(filtered_text, filtered_bbox, classified_tokens)` 把分類結果和 OCR 的
   bbox 逐一配對，**長度不符時 zip 會靜默截斷**，導致 token 與座標錯位而且
   毫無錯誤訊息。GPT 版 1063 頁全部對得上，本專案的模型有 12~19 頁（約 1.2~1.8%）
   對不上（模型漏掉或多吐了 token）。

   這裡不是粗暴地補在尾端 —— 若模型是在中間漏掉 token，尾端補齊仍然全錯位。
   改用 difflib 依 token 文字做序列對齊，把模型的分類結果放回它對應的原始位置，
   對不到的位置補一個中性項目（class_id=4「以上皆非」、樂器留空），
   確保輸出長度與 OCR token 數完全一致、且位置正確。

原始輸出【不會被修改】，轉換後的副本另外寫到 out_gptcompat/，保留 provenance。

用法：
  python gpt_compat.py                    # 轉換全部 model_tag/res
  python gpt_compat.py --tag qwen3vl_8b --res api
"""
import argparse
import difflib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT_COMPAT = HERE / "out_gptcompat"
DATASET = Path(os.environ.get("OMR_DATASET_ROOT", "data"))

# GPT 版對「沒有值」的寫法：字串欄位是 ""、Parts 是 []
NEUTRAL = {"class_id": 4, "Instrument": "", "Tone": "", "Instruction": "", "Parts": []}


def _clean(tok):
    """單筆 token：null -> GPT 的空值寫法，並確保欄位齊全。"""
    out = {"token": str(tok.get("token") or "")}
    cid = tok.get("class_id")
    out["class_id"] = cid if isinstance(cid, int) and 0 <= cid <= 5 else 4
    for k in ("Instrument", "Tone", "Instruction"):
        v = tok.get(k)
        out[k] = "" if v is None else str(v)
    p = tok.get("Parts")
    out["Parts"] = [] if p is None else (list(p) if isinstance(p, list) else [p])
    return out


def align_to_source(model_tokens, source_texts):
    """把模型的分類結果對齊回原始 OCR token 序列，回傳長度 == len(source_texts) 的 list。"""
    cleaned = [_clean(t) for t in model_tokens]
    if len(cleaned) == len(source_texts):
        # 長度相同就直接照位置對應（絕大多數頁面走這條）
        return cleaned

    result = [dict(NEUTRAL, token=str(t)) for t in source_texts]
    sm = difflib.SequenceMatcher(
        a=[str(t) for t in source_texts],
        b=[t["token"] for t in cleaned],
        autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for off in range(i2 - i1):
            item = dict(cleaned[j1 + off])
            item["token"] = str(source_texts[i1 + off])   # 一律以原始 token 為準
            result[i1 + off] = item
    return result


def convert_dir(tag, res, verbose=True):
    src_root = OUT / tag / res
    if not src_root.is_dir():
        return 0, 0
    n_files = n_realigned = 0
    for piece_dir in sorted(src_root.glob("*_ocr_filtered_classified_normalized")):
        piece = piece_dir.name.replace("_ocr_filtered_classified_normalized", "")
        dst_dir = OUT_COMPAT / tag / res / piece_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(piece_dir.glob("*_ocr_filtered_classified_normalized.json")):
            base = f.name.replace("_ocr_filtered_classified_normalized.json", "")
            ocr_f = DATASET / f"{piece}_ocr_filtered" / f"{base}_ocr_filtered.json"
            if not ocr_f.exists():
                continue
            try:
                model_tokens = json.loads(f.read_text(encoding="utf-8")).get("tokens", [])
                source_texts = json.loads(ocr_f.read_text(encoding="utf-8"))["filtered_text"]
            except (json.JSONDecodeError, KeyError, OSError):
                continue
            if len(model_tokens) != len(source_texts):
                n_realigned += 1
            aligned = align_to_source(model_tokens, source_texts)
            (dst_dir / f.name).write_text(
                json.dumps({"tokens": aligned}, indent=2, ensure_ascii=False), encoding="utf-8")
            n_files += 1
    if verbose:
        print(f"{tag}/{res}: 轉換 {n_files} 頁（其中 {n_realigned} 頁需要重新對齊）")
    return n_files, n_realigned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--res", default=None)
    args = ap.parse_args()

    tags = [args.tag] if args.tag else [d.name for d in sorted(OUT.glob("*")) if d.is_dir()]
    total = 0
    for tag in tags:
        for res in ([args.res] if args.res else ["api"]):
            n, _ = convert_dir(tag, res)
            total += n
    print(f"\n完成，共 {total} 個檔案 -> {OUT_COMPAT}")


if __name__ == "__main__":
    main()
