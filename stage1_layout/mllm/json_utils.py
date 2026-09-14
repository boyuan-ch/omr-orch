"""從模型的自由文字輸出抽出 JSON，並正規化成與 GPT 版 (8_gpt_staffinfo.py) 完全一致的 schema。

沒有 vLLM / outlines / xgrammar 可用，無法做 constrained decoding，
所以改成「prompt 要求純 JSON → 抽取 → 修復 → retry → 正規化」。
"""
import json
import re

ROMAN = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6,
    "vii": 7, "viii": 8, "ix": 9, "x": 10, "xi": 11, "xii": 12,
}

STAFF_KEYS = ["staff_id", "staff_group_id", "system_id", "ocr", "instrument", "tone", "part"]


def strip_fences(text):
    """去掉 ```json ... ``` 之類的 markdown 圍籬與常見前綴。"""
    text = text.strip()
    text = re.sub(r"^<\|.*?\|>", "", text).strip()
    fence = re.match(r"^```(?:json|JSON)?\s*(.*?)\s*```$", text, flags=re.S)
    if fence:
        return fence.group(1).strip()
    # 只有開頭圍籬、結尾被截斷的情況
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_block(text):
    """用括號配對抽出最外層的 JSON 物件或陣列（會跳過字串內的括號）。"""
    candidates = [p for p in (text.find("{"), text.find("[")) if p != -1]
    if not candidates:
        return None
    start = min(candidates)
    opener = text[start]
    closer = "}" if opener == "{" else "]"

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    # 沒有閉合 → 截斷的輸出，整段交給 repair
    return text[start:]


def repair_json(block):
    """修復被 max_new_tokens 截斷或有尾逗號的 JSON。"""
    candidate = block.rstrip()

    # 字串在中途被截斷 → 砍掉最後一個未閉合的字串
    if candidate.count('"') % 2 == 1:
        candidate = candidate[:candidate.rfind('"')].rstrip()

    # 砍掉尾端不完整的片段（回到最後一個結構完整的位置）
    candidate = re.sub(r",\s*$", "", candidate)
    candidate = re.sub(r":\s*$", ": null", candidate)

    # 依配對狀況補上閉合括號
    depth_curly = 0
    depth_square = 0
    in_str = False
    escape = False
    for ch in candidate:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth_curly += 1
        elif ch == "}":
            depth_curly -= 1
        elif ch == "[":
            depth_square += 1
        elif ch == "]":
            depth_square -= 1

    if in_str:
        candidate += '"'

    # 由內而外閉合：先看最後未閉合的是哪一種
    tail = []
    stack = []
    in_str = False
    escape = False
    for ch in candidate:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    for opener in reversed(stack):
        tail.append("}" if opener == "{" else "]")

    candidate = re.sub(r",\s*$", "", candidate) + "".join(tail)
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    return candidate


def parse_model_json(text):
    """回傳 (data, status)。status ∈ {'ok', 'repaired', 'failed'}。"""
    cleaned = strip_fences(text)
    block = extract_json_block(cleaned)
    if block is None:
        return None, "failed"

    try:
        return json.loads(block), "ok"
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(repair_json(block)), "repaired"
    except json.JSONDecodeError:
        return None, "failed"


# --------------------------------------------------------------------------
# 正規化成 GPT schema
# --------------------------------------------------------------------------

def _to_int(value, default=0):
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if s.lower() in ROMAN:
            return ROMAN[s.lower()]
        m = re.search(r"-?\d+", s)
        if m:
            return int(m.group())
    return default


def _to_str_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [] if s == "" else [s]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_to_str_list(item))
        return out
    return [str(value)]


def _to_part_list(value):
    """part 一律轉成 list[int]；支援 '1-2'、'I'、'1,2'、[1,2]、'1 & 2'。"""
    if value is None:
        return []
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_to_part_list(item))
        return out

    s = str(value).strip()
    if s == "" or s.lower() in {"none", "null", "n/a"}:
        return []

    rng = re.fullmatch(r"\s*(\d+)\s*[-–~]\s*(\d+)\s*", s)
    if rng:
        lo, hi = int(rng.group(1)), int(rng.group(2))
        if lo <= hi and hi - lo < 32:
            return list(range(lo, hi + 1))
        return [lo, hi]

    parts = [p for p in re.split(r"[,\s&/]+", s) if p]
    out = []
    for p in parts:
        if p.lower() in ROMAN:
            out.append(ROMAN[p.lower()])
            continue
        m = re.search(r"\d+", p)
        if m:
            out.append(int(m.group()))
    return out


def _normalize_stave(raw, fallback_id):
    if not isinstance(raw, dict):
        return None
    return {
        "staff_id": _to_int(raw.get("staff_id"), fallback_id),
        "staff_group_id": _to_int(raw.get("staff_group_id"), 1),
        "system_id": _to_int(raw.get("system_id"), 1),
        "ocr": "" if raw.get("ocr") is None else str(raw.get("ocr")),
        "instrument": _to_str_list(raw.get("instrument")),
        "tone": _to_str_list(raw.get("tone")),
        "part": _to_part_list(raw.get("part")),
    }


def _find_staves(data):
    """模型可能回 {"pages":[...]}、單一 page 物件、或直接一個 staves list。"""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []

    pages = data.get("pages")
    if isinstance(pages, list) and pages:
        staves = []
        for page in pages:
            if isinstance(page, dict) and isinstance(page.get("staves"), list):
                staves.extend(page["staves"])
        return staves
    if isinstance(data.get("staves"), list):
        return data["staves"]
    if "staff_id" in data or "ocr" in data:  # 只回了單一 staff
        return [data]
    return []


def normalize_to_gpt_schema(data, filename, image_index, renumber=True):
    """輸出與 GPT 版逐鍵一致的 {"pages":[{"image_index","filename","staves":[...]}]}。"""
    staves = []
    for i, raw in enumerate(_find_staves(data), start=1):
        stave = _normalize_stave(raw, i)
        if stave is not None:
            staves.append(stave)

    if renumber:
        for i, stave in enumerate(staves, start=1):
            stave["staff_id"] = i

    return {
        "pages": [{
            "image_index": image_index,
            "filename": filename,
            "staves": staves,
        }]
    }


def empty_result(filename, image_index):
    return {"pages": [{"image_index": image_index, "filename": filename, "staves": []}]}
