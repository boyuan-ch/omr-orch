"""從模型的自由文字輸出抽出 JSON，並正規化成與 GPT 版
Token classification schema, key-for-key identical to the GPT step (6_1_gpt_classify_reformat.py).

沒有 vLLM / outlines / xgrammar 可用，無法做 constrained decoding，
所以改成「prompt 要求純 JSON → 抽取 → 修復 → retry → 正規化」
（跟 vlm_staffinfo/json_utils.py 同一套策略，這裡是獨立複製一份，
schema 不同：token 分類是扁平的 {"tokens":[...]}，不是 staff 的 {"pages":[...]}）。
"""
import json
import re

ROMAN = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6,
    "vii": 7, "viii": 8, "ix": 9, "x": 10, "xi": 11, "xii": 12,
}


def strip_fences(text):
    """去掉 ```json ... ``` 之類的 markdown 圍籬與常見前綴。"""
    text = text.strip()
    text = re.sub(r"^<\|.*?\|>", "", text).strip()
    fence = re.match(r"^```(?:json|JSON)?\s*(.*?)\s*```$", text, flags=re.S)
    if fence:
        return fence.group(1).strip()
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
    return text[start:]


def repair_json(block):
    """修復被 max_new_tokens 截斷或有尾逗號的 JSON。"""
    candidate = block.rstrip()

    if candidate.count('"') % 2 == 1:
        candidate = candidate[:candidate.rfind('"')].rstrip()

    candidate = re.sub(r",\s*$", "", candidate)
    candidate = re.sub(r":\s*$", ": null", candidate)

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
# 正規化成 GPT 版 CLASSIFICATION_SCHEMA：
#   {"tokens": [{"token","class_id","Instrument","Tone","Instruction","Parts"}, ...]}
# --------------------------------------------------------------------------

def _to_part_list(value):
    """Parts 一律轉成 list[int] 或 None；支援 '1-2'、'I'、'1,2'、[1,2]、'1 & 2'。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, list):
        out = []
        for item in value:
            sub = _to_part_list(item)
            if sub:
                out.extend(sub)
        return out or None

    s = str(value).strip()
    if s == "" or s.lower() in {"none", "null", "n/a"}:
        return None

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
    return out or None


def _norm_class_id(value):
    """class_id 必須是 0-5 的整數；解析不出來或超出範圍一律歸類成 4（None of the above），
    這是本地模型沒有 constrained decoding 時最安全的失敗預設值。"""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return 4
    return v if 0 <= v <= 5 else 4


def _norm_nullable_str(value):
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _normalize_token(raw):
    if not isinstance(raw, dict):
        return None
    return {
        "token": "" if raw.get("token") is None else str(raw.get("token")),
        "class_id": _norm_class_id(raw.get("class_id")),
        "Instrument": _norm_nullable_str(raw.get("Instrument")),
        "Tone": _norm_nullable_str(raw.get("Tone")),
        "Instruction": _norm_nullable_str(raw.get("Instruction")),
        "Parts": _to_part_list(raw.get("Parts")),
    }


def _find_tokens(data):
    """模型可能回 {"tokens":[...]}、直接一個 list、或單一 token 物件。"""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    toks = data.get("tokens")
    if isinstance(toks, list):
        return toks
    if "token" in data:
        return [data]
    return []


def normalize_tokens(data):
    tokens = []
    for raw in _find_tokens(data):
        t = _normalize_token(raw)
        if t is not None:
            tokens.append(t)
    return {"tokens": tokens}


def empty_result():
    return {"tokens": []}
