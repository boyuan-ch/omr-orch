"""影像前處理：把樂譜頁面縮到與 OpenAI Responses API 等效的解析度。

OpenAI 視覺輸入（detail: auto / high）的規則：
  1. 先等比縮放，使影像放得進 2048x2048 的框（原本就比較小則不放大）
  2. 再等比縮放，使「短邊」= 768
  3. 切成 512x512 的 tiles，token 數 = 85 + 170 * tiles

本資料集（2415x3300 ~ 2550x4200）縮完是 768x1041 ~ 768x1265，6 tiles，約 1105 影像 tokens。
"""
import math

from PIL import Image

MAX_BOX = 2048
SHORT_SIDE = 768
TILE = 512


def openai_equivalent_size(width, height):
    """回傳 (fit2048 後尺寸, 最終尺寸, tile 數, 估計影像 tokens)。"""
    s = min(MAX_BOX / width, MAX_BOX / height, 1.0)
    w1, h1 = round(width * s), round(height * s)

    s2 = SHORT_SIDE / min(w1, h1)
    w2, h2 = round(w1 * s2), round(h1 * s2)

    tiles = math.ceil(w2 / TILE) * math.ceil(h2 / TILE)
    return (w1, h1), (w2, h2), tiles, 85 + 170 * tiles


def openai_equivalent_resize(image):
    """把 PIL 影像縮成 API 實際看到的那一張，回傳 (image, meta)。"""
    w, h = image.size
    fit_box, final, tiles, tokens = openai_equivalent_size(w, h)

    resized = image.resize(final, Image.LANCZOS) if final != (w, h) else image
    meta = {
        "orig_size": [w, h],
        "fit2048_size": list(fit_box),
        "final_size": list(final),
        "megapixels": round(final[0] * final[1] / 1e6, 3),
        "openai_tiles": tiles,
        "openai_image_tokens": tokens,
    }
    return resized, meta


def prepare_image(path, res_mode="api"):
    """Downscale a page to the resolution the OpenAI vision API would actually see.

    This is the only resolution mode in this release, so every backend (GPT, Qwen, InternVL) is
    shown the same pixels and the Table 1 comparison is apples to apples.
    """
    assert res_mode == "api", f"unsupported res_mode: {res_mode}"
    return openai_equivalent_resize(Image.open(path).convert("RGB"))


if __name__ == "__main__":
    for size in [(2550, 4200), (2438, 3305), (2480, 3508), (2524, 3511)]:
        fit_box, final, tiles, tokens = openai_equivalent_size(*size)
        print(f"{size[0]}x{size[1]} -> {fit_box} -> {final} "
              f"({final[0] * final[1] / 1e6:.2f} MP), tiles={tiles}, tokens≈{tokens}")
