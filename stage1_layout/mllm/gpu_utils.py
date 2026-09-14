"""GPU 共用機的韌性工具：遇到 CUDA OOM 時等待重試，而不是直接放棄那一頁。

為什麼需要這個：這台機器的 GPU 是跟別人共用的（實測過 3090 會突然被別的
container 佔滿到只剩 2GB）。長跑數天的批次任務如果一撞到別人的記憶體尖峰
就把整頁記成 error，會白白丟掉大量本來跑得動的頁面 —— 那頁通常只是剛好
撞上尖峰，等幾分鐘就好了。

但也不能無限重試（會卡死整批任務），所以用「有上限的指數退避」：
清快取 → 等待 → 重試，等待時間逐次加倍，超過上限才真的放棄那一頁。
"""
import gc
import time

import torch


def _free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom(exc):
    """torch 的 OOM 有時候是 torch.cuda.OutOfMemoryError，有時候是包在
    RuntimeError 裡的 'CUDA out of memory' 字串（不同版本／不同 kernel 路徑），
    兩種都要認得。"""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda error" in msg
    return False


def retry_on_oom(fn, *, max_retries=6, base_wait=60, max_wait=900, label="", log=print):
    """呼叫 fn()；遇到 OOM 就清快取、等待後重試，回傳 fn() 的結果。

    max_retries=6 / base_wait=60 → 最久會等 60+120+240+480+900+900 ≈ 45 分鐘
    才真的放棄一頁。對「別人跑一個中等長度的 job」這種情境夠用，也不會讓
    整批任務卡死太久。

    非 OOM 的例外一律直接往上拋（那是真的程式錯誤，重試也沒用），
    交給呼叫端的 per-page try/except 記錄成 error 並繼續下一頁。
    """
    wait = base_wait
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - 需要辨識 OOM 後再決定要不要重試
            if not is_oom(exc):
                raise
            _free_cuda()
            if attempt >= max_retries:
                log(f"[oom] {label}: 重試 {max_retries} 次仍 OOM，放棄這一頁", flush=True)
                raise
            log(f"[oom] {label}: 第 {attempt + 1}/{max_retries} 次 OOM，"
                f"等 {wait}s 後重試（可能是別人在用 GPU）", flush=True)
            time.sleep(wait)
            wait = min(wait * 2, max_wait)


def load_with_oom_retry(build_fn, *, max_retries=20, base_wait=120, max_wait=900,
                        label="model load", log=print):
    """載入模型專用的重試：權重載入要一次吃下十幾 GB，最容易撞到別人的尖峰。

    這裡給比較多次、比較長的等待（最久約 4 小時），因為載不進模型的話整個
    stage 都不用跑了 —— 與其讓 stage 直接失敗，不如耐心等 GPU 空出來。
    """
    return retry_on_oom(build_fn, max_retries=max_retries, base_wait=base_wait,
                        max_wait=max_wait, label=label, log=log)
