"""開源 VLM backend：Qwen3-VL-8B-Instruct 與 InternVL3.5-8B（皆 FP16）。

硬體限制（TITAN RTX / Turing sm_75）：
  * 不支援 bf16 → 一律 torch.float16
  * 沒有 flash-attn 2 → Qwen 用 sdpa、InternVL 用 use_flash_attn=False
"""
import gc
import math
import time

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
INTERNVL_MODEL_ID = "OpenGVLab/InternVL3_5-8B"

# ---------------------------------------------------------------------------
# 輸出資料夾名稱（tag）一律由 model_id 決定，不要寫死在 class 上。
#
# 為什麼：out/{tag}/{res}/{task}/ 是輸出路徑。如果換了 model_id 卻沿用舊 tag，
# 新結果會【蓋掉】舊模型的結果；更糟的是配上 --resume 會【跳過】已存在的頁，
# 產生一個一半舊模型、一半新模型的混合資料夾，而且不會有任何錯誤訊息。
#
# 下面這張表把 model_id -> 資料夾名稱釘死。既有的兩個 tag 不可更動
# （out/ 底下已經有結果，meta.ipynb 的 MODEL_TAGS 也指向它們）。
# 要加新模型就在這裡加一行。
# ---------------------------------------------------------------------------
MODEL_TAGS_BY_ID = {
    "Qwen/Qwen3-VL-8B-Instruct":   "qwen3vl_8b",
    "OpenGVLab/InternVL3_5-8B":    "internvl3_5_8b",
}


def tag_for_model_id(model_id):
    """model_id -> 輸出資料夾名稱。未登記的模型自動 slug 並警告。"""
    if model_id in MODEL_TAGS_BY_ID:
        return MODEL_TAGS_BY_ID[model_id]

    slug = model_id.split("/")[-1].lower()
    slug = slug.replace("-instruct", "").replace(".", "_").replace("-", "_")
    slug = "".join(ch for ch in slug if ch.isalnum() or ch == "_").strip("_")
    print(f"[warn] model_id 未登記在 MODEL_TAGS_BY_ID：{model_id}\n"
          f"       自動使用資料夾名稱 '{slug}'。建議去 backends.py 明確登記，"
          f"以免日後名稱對不上。", flush=True)
    return slug

# api 模式下影像已經是 API 等效尺寸（≤0.97MP），這個上限只是讓 processor 不要再縮
QWEN_MAX_PIXELS_API = 1_300_000
# high 模式：約 4096 個視覺 token（= 4096 * 28 * 28 像素）
# (a full-resolution mode existed during development; it is not part of this release)
# Qwen 官方預設下限，避免極小圖被縮得更小（本專案頁面都很大，這個下限實際上很少生效）
QWEN_MIN_PIXELS = 256 * 28 * 28

INTERNVL_MAX_TILES_API = 6
# (see the note above: no full-resolution mode in this release)


class Backend:
    model_id = None

    @property
    def tag(self):
        """輸出資料夾名稱，永遠跟著 self.model_id 走（不是寫死的 class 屬性）。"""
        return tag_for_model_id(self.model_id)

    def load(self):
        raise NotImplementedError

    def generate(self, image, prompt, max_new_tokens):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Qwen3-VL
# ---------------------------------------------------------------------------

class QwenVLBackend(Backend):
    def __init__(self, res_mode="api", max_pixels=None, min_pixels=None, model_id=QWEN_MODEL_ID):
        self.res_mode = res_mode
        self.model_id = model_id
        if max_pixels is None:
            max_pixels = QWEN_MAX_PIXELS_API
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels or QWEN_MIN_PIXELS
        self.model = None
        self.processor = None

    def load(self):
        # AutoModelForImageTextToText 依 config 的 architectures 自動分派到
        # Qwen2_5_VLForConditionalGeneration / Qwen3VLForConditionalGeneration，
        # 不必寫死 class（寫死會導致「用某一代的 class 硬載另一代 checkpoint」
        # 這種看似能跑、實際架構不符的錯誤）。
        from transformers import AutoModelForImageTextToText, AutoProcessor

        n_gpu = torch.cuda.device_count()
        kwargs = dict(
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory={i: "22GiB" for i in range(n_gpu)} if n_gpu > 1 else None,
        )
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, dtype=torch.float16, **kwargs)
        except TypeError:  # transformers < 4.56 用 torch_dtype
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, torch_dtype=torch.float16, **kwargs)
        self.model.eval()

        # 注意：AutoProcessor.from_pretrained(model_id, max_pixels=...) 這個舊式
        # kwarg 在這個 transformers 版本的 fast image processor 上【不會生效】
        # ——實測過 processor.image_processor.max_pixels 屬性確實被設成目標值，
        # 但實際 resize 用的是 size["longest_edge"]，兩者沒有同步，導致 high 模式
        # 下 10.7MP 原圖幾乎沒被縮小（10480 視覺 token，不是預期的 ~4096），
        # 大圖直接 OOM。必須明確傳 size={"longest_edge","shortest_edge"} 才會套用。
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            size={"longest_edge": self.max_pixels, "shortest_edge": self.min_pixels})
        return self

    def generate(self, image, prompt, max_new_tokens=4096):
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = inputs.to(self.model.device)

        grid = inputs["image_grid_thw"][0].tolist()
        merge = self.processor.image_processor.merge_size ** 2
        visual_tokens = int(grid[0] * grid[1] * grid[2] / merge)

        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = out[0][inputs["input_ids"].shape[1]:]
        answer = self.processor.tokenizer.decode(generated, skip_special_tokens=True)

        meta = {
            "backend": self.tag,
            "model_id": self.model_id,
            "image_grid_thw": grid,
            "processor_size": [grid[2] * 14, grid[1] * 14],
            "visual_tokens": visual_tokens,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "output_tokens": int(generated.shape[0]),
            "gen_seconds": round(time.time() - t0, 2),
        }
        return answer, meta


# ---------------------------------------------------------------------------
# InternVL3.5（remote code，需 model.chat()）
# Image preprocessing follows the official InternVL reference implementation.
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images, target_aspect_ratio


def load_image_pil(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images, grid = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values, grid


def split_model_device_map(model_id):
    """InternVL 官方的多卡切法：第 0 張卡要放 ViT，所以只算半張。"""
    from transformers import AutoConfig

    world_size = torch.cuda.device_count()
    if world_size <= 1:
        return {"": 0}

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    per_gpu = math.ceil(num_layers / (world_size - 0.5))
    layers_per_gpu = [per_gpu] * world_size
    layers_per_gpu[0] = math.ceil(per_gpu * 0.5)

    device_map = {}
    layer_cnt = 0
    for i, n_layer in enumerate(layers_per_gpu):
        for _ in range(n_layer):
            if layer_cnt >= num_layers:
                break
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1

    for key in [
        "vision_model", "mlp1",
        "language_model.model.tok_embeddings",
        "language_model.model.embed_tokens",
        "language_model.output",
        "language_model.model.norm",
        "language_model.model.rotary_emb",
        "language_model.lm_head",
        f"language_model.model.layers.{num_layers - 1}",
    ]:
        device_map[key] = 0
    return device_map


class InternVLBackend(Backend):
    def __init__(self, res_mode="api", max_tiles=None, model_id=INTERNVL_MODEL_ID):
        self.res_mode = res_mode
        self.model_id = model_id
        if max_tiles is None:
            max_tiles = INTERNVL_MAX_TILES_API
        self.max_tiles = max_tiles
        self.model = None
        self.tokenizer = None

    @staticmethod
    def _patch_generation_mixin(model):
        """transformers >= 4.50 起 PreTrainedModel 不再繼承 GenerationMixin，
        InternVL 的 remote code 沒跟上 transformers 時，language_model 會失去 .generate()。
        這裡動態換成同時繼承 GenerationMixin 的子類別。"""
        from transformers import GenerationConfig
        from transformers.generation import GenerationMixin

        lm = getattr(model, "language_model", None)
        if lm is None:
            return
        if not isinstance(lm, GenerationMixin):
            lm.__class__ = type(
                f"{lm.__class__.__name__}WithGenerationMixin",
                (lm.__class__, GenerationMixin),
                # remote code 只認得 legacy tuple 形式的 past_key_values，
                # 關掉 DynamicCache，讓 generate() 走舊的 tuple cache 路徑
                {"_supports_default_dynamic_cache": classmethod(lambda cls: False)},
            )
        # can_generate() 在 __init__ 當下是 False，所以 generation_config 沒被建立
        if getattr(lm, "generation_config", None) is None:
            lm.generation_config = GenerationConfig.from_model_config(lm.config)

    def load(self):
        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
            device_map=split_model_device_map(self.model_id),
        ).eval()
        self._patch_generation_mixin(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False)
        return self

    def generate(self, image, prompt, max_new_tokens=4096):
        # 同樣尺寸的頁面依長寬比四捨五入可能落在不同 tile grid（例如 3x4=13 vs
        # 4x6=25 tiles），tile 數翻倍時 eager attention 的 O(seq_len^2) fp32
        # 注意力矩陣也跟著翻倍，可能讓少數頁面 OOM。遇到 OOM 就清快取、降低
        # tile 上限重試，最多降兩次，讓極少數超大頁犧牲一點解析度也好過整批中斷。
        max_tiles = self.max_tiles
        attempt = 0
        while True:
            try:
                pixel_values, grid = load_image_pil(image, max_num=max_tiles)
                pixel_values = pixel_values.to(torch.float16).to(self.model.device)

                question = f"<image>\n{prompt}"
                generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)

                t0 = time.time()
                with torch.inference_mode():
                    answer = self.model.chat(
                        self.tokenizer, pixel_values, question, generation_config)
                break
            except torch.cuda.OutOfMemoryError:
                del pixel_values
                gc.collect()
                torch.cuda.empty_cache()
                attempt += 1
                if attempt > 2 or max_tiles <= 4:
                    raise
                max_tiles = max(4, max_tiles // 2)

        n_tiles = pixel_values.shape[0]
        meta = {
            "backend": self.tag,
            "model_id": self.model_id,
            "tile_grid": list(grid),
            "n_tiles_with_thumbnail": int(n_tiles),
            "processor_size": [grid[0] * 448, grid[1] * 448],
            "visual_tokens": int(n_tiles * 256),
            "output_tokens": len(self.tokenizer(answer).input_ids),
            "gen_seconds": round(time.time() - t0, 2),
            "oom_retries": attempt,
        }
        return answer, meta


# --model 的簡稱 -> 預設 model_id。用 --model-id 可覆寫成任何 HF repo。
DEFAULT_MODEL_IDS = {
    "qwen3": QWEN_MODEL_ID,
    "internvl3_5": INTERNVL_MODEL_ID,
}

# 哪個簡稱要用哪個 backend class（Qwen 系 / InternVL 系載入方式不同）
_BACKEND_FAMILY = {
    "qwen3": "qwen",
    "internvl3_5": "internvl",
}


def resolve_model_id(name, model_id=None):
    """(--model 簡稱, --model-id 覆寫) -> 實際 model_id。"""
    if model_id:
        return model_id
    if name not in DEFAULT_MODEL_IDS:
        raise ValueError(f"unknown model: {name}（可用：{sorted(DEFAULT_MODEL_IDS)}）")
    return DEFAULT_MODEL_IDS[name]


def build_backend(name, res_mode, max_pixels=None, max_tiles=None, model_id=None):
    model_id = resolve_model_id(name, model_id)
    family = _BACKEND_FAMILY.get(name)
    if family == "qwen":
        return QwenVLBackend(res_mode=res_mode, max_pixels=max_pixels, model_id=model_id)
    if family == "internvl":
        return InternVLBackend(res_mode=res_mode, max_tiles=max_tiles, model_id=model_id)
    raise ValueError(f"unknown model: {name}（可用：{sorted(DEFAULT_MODEL_IDS)}）")


def backend_tag(name, model_id=None):
    """不載入模型也能算出輸出資料夾名稱（給 run script 先建目錄用）。"""
    return tag_for_model_id(resolve_model_id(name, model_id))
