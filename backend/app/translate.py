# app/translate.py
import os
from functools import lru_cache
from transformers import pipeline

def _resolve_device_index() -> int:
    """
    CUDA 사용 여부를 환경변수/실장 상태로 결정.
    - MT_DEVICE=cuda 이고 CUDA 사용 가능 => 0 (GPU 0)
    - 그 외 => -1 (CPU)
    """
    want = os.getenv("MT_DEVICE", "").lower()
    use_gpu_flag = os.getenv("USE_GPU", "1") == "1"
    if want in ("cuda", "gpu") and use_gpu_flag:
        try:
            import torch
            if torch.cuda.is_available():
                return 0  # 첫 번째 GPU
        except Exception:
            pass
    return -1

_DEVICE_INDEX = _resolve_device_index()
_USE_FP16 = os.getenv("MT_FP16", "1") == "1"  # GPU에서만 적용

@lru_cache(maxsize=8)
def _get_mt(model_name: str):
    # transformers 버전에 따라 torch_dtype 전달 방식이 조금 다를 수 있어서 안전하게 처리
    kwargs = {}
    if _DEVICE_INDEX >= 0 and _USE_FP16:
        try:
            import torch
            kwargs["torch_dtype"] = torch.float16
        except Exception:
            pass
    try:
        # 최신 버전 대부분에서 동작
        return pipeline("translation", model=model_name, device=_DEVICE_INDEX, **kwargs)
    except TypeError:
        # 구버전 호환 (torch_dtype 미지원 등)
        kwargs.pop("torch_dtype", None)
        return pipeline("translation", model=model_name, device=_DEVICE_INDEX)

def translate_texts(texts, src: str, tgt: str):
    if src == tgt:
        return texts

    if src == "ko" and tgt == "en":
        mt = _get_mt("Helsinki-NLP/opus-mt-ko-en")
        outs = mt(texts, max_length=512)
        return [o['translation_text'] for o in outs]

    if src == "ko" and tgt == "ja":
        # two-hop: ko->en -> en->ja
        mt1 = _get_mt("Helsinki-NLP/opus-mt-ko-en")
        mid = [o['translation_text'] for o in mt1(texts, max_length=512)]
        mt2 = _get_mt("Helsinki-NLP/opus-mt-en-jap")
        outs = mt2(mid, max_length=512)
        return [o['translation_text'] for o in outs]

    if src == "ja" and tgt == "en":
        mt = _get_mt("Helsinki-NLP/opus-mt-ja-en")
        outs = mt(texts, max_length=512)
        return [o['translation_text'] for o in outs]

    if src == "en" and tgt == "ja":
        mt = _get_mt("Helsinki-NLP/opus-mt-en-jap")
        outs = mt(texts, max_length=512)
        return [o['translation_text'] for o in outs]

    if src == "en" and tgt == "ko":
        mt = _get_mt("Helsinki-NLP/opus-mt-en-ko")
        outs = mt(texts, max_length=512)
        return [o['translation_text'] for o in outs]

    # fallback: identity
    return texts
