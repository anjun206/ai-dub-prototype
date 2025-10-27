# app/translate.py
import os
from functools import lru_cache

import torch
from transformers import pipeline

# ---------------------------
# Device / auth 설정
# ---------------------------
# MT_DEVICE: "cuda" 또는 "cpu" (기본 cuda)
_MT_DEVICE = os.getenv("MT_DEVICE", "cuda").lower()
_USE_GPU = (_MT_DEVICE == "cuda") and torch.cuda.is_available()
_DEVICE_INDEX = 0 if _USE_GPU else -1

# 선택적 토큰 (있으면 사용, 401나면 자동으로 토큰 없이 재시도)
_HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
    or None
)

_PIPE_KW = {}
if _DEVICE_INDEX >= 0:
    _PIPE_KW["device"] = _DEVICE_INDEX
    if _USE_GPU:
        _PIPE_KW["torch_dtype"] = torch.float16  # 작은 마리안 모델엔 반정밀도 OK

if _HF_TOKEN:
    _PIPE_KW["token"] = _HF_TOKEN

# ---------------------------
# 모델 우선순위 테이블
# ---------------------------
# 1순위 실패 시 다음 후보로 폴백. 없으면 2-hop(en)을 시도.
MODEL_PREFS = {
    ("ko", "en"): ["Helsinki-NLP/opus-mt-ko-en"],
    ("en", "ko"): [
        "Helsinki-NLP/opus-mt-en-ko",
        "Helsinki-NLP/opus-mt-tc-big-en-ko",  # 대체
    ],
    ("ja", "en"): ["Helsinki-NLP/opus-mt-ja-en", "Helsinki-NLP/opus-mt-jap-en"],
    ("en", "ja"): ["Helsinki-NLP/opus-mt-en-jap"],
    # ja<->ko는 보통 직통 모델이 부정확/없으므로 2-hop(en)로 처리
}

# ---------------------------
# 파이프라인 로더 (캐시)
# ---------------------------
@lru_cache(maxsize=16)
def _get_mt(model_name: str, token_key: int = 1):
    """
    token_key: 1이면 토큰 포함 시도, 0이면 토큰 제거 후 강제 시도.
    캐시 키에 token_key를 넣어 '토큰 포함/제외' 케이스를 분리.
    """
    kwargs = dict(_PIPE_KW)
    if token_key == 0 and "token" in kwargs:
        kwargs = dict(kwargs)
        kwargs.pop("token", None)
    try:
        return pipeline("translation", model=model_name, **kwargs)
    except Exception as e:
        # 토큰 때문에 401이면 토큰 없이 재시도
        if token_key == 1:
            try:
                return _get_mt(model_name, token_key=0)
            except Exception:
                pass
        raise

def _run_pipe(mt, texts):
    # transformers의 translation pipeline 호출
    # 긴 텍스트도 문장 리스트로 넣는 현재 구조에선 기본값으로 충분
    outs = mt(texts, max_length=512)
    return [o["translation_text"] for o in outs]

# ---------------------------
# 2-hop 보조 (en 경유)
# ---------------------------
def _two_hop(texts, src: str, mid: str, tgt: str):
    # 1) src->mid
    first_models = MODEL_PREFS.get((src, mid), [])
    if not first_models:
        raise RuntimeError(f"No model for hop {src}->{mid}")
    last_err = None
    for m in first_models:
        try:
            t1 = _run_pipe(_get_mt(m), texts)
            break
        except Exception as e:
            last_err = e
            continue
    else:
        # 모든 후보 실패 → 토큰 제거 재시도
        for m in first_models:
            try:
                t1 = _run_pipe(_get_mt(m, token_key=0), texts)
                break
            except Exception as e:
                last_err = e
                continue
        else:
            raise last_err or RuntimeError(f"Failed hop {src}->{mid}")

    # 2) mid->tgt
    second_models = MODEL_PREFS.get((mid, tgt), [])
    if not second_models:
        raise RuntimeError(f"No model for hop {mid}->{tgt}")
    last_err = None
    for m in second_models:
        try:
            t2 = _run_pipe(_get_mt(m), t1)
            return t2
        except Exception as e:
            last_err = e
            continue
    # 토큰 제거 재시도
    for m in second_models:
        try:
            t2 = _run_pipe(_get_mt(m, token_key=0), t1)
            return t2
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError(f"Failed hop {mid}->{tgt}")

# ---------------------------
# Public API
# ---------------------------
def translate_texts(texts, src: str, tgt: str):
    """
    - src==tgt면 그대로 반환
    - MODEL_PREFS에 직접 모델이 있으면 그걸 우선 사용
    - 없거나 실패하면 2-hop(en) 시도
    - 토큰 문제(401)가 있어도 자동으로 토큰 없는 다운로드 재시도
    - GPU 가능 시 GPU 사용
    """
    if src == tgt:
        return texts

    # 1) direct
    models = MODEL_PREFS.get((src, tgt))
    if models:
        last_err = None
        for m in models:
            try:
                return _run_pipe(_get_mt(m), texts)
            except Exception as e:
                last_err = e
                continue
        # 토큰 제거 재시도
        for m in models:
            try:
                return _run_pipe(_get_mt(m, token_key=0), texts)
            except Exception as e:
                last_err = e
                continue
        # direct 전부 실패 → 2-hop(en)로 폴백
        try:
            return _two_hop(texts, src, "en", tgt)
        except Exception:
            # 그래도 실패면 마지막 에러 전달
            raise last_err or RuntimeError(f"Direct models failed for {src}->{tgt}")

    # 2) no direct mapping → 2-hop(en)
    return _two_hop(texts, src, "en", tgt)
