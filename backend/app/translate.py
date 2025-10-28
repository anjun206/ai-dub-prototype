# app/translate.py
import os
from functools import lru_cache

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

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
        "facebook/m2m100_418M",
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
    # 번역은 확정적 추론: 샘플링 금지, 빔검색 안정화
    outs = mt(
        texts,
        max_length=512,
        do_sample=False,
        num_beams=4,
        clean_up_tokenization_spaces=True,
    )
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
                out = _run_pipe(_get_mt(m), texts)
                if (src, tgt) == ("en", "ko") and any(_looks_garbled_ko(x) for x in out):
                        return _run_m2m100(texts, "en", "ko")
                return out
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

def _looks_garbled_ko(s: str) -> bool:
    # 한글 비율이 매우 낮고, 일본어 구두점/난수토큰이 섞였으면 가비지로 판단
    if not s: return True
    ja_punct = ("、", "。", "・")
    bad_hit = any(p in s for p in ja_punct)
    hangul = sum(0xAC00 <= ord(ch) <= 0xD7A3 for ch in s)
    latin  = sum(('A' <= ch <= 'Z') or ('a' <= ch <= 'z') for ch in s)
    digits = sum(ch.isdigit() for ch in s)
    total  = len(s)
    return bad_hit or (hangul/ max(1,total) < 0.15 and (latin+digits)/max(1,total) > 0.6)

def _run_m2m100(texts, src: str, tgt: str):
    name = "facebook/m2m100_418M"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(name)
    # 언어 코드 지정
    tok.src_lang = src if len(src) == 2 else "en"
    t = tok(texts, return_tensors="pt", padding=True, truncation=True)
    gen = mdl.generate(**t, forced_bos_token_id=tok.get_lang_id(tgt if len(tgt)==2 else "ko"),
                       max_length=512, num_beams=4)
    outs = tok.batch_decode(gen, skip_special_tokens=True)
    return outs
