from transformers import pipeline
from functools import lru_cache

@lru_cache(maxsize=8)
def _get_mt(model_name: str):
    return pipeline("translation", model=model_name, device=-1)  # CPU 고정

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
