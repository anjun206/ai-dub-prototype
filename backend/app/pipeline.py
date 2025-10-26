# app/pipeline.py
import os
import uuid
import json
import shutil
from typing import Optional, List, Dict

from faster_whisper import WhisperModel
from fastapi import UploadFile

from .translate import translate_texts
from .tts import synthesize
from .utils import (
    run, ffprobe_duration, extract_wav_segment, piecewise_fit, replace_audio_in_video
)
from .utils_meta import load_meta, save_meta
from .text_fit import estimate_char_budget, simple_fit

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # base|small|medium|large-v3
TTS_MODEL = os.getenv("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")

USE_GPU = os.getenv("USE_GPU", "1") == "1"
_device = "cuda" if USE_GPU else "cpu"
_compute = "float16" if _device == "cuda" else "int8"


# ----------------- 공통 유틸 -----------------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _extract_audio_16k_mono(media_path: str, wav_path: str):
    run(f"ffmpeg -y -i {media_path} -ac 1 -ar 16000 -vn -c:a pcm_s16le {wav_path}")

def _pick_ref(audio_wav_16k: str, out_ref_24k: str):
    # 6초 참조 (24k mono)
    run(f"ffmpeg -y -i {audio_wav_16k} -t 6 -ar 24000 -ac 1 {out_ref_24k}")

def _whisper_transcribe(audio_wav_16k: str):
    model = WhisperModel(WHISPER_MODEL, device=_device, compute_type=_compute)
    segments, info = model.transcribe(audio_wav_16k, language=None, vad_filter=True, word_timestamps=False)
    out = []
    for seg in segments:
        out.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()})
    return out

def _absolute_timeline_concat(work: str, parts_24k: List[str]) -> str:
    """
    입력 WAV 리스트(24k mono)를 그대로 concat(copy) → 24k mono
    호출 전 parts_24k에는 [leading_sil?, seg0_fit, seg1_fit, ...] 등 '절대 배치 결과'가 순서대로 들어있어야 함.
    """
    list_path = os.path.join(work, "conc.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in parts_24k:
            f.write(f"file '{os.path.basename(p)}'\n")
    out_24k = os.path.join(work, "dubbed_24k.wav")
    run(f"ffmpeg -y -f concat -safe 0 -i {list_path} -c copy {out_24k}")
    # 최종 48k 리샘플
    out_48k = os.path.join(work, "dubbed.wav")
    run(f"ffmpeg -y -i {out_24k} -ar 48000 -ac 1 {out_48k}")
    return out_48k


# ----------------- 단계형 API -----------------
async def asr_only(file: UploadFile) -> Dict:
    job_id = str(uuid.uuid4())[:8]
    work = os.path.join("/app/data", job_id); _ensure_dir(work)
    in_path = os.path.join(work, file.filename)
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 오디오 추출(ASR용 16k mono)
    wav_16k = os.path.join(work, "audio_16k.wav")
    _extract_audio_16k_mono(in_path, wav_16k)

    # ASR
    segments = _whisper_transcribe(wav_16k)
    meta = {
        "job_id": job_id,
        "workdir": work,
        "input": in_path,
        "wav_16k": wav_16k,
        "segments": segments,
    }
    save_meta(work, meta)
    return meta

def translate_stage(segments: List[Dict], src: str, tgt: str, length_mode: str = "auto") -> List[str]:
    texts = [s["text"] for s in segments]
    base = translate_texts(texts, src=src, tgt=tgt)
    if length_mode != "auto":
        return base

    # 길이 맞춤(휴리스틱)
    outs = []
    for s, t in zip(segments, base):
        dur = max(0.2, s["end"] - s["start"])
        budget = estimate_char_budget(dur, tgt)
        outs.append(simple_fit(t, budget))
    return outs

def _ensure_ref_voice(work: str, wav_16k: str, ref_voice: Optional[UploadFile]) -> str:
    ref_path = os.path.join(work, "ref.wav")
    if ref_voice is not None:
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(ref_voice.file, f)
        return ref_path
    _pick_ref(wav_16k, ref_path)
    return ref_path

async def tts_stage(job_id: str, target_lang: str, ref_voice: Optional[UploadFile]):
    assert target_lang in ("en", "ja")
    work = os.path.join("/app/data", job_id)
    meta = load_meta(work)
    assert "segments" in meta, "No segments"
    assert "wav_16k" in meta, "No wav_16k"
    assert "input" in meta, "No input"
    assert "translations" in meta, "No translations"

    # 레퍼런스 보이스
    ref = _ensure_ref_voice(work, meta["wav_16k"], ref_voice)

    # 절대 타임라인 오디오 생성 (24k 기반 → 마지막에 48k로)
    parts = []
    # ① 맨 앞 무음(leading gap)
    lead = max(0.0, meta["segments"][0]["start"])
    if lead > 0.0001:
        lead_wav = os.path.join(work, "lead.wav")
        run(f"ffmpeg -y -f lavfi -i anullsrc=r=24000:cl=mono -t {lead:.3f} {lead_wav}")
        parts.append(lead_wav)

    for i, (seg, txt) in enumerate(zip(meta["segments"], meta["translations"])):
        raw = os.path.join(work, f"seg_{i:04d}_raw.wav")
        fit = os.path.join(work, f"seg_{i:04d}_fit.wav")
        # 합성 (XTTS v2)
        synthesize(txt, ref, language=target_lang, out_path=raw, model_name=TTS_MODEL)
        # 참조 세그먼트(24k mono) 추출 (원본 시간 구조)
        ref_seg = os.path.join(work, f"seg_{i:04d}_ref.wav")
        extract_wav_segment(meta["wav_16k"], ref_seg, seg["start"], seg["end"])
        # 조각 맞춤 (발화/무음 구조 맞추기)
        piecewise_fit(raw, ref_seg, fit)
        parts.append(fit)

    dubbed = _absolute_timeline_concat(work, parts)
    meta["dubbed_wav"] = dubbed
    save_meta(work, meta)
    return {"workdir": work, "dubbed_wav": dubbed}

def mux_stage(job_id: str) -> str:
    work = os.path.join("/app/data", job_id)
    meta = load_meta(work)
    assert "input" in meta and "dubbed_wav" in meta
    out_video = os.path.join(work, "output.mp4")
    # 오디오 PTS 정규화 + aac 48k + faststart
    run(
        f'ffmpeg -y -i {meta["input"]} -i {meta["dubbed_wav"]} '
        f'-map 0:v:0 -map 1:a:0 -c:v copy '
        f'-af "asetpts=PTS-STARTPTS" -c:a aac -ar 48000 -b:a 192k '
        f'-shortest -movflags +faststart {out_video}'
    )
    meta["output"] = out_video
    save_meta(work, meta)
    return out_video


# ----------------- 원샷 dub(개선) -----------------
def dub(video_in: str, target_lang: str, ref_wav: Optional[str] = None) -> Dict:
    job_id = str(uuid.uuid4())[:8]
    work = os.path.join("/app/data", job_id); _ensure_dir(work)

    # 1) 오디오 추출(ASR용 16k mono)
    wav_16k = os.path.join(work, "audio_16k.wav")
    _extract_audio_16k_mono(video_in, wav_16k)

    # 2) 레퍼런스 보이스(24k mono)
    ref_path = ref_wav or os.path.join(work, "ref.wav")
    if not ref_wav:
        _pick_ref(wav_16k, ref_path)

    # 3) ASR
    segments = _whisper_transcribe(wav_16k)
    if not segments:
        raise RuntimeError("No speech detected.")

    # 4) MT (간단 ko->(en|ja) 가정, 길이자동은 생략 / 원샷은 라이트하게)
    texts = [s["text"] for s in segments]
    translated = translate_texts(texts, src="ko", tgt=target_lang)

    # 5) TTS + piecewise-fit + 절대 배치
    parts = []
    lead = max(0.0, segments[0]["start"])
    if lead > 0.0001:
        lead_wav = os.path.join(work, "lead.wav")
        run(f"ffmpeg -y -f lavfi -i anullsrc=r=24000:cl=mono -t {lead:.3f} {lead_wav}")
        parts.append(lead_wav)

    for i, (seg, txt) in enumerate(zip(segments, translated)):
        raw = os.path.join(work, f"seg_{i:04d}_raw.wav")
        fit = os.path.join(work, f"seg_{i:04d}_fit.wav")
        synthesize(txt, ref_path, language=target_lang, out_path=raw, model_name=TTS_MODEL)
        ref_seg = os.path.join(work, f"seg_{i:04d}_ref.wav")
        extract_wav_segment(wav_16k, ref_seg, seg["start"], seg["end"])
        piecewise_fit(raw, ref_seg, fit)
        parts.append(fit)

    dubbed_wav = _absolute_timeline_concat(work, parts)

    # 6) 비디오와 합치기
    out_video = os.path.join(work, "output.mp4")
    run(
        f'ffmpeg -y -i {video_in} -i {dubbed_wav} '
        f'-map 0:v:0 -map 1:a:0 -c:v copy '
        f'-af "asetpts=PTS-STARTPTS" -c:a aac -ar 48000 -b:a 192k '
        f'-shortest -movflags +faststart {out_video}'
    )

    # Save metadata
    meta = {
        "job_id": job_id,
        "target_lang": target_lang,
        "segments": segments,
        "workdir": work,
        "input": video_in,
        "wav_16k": wav_16k,
        "dubbed_wav": dubbed_wav,
        "output": out_video
    }
    with open(os.path.join(work, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta
