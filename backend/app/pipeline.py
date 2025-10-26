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
    run, ffprobe_duration, make_silence, time_stretch,
    trim_or_pad_to_duration, concat_audio, replace_audio_in_video
)
from .utils_meta import load_meta, save_meta

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # base|small|medium|large-v3
TTS_MODEL = os.getenv("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")

USE_GPU = os.getenv("USE_GPU", "1") == "1"
_device = "cuda" if USE_GPU else "cpu"
_compute = "float16" if _device == "cuda" else "int8"

EPS = 0.02  # 20ms 허용오차


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
    return [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segments]

def _ensure_ref_voice(work: str, wav_16k: str, ref_voice: Optional[UploadFile]) -> str:
    ref_path = os.path.join(work, "ref.wav")
    if ref_voice is not None:
        with open(ref_path, "wb") as f:
            shutil.copyfileobj(ref_voice.file, f)
        return ref_path
    _pick_ref(wav_16k, ref_path)
    return ref_path


# ----------------- 번역 (자동 수정 없음) -----------------
def translate_stage(segments: List[Dict], src: str, tgt: str, length_mode: str = "off") -> List[Dict]:
    """
    자동 길이 조정/문자 수정 없음.
    입력 세그먼트의 절대좌표(start,end)를 그대로 복사해 번역 텍스트만 채움.
    반환: [{"start": float, "end": float, "text": str}, ...]
    """
    texts = [s["text"] for s in segments]
    base = translate_texts(texts, src=src, tgt=tgt)
    outs: List[Dict] = []
    for s, t in zip(segments, base):
        outs.append({"start": s["start"], "end": s["end"], "text": t})
    return outs


# ----------------- 1차 TTS: 길이 측정(Probe) -----------------
async def tts_probe_stage(job_id: str, target_lang: str, ref_voice: Optional[UploadFile]):
    """
    세그먼트별로 TTS를 '원문 그대로' 합성하여 실제 길이를 측정하고,
    slot(=end-start)과 비교해 over/less/fit 및 초 차이(delta)를 기록.
    오디오는 결합하지 않음. 리포트만 저장/반환.
    """
    assert target_lang in ("en", "ja")
    work = os.path.join("/app/data", job_id)
    meta = load_meta(work)
    assert "translations" in meta, "No translations"
    assert "wav_16k" in meta and "input" in meta, "Invalid meta"

    trs: List[Dict] = meta["translations"]
    ref = _ensure_ref_voice(work, meta["wav_16k"], ref_voice)

    report = []
    for i, tr in enumerate(trs):
        start = float(tr["start"]); end = float(tr["end"])
        slot = max(0.05, end - start)

        raw = os.path.join(work, f"probe_{i:04d}_raw.wav")
        synthesize(tr["text"], ref, language=target_lang, out_path=raw, model_name=TTS_MODEL)
        raw_dur = ffprobe_duration(raw)

        delta = raw_dur - slot
        if delta > EPS:
            status = "over"
        elif delta < -EPS:
            status = "less"
        else:
            status = "fit"

        report.append({
            "i": i,
            "start": start,
            "end": end,
            "slot_dur": slot,
            "raw_dur": raw_dur,
            "delta": delta,
            "status": status
        })

    meta["duration_report"] = report
    save_meta(work, meta)
    return {"workdir": work, "duration_report": report}


# ----------------- 2차 TTS: 최종 보정/결합(Finalize) -----------------
async def tts_finalize_stage(job_id: str, target_lang: str, ref_voice: Optional[UploadFile]):
    """
    다시 TTS 합성 후 각 세그먼트별로:
      - raw_dur > slot → tempo = raw/slot 로 '빠르게' 줄여 정확히 맞춤
      - raw_dur <= slot → 뒤 무음 패드로 정확히 맞춤
    절대좌표로 리딩갭/세그 간 갭을 삽입해 최종 오디오 결합.
    """
    assert target_lang in ("en", "ja")
    work = os.path.join("/app/data", job_id)
    meta = load_meta(work)
    assert "translations" in meta, "No translations"
    assert "wav_16k" in meta and "input" in meta, "Invalid meta"

    trs: List[Dict] = meta["translations"]
    ref = _ensure_ref_voice(work, meta["wav_16k"], ref_voice)

    parts: List[str] = []
    final_report: List[Dict] = []

    # 리딩 갭
    lead = max(0.0, float(trs[0]["start"]))
    if lead > 0.0001:
        lead_wav = os.path.join(work, "lead.wav")
        make_silence(lead_wav, lead, ar=24000)
        parts.append(lead_wav)

    for i, tr in enumerate(trs):
        start = float(tr["start"]); end = float(tr["end"])
        slot = max(0.05, end - start)

        raw = os.path.join(work, f"final_{i:04d}_raw.wav")
        synthesize(tr["text"], ref, language=target_lang, out_path=raw, model_name=TTS_MODEL)
        raw_dur = ffprobe_duration(raw)

        fit = os.path.join(work, f"final_{i:04d}_slot.wav")
        if raw_dur > slot + EPS:
            # over → 속도 줄여 맞춤 (빠르게: tempo = raw/slot)
            tempo = raw_dur / slot
            tmp = fit.replace(".wav", "_tempo.wav")
            time_stretch(raw, tmp, tempo=tempo, ar=24000)
            info = trim_or_pad_to_duration(tmp, fit, slot, ar=24000)
            final_report.append({
                "i": i, "mode": "speedup", "tempo": tempo,
                "raw_dur": raw_dur, "slot_dur": slot,
                "padded": info["padded"], "trimmed": info["trimmed"]
            })
        else:
            # less/fit → 무음 패드로 정확히 맞춤
            info = trim_or_pad_to_duration(raw, fit, slot, ar=24000)
            final_report.append({
                "i": i, "mode": "pad",
                "raw_dur": raw_dur, "slot_dur": slot,
                "padded": info["padded"], "trimmed": info["trimmed"]
            })

        parts.append(fit)

        # 세그 간 절대 갭
        if i < len(trs) - 1:
            next_start = float(trs[i+1]["start"])
            gap = max(0.0, next_start - end)
            if gap > 0.0001:
                g = os.path.join(work, f"gap_{i:04d}.wav")
                make_silence(g, gap, ar=24000)
                parts.append(g)

    # 24k concat → 48k 업샘플
    out24 = os.path.join(work, "dubbed_24k.wav")
    concat_audio(parts, out24)
    dubbed = os.path.join(work, "dubbed.wav")
    run(f"ffmpeg -y -i {out24} -ar 48000 -ac 1 {dubbed}")

    meta["dubbed_wav"] = dubbed
    meta["final_report"] = final_report
    save_meta(work, meta)
    return {"workdir": work, "dubbed_wav": dubbed, "final_report": final_report}


# ----------------- ASR/번역/원샷 -----------------
async def asr_only(file: UploadFile) -> Dict:
    job_id = str(uuid.uuid4())[:8]
    work = os.path.join("/app/data", job_id); _ensure_dir(work)
    in_path = os.path.join(work, file.filename)
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    wav_16k = os.path.join(work, "audio_16k.wav")
    _extract_audio_16k_mono(in_path, wav_16k)
    orig_dur = ffprobe_duration(wav_16k)

    segments = _whisper_transcribe(wav_16k)
    meta = {
        "job_id": job_id,
        "workdir": work,
        "input": in_path,
        "wav_16k": wav_16k,
        "orig_duration": orig_dur,
        "segments": segments,
    }
    save_meta(work, meta)
    return meta

def mux_stage(job_id: str) -> str:
    work = os.path.join("/app/data", job_id)
    meta = load_meta(work)
    assert "input" in meta and "dubbed_wav" in meta
    out_video = os.path.join(work, "output.mp4")
    replace_audio_in_video(meta["input"], meta["dubbed_wav"], out_video)
    meta["output"] = out_video
    save_meta(work, meta)
    return out_video


# (선택) 원샷 dub: 번역 자동수정 없이, 최종 규칙으로 바로 처리
def dub(video_in: str, target_lang: str, ref_wav: Optional[str] = None) -> Dict:
    job_id = str(uuid.uuid4())[:8]
    work = os.path.join("/app/data", job_id); _ensure_dir(work)

    # 1) 오디오 추출
    wav_16k = os.path.join(work, "audio_16k.wav")
    _extract_audio_16k_mono(video_in, wav_16k)

    # 2) 레퍼런스
    ref_path = ref_wav or os.path.join(work, "ref.wav")
    if not ref_wav:
        _pick_ref(wav_16k, ref_path)

    # 3) ASR
    segments = _whisper_transcribe(wav_16k)
    if not segments:
        raise RuntimeError("No speech detected.")

    # 4) MT (자동수정 없음) → translations에 절대좌표 포함
    base = translate_texts([s["text"] for s in segments], src="ko", tgt=target_lang)
    translations = [{"start": s["start"], "end": s["end"], "text": t} for s, t in zip(segments, base)]

    # 5) 즉시 Finalize 규칙으로 합성/보정/결합
    trs = translations
    parts: List[str] = []

    lead = max(0.0, float(trs[0]["start"]))
    if lead > 0.0001:
        lead_wav = os.path.join(work, "lead.wav")
        make_silence(lead_wav, lead, ar=24000)
        parts.append(lead_wav)

    final_report: List[Dict] = []
    for i, tr in enumerate(trs):
        start = float(tr["start"]); end = float(tr["end"])
        slot = max(0.05, end - start)

        raw = os.path.join(work, f"seg_{i:04d}_raw.wav")
        synthesize(tr["text"], ref_path, language=target_lang, out_path=raw, model_name=TTS_MODEL)
        raw_dur = ffprobe_duration(raw)

        fit = os.path.join(work, f"seg_{i:04d}_slot.wav")
        if raw_dur > slot + EPS:
            tempo = raw_dur / slot
            tmp = fit.replace(".wav", "_tempo.wav")
            time_stretch(raw, tmp, tempo=tempo, ar=24000)
            info = trim_or_pad_to_duration(tmp, fit, slot, ar=24000)
            final_report.append({"i": i, "mode": "speedup", "tempo": tempo,
                                 "raw_dur": raw_dur, "slot_dur": slot,
                                 "padded": info["padded"], "trimmed": info["trimmed"]})
        else:
            info = trim_or_pad_to_duration(raw, fit, slot, ar=24000)
            final_report.append({"i": i, "mode": "pad",
                                 "raw_dur": raw_dur, "slot_dur": slot,
                                 "padded": info["padded"], "trimmed": info["trimmed"]})

        parts.append(fit)

        if i < len(trs) - 1:
            next_start = float(trs[i+1]["start"])
            gap = max(0.0, next_start - end)
            if gap > 0.0001:
                g = os.path.join(work, f"gap_{i:04d}.wav")
                make_silence(g, gap, ar=24000)
                parts.append(g)

    out24 = os.path.join(work, "dubbed_24k.wav")
    concat_audio(parts, out24)
    dubbed_wav = os.path.join(work, "dubbed.wav")
    run(f"ffmpeg -y -i {out24} -ar 48000 -ac 1 {dubbed_wav}")

    out_video = os.path.join(work, "output.mp4")
    replace_audio_in_video(video_in, dubbed_wav, out_video)

    meta = {
        "job_id": job_id,
        "target_lang": target_lang,
        "segments": segments,
        "translations": translations,
        "workdir": work,
        "input": video_in,
        "wav_16k": wav_16k,
        "dubbed_wav": dubbed_wav,
        "final_report": final_report,
        "output": out_video
    }
    with open(os.path.join(work, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta
