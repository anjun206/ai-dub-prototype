import os
import uuid
import json
from typing import Optional, List, Dict
from faster_whisper import WhisperModel
from .translate import translate_texts
from .tts import synthesize
from .utils import run, ffprobe_duration, time_stretch, concat_audio, replace_audio_in_video

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  # base|small|medium|large-v3
TTS_MODEL = os.getenv("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")  # Coqui XTTS v2

USE_GPU = os.getenv("USE_GPU", "1") == "1"
_device = "cuda" if USE_GPU else "cpu"
_compute = "float16" if _device == "cuda" else "int8"

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _extract_audio(video_path: str, wav_path: str):
    run(f"ffmpeg -y -i {video_path} -ac 1 -ar 16000 -vn -c:a pcm_s16le {wav_path}")

def _pick_ref(audio_wav: str, out_ref: str):
    # Simple: take first 6s as reference (prototype). In practice use VAD+diarization.
    run(f"ffmpeg -y -i {audio_wav} -t 6 {out_ref}")

def _whisper_transcribe(audio_wav: str):
    model = WhisperModel(WHISPER_MODEL, device=_device, compute_type=_compute)
    segments, info = model.transcribe(audio_wav, language="ko", vad_filter=True, word_timestamps=False)
    out = []
    for seg in segments:
        out.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()})
    return out

def _stretch_to_duration(in_wav: str, target_seconds: float, out_wav: str):
    dur = ffprobe_duration(in_wav)
    if dur <= 0.0:
        # fallback copy & trust concat will handle
        run(f"ffmpeg -y -i {in_wav} -ar 24000 -ac 1 {out_wav}")
        return dur
    ratio = max(0.25, min(4.0, target_seconds / dur))
    time_stretch(in_wav, out_wav, ratio)
    return dur

def dub(video_in: str, target_lang: str, ref_wav: Optional[str]=None) -> Dict:
    job_id = str(uuid.uuid4())[:8]
    work = os.path.join("/app/data", job_id)
    _ensure_dir(work)
    # 1) extract audio
    wav_16k = os.path.join(work, "audio_16k.wav")
    _extract_audio(video_in, wav_16k)

    # 2) reference voice
    ref_path = ref_wav or os.path.join(work, "ref.wav")
    if not ref_wav:
        _pick_ref(wav_16k, ref_path)

    # 3) ASR
    segments = _whisper_transcribe(wav_16k)
    if not segments:
        raise RuntimeError("No speech detected.")

    # 4) MT
    texts = [s["text"] for s in segments]
    translated = translate_texts(texts, src="ko", tgt=target_lang)

    # 5) TTS per segment
    seg_files = []
    for i, (seg, txt) in enumerate(zip(segments, translated)):
        raw = os.path.join(work, f"seg_{i:04d}_raw.wav")
        out = os.path.join(work, f"seg_{i:04d}_fit.wav")
        synthesize(txt, ref_path, language=target_lang, out_path=raw, model_name=TTS_MODEL)
        seg_dur = _stretch_to_duration(raw, seg["end"] - seg["start"], out)
        # gap to next start
        gap = 0.0
        if i < len(segments) - 1:
            gap = max(0.0, segments[i+1]["start"] - seg["end"])  # keep original pauses
        seg_files.append((out, gap))

    dubbed_wav = concat_audio(seg_files)

    # 6) mux into video
    out_video = os.path.join(work, "output.mp4")
    replace_audio_in_video(video_in, dubbed_wav, out_video)

    # Save metadata
    meta = {
        "job_id": job_id,
        "target_lang": target_lang,
        "segments": segments,
        "workdir": work,
        "output": out_video
    }
    with open(os.path.join(work, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta
