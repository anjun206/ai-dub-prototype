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
    trim_or_pad_to_duration, concat_audio, replace_audio_in_video,
    split_audio_by_targets, 
)
from .utils_meta import load_meta, save_meta

from typing import Tuple

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
    assert target_lang in ("en", "ja")
    work = os.path.join("/app/data", job_id)
    meta = load_meta(work)
    assert "translations" in meta and "wav_16k" in meta and "input" in meta

    trs: List[Dict] = meta["translations"]
    ref = _ensure_ref_voice(work, meta["wav_16k"], ref_voice)
    layout = meta.get("merge_layouts", {})  # new_index 기준

    parts: List[str] = []
    final_report: List[Dict] = []

    # 리딩 갭
    lead = max(0.0, float(trs[0]["start"]))
    if lead > 0.0001:
        lead_wav = os.path.join(work, "lead.wav"); make_silence(lead_wav, lead, ar=24000); parts.append(lead_wav)

    for i, tr in enumerate(trs):
        start = float(tr["start"]); end = float(tr["end"])
        slot = max(0.05, end - start)

        # 이 번역 세그가 머지된 것인지 확인
        lay = layout.get(i)
        if lay and len(lay.get("slots", [])) > 1:
            # 🔹 머지된 세그먼트: 텍스트 1번 합성 → 원본 슬롯 길이에 따라 '스마트 컷'
            raw_all = os.path.join(work, f"final_{i:04d}_raw_all.wav")
            synthesize(tr["text"], ref, language=target_lang, out_path=raw_all, model_name=TTS_MODEL)
            slot_durs = [max(0.05, float(s["end"]) - float(s["start"])) for s in lay["slots"]]
            inner_gaps = [max(0.0, float(g)) for g in lay.get("gaps", [])]

            # 무음 스냅 컷
            chunks = split_audio_by_targets(raw_all, slot_durs, work, f"final_{i:04d}")
            # 각 슬롯별 규칙 적용(오버=배속, 레스=패드)
            for j, ch in enumerate(chunks):
                ch_dur = ffprobe_duration(ch); tgt = slot_durs[j]
                outp = os.path.join(work, f"final_{i:04d}_slot_{j:02d}.wav")
                if ch_dur > tgt + EPS:
                    tempo = ch_dur / tgt
                    tmp = outp.replace(".wav", "_tempo.wav")
                    time_stretch(ch, tmp, tempo=tempo, ar=24000)
                    info = trim_or_pad_to_duration(tmp, outp, tgt, ar=24000)
                    final_report.append({"i": i, "sub": j, "mode": "speedup", "tempo": tempo,
                                         "raw_dur": ch_dur, "slot_dur": tgt, "padded": info["padded"], "trimmed": info["trimmed"]})
                else:
                    info = trim_or_pad_to_duration(ch, outp, tgt, ar=24000)
                    final_report.append({"i": i, "sub": j, "mode": "pad",
                                         "raw_dur": ch_dur, "slot_dur": tgt, "padded": info["padded"], "trimmed": info["trimmed"]})
                parts.append(outp)

                # 내부 갭 삽입
                if j < len(inner_gaps):
                    gap = inner_gaps[j]
                    if gap > 0.0001:
                        g = os.path.join(work, f"final_{i:04d}_gap_{j:02d}.wav")
                        make_silence(g, gap, ar=24000)
                        parts.append(g)
        else:
            # 🔸 일반(머지 아님 or 단일 슬롯) 처리: 기존 규칙
            raw = os.path.join(work, f"final_{i:04d}_raw.wav")
            synthesize(tr["text"], ref, language=target_lang, out_path=raw, model_name=TTS_MODEL)
            raw_dur = ffprobe_duration(raw)

            fit = os.path.join(work, f"final_{i:04d}_slot.wav")
            if raw_dur > slot + EPS:
                tempo = raw_dur / slot
                tmp = fit.replace(".wav", "_tempo.wav")
                time_stretch(raw, tmp, tempo=tempo, ar=24000)
                info = trim_or_pad_to_duration(tmp, fit, slot, ar=24000)
                final_report.append({"i": i, "mode": "speedup", "tempo": tempo,
                                     "raw_dur": raw_dur, "slot_dur": slot, "padded": info["padded"], "trimmed": info["trimmed"]})
            else:
                info = trim_or_pad_to_duration(raw, fit, slot, ar=24000)
                final_report.append({"i": i, "mode": "pad",
                                     "raw_dur": raw_dur, "slot_dur": slot, "padded": info["padded"], "trimmed": info["trimmed"]})
            parts.append(fit)

        # 번역 세그 간 외부 절대 갭
        if i < len(trs) - 1:
            next_start = float(trs[i+1]["start"])
            gap = max(0.0, next_start - end)
            if gap > 0.0001:
                g = os.path.join(work, f"gap_{i:04d}.wav")
                make_silence(g, gap, ar=24000)
                parts.append(g)

    # 24k concat → 48k
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

    # 🔹 세그먼트 간 무음(gap_after) 기록
    for i in range(len(segments)):
        if i < len(segments) - 1:
            segments[i]["gap_after"] = max(0.0, float(segments[i+1]["start"]) - float(segments[i]["end"]))
        else:
            segments[i]["gap_after"] = 0.0

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

def _apply_merges(segments: List[Dict], merges: List[Tuple[int,int]]) -> Tuple[List[Dict], List[Dict]]:
    """
    merges: [(start_idx, end_idx), ...] 0-based, inclusive, 오름차순/비겹침/연속 범위만 허용
    반환: (merged_segments, merge_map)
      - merged_segments: [{"start":..,"end":..,"text":..}, ...]
      - merge_map: [{"new_index": k, "from": [i..j]} ...]
    """
    n = len(segments)
    if n == 0:
        return [], []

    # 정규화 & 검증
    norm = []
    for s, e in merges or []:
        s, e = int(s), int(e)
        if not (0 <= s <= e < n):
            raise ValueError(f"merge range out of bounds: ({s},{e}) with n={n}")
        norm.append((s, e))
    norm.sort(key=lambda x: x[0])

    # 겹치거나 역순/중첩 금지, 반드시 오름차순 비중첩
    for i in range(1, len(norm)):
        prev = norm[i-1]; cur = norm[i]
        if cur[0] <= prev[1]:
            raise ValueError(f"overlapping merges: {prev} and {cur}")

    merged = []
    mapping = []
    cur_i = 0
    new_idx = 0

    def _append_original(idx):
        nonlocal new_idx
        merged.append({
            "start": float(segments[idx]["start"]),
            "end": float(segments[idx]["end"]),
            "text": segments[idx]["text"],
        })
        mapping.append({"new_index": new_idx, "from": [idx]})
        new_idx += 1

    def _append_merged(a, b):
        nonlocal new_idx
        start = float(segments[a]["start"])
        end   = float(segments[b]["end"])
        # 텍스트는 공백 하나로 연결
        text = " ".join(segments[k]["text"].strip() for k in range(a, b+1) if segments[k]["text"].strip())
        merged.append({"start": start, "end": end, "text": text})
        mapping.append({"new_index": new_idx, "from": list(range(a, b+1))})
        new_idx += 1

    for rng in norm:
        a, b = rng
        # 병합 구간 전의 원본 세그먼트들 추가
        while cur_i < a:
            _append_original(cur_i)
            cur_i += 1
        # 병합 구간 추가
        _append_merged(a, b)
        cur_i = b + 1

    # 남은 꼬리 구간 추가
    while cur_i < n:
        _append_original(cur_i)
        cur_i += 1

    return merged, mapping


def merge_segments_stage(job_id: str, merges: Optional[List[List[int]]] = None):
    work = os.path.join("/app/data", job_id)
    meta = load_meta(work)
    if "segments" not in meta:
        raise RuntimeError("No ASR segments to merge")

    src = meta["segments"]
    plan = merges if merges is not None else meta.get("merge_plan")
    if not plan:
        return {"segments": src, "merge_map": [], "note": "no merges applied"}

    # 정규화 & 정렬
    rngs = [(int(a), int(b)) for a,b in plan]
    rngs.sort(key=lambda x: x[0])
    for i in range(1, len(rngs)):
        if rngs[i][0] <= rngs[i-1][1]:
            raise ValueError(f"overlapping merges: {rngs[i-1]} and {rngs[i]}")

    merged = []
    merge_map = []
    merge_layouts = {}  # new_index -> {"from":[...], "slots":[{start,end}], "gaps":[...]}
    cur = 0; new_idx = 0

    def append_original(k):
        nonlocal new_idx
        merged.append({"start": src[k]["start"], "end": src[k]["end"], "text": src[k]["text"]})
        merge_map.append({"new_index": new_idx, "from": [k]})
        # 단일 슬롯(머지 아님)도 레이아웃을 둬두면 일관 처리 쉬움
        merge_layouts[new_idx] = {
            "from": [k],
            "slots": [{"start": src[k]["start"], "end": src[k]["end"]}],
            "gaps":  []
        }
        new_idx += 1

    def append_merged(a,b):
        nonlocal new_idx
        start = float(src[a]["start"]); end = float(src[b]["end"])
        text = " ".join(s["text"].strip() for s in src[a:b+1] if s["text"].strip())
        merged.append({"start": start, "end": end, "text": text})
        merge_map.append({"new_index": new_idx, "from": list(range(a,b+1))})
        slots = [{"start": src[k]["start"], "end": src[k]["end"]} for k in range(a,b+1)]
        gaps  = [max(0.0, float(src[k+1]["start"]) - float(src[k]["end"])) for k in range(a, b)]
        merge_layouts[new_idx] = {"from": list(range(a,b+1)), "slots": slots, "gaps": gaps}
        new_idx += 1

    for a,b in rngs:
        while cur < a:
            append_original(cur)
            cur += 1
        append_merged(a,b)
        cur = b+1
    while cur < len(src):
        append_original(cur)
        cur += 1

    # 백업 & 기록
    if "segments_backup" not in meta:
        meta["segments_backup"] = meta["segments"]
    meta["segments"] = merged
    meta["merge_history"] = meta.get("merge_history", []) + [{"plan": plan, "merge_map": merge_map}]
    meta["merge_layouts"] = merge_layouts

    # 이전 산물 무효화
    for k in ("translations","duration_report","final_report","dubbed_wav","output"):
        meta.pop(k, None)

    save_meta(work, meta)
    return {"segments": merged, "merge_map": merge_map}
