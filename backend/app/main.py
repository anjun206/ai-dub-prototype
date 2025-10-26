# app/main.py
import os

# 🔴 TTS를 CPU로 돌릴 땐 cuDNN 로딩 자체를 사전에 차단
if os.getenv("TTS_DEVICE", "cpu").lower() == "cpu":
    import torch
    torch.backends.cudnn.enabled = False

# ✅ torchaudio가 soundfile 백엔드를 쓰도록 강제
try:
    import torchaudio
    torchaudio.set_audio_backend("soundfile")
except Exception as e:
    print("WARN: torchaudio.set_audio_backend('soundfile') failed:", e)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Swagger가 localhost/127.0.0.1 혼선을 만들지 않도록 상대 경로 서버로 고정
app = FastAPI(title="AI Dub Prototype", version="0.2.0", servers=[{"url": "/"}])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 필요시 특정 오리진만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 파이프라인 단계 함수들
from .pipeline import dub, asr_only, translate_stage, tts_stage, mux_stage
from .utils_meta import load_meta, save_meta
import shutil
import os

# ---------- 모델 (요청 바디) ----------
class SegmentPatch(BaseModel):
    i: int
    text: str

class SegmentsPatch(BaseModel):
    items: list[SegmentPatch]

class TranslateBody(BaseModel):
    src: str
    tgt: str
    length_mode: Optional[str] = "auto"  # "auto" | "off"

class TranslationPatch(BaseModel):
    i: int
    text: str

class TranslationsPatch(BaseModel):
    items: list[TranslationPatch]

# ---------- Health ----------
@app.get("/health")
def health():
    return {"ok": True}

# ---------- 원샷 dub (개선버전) ----------
@app.post("/dub")
async def dub_endpoint(
    file: UploadFile = File(..., description="Video or audio file"),
    target_lang: str = Form(..., description="'en' or 'ja'"),
    ref_voice: Optional[UploadFile] = File(None, description="Optional reference WAV (>=6s)"),
):
    assert target_lang in ("en", "ja"), "target_lang must be 'en' or 'ja'"
    tmp_dir = "/app/data/tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    in_path = os.path.join(tmp_dir, file.filename)
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ref_path = None
    if ref_voice is not None:
        ref_path = os.path.join(tmp_dir, ref_voice.filename)
        with open(ref_path, "wb") as f2:
            shutil.copyfileobj(ref_voice.file, f2)

    meta = dub(in_path, target_lang=target_lang, ref_wav=ref_path)
    return JSONResponse({"job_id": meta["job_id"], "output": meta["output"], "workdir": meta["workdir"]})

@app.get("/download/{job_id}")
def download(job_id: str):
    file_path = f"/app/data/{job_id}/output.mp4"
    if not os.path.exists(file_path):
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(file_path, media_type="video/mp4", filename=f"dub_{job_id}.mp4")

# ---------- 단계형 파이프라인 ----------
# 1) ASR만 수행 → 세그먼트 반환 (사용자 수정 단계)
@app.post("/asr")
async def asr_endpoint(
    file: UploadFile = File(..., description="Video or audio file"),
):
    meta = await asr_only(file)  # 파일 저장 + 오디오 추출 + STT
    return JSONResponse({"job_id": meta["job_id"], "segments": meta["segments"], "workdir": meta["workdir"]})

# 1.5) 세그먼트 텍스트 수정
@app.patch("/segments/{job_id}")
def patch_segments(job_id: str, body: SegmentsPatch):
    workdir = f"/app/data/{job_id}"
    meta = load_meta(workdir)
    if "segments" not in meta:
        return JSONResponse({"error": "No ASR segments for this job"}, status_code=400)

    segs = meta["segments"]
    for item in body.items:
        if 0 <= item.i < len(segs):
            segs[item.i]["text"] = item.text.strip()
    meta["segments"] = segs
    save_meta(workdir, meta)
    return {"ok": True, "segments": segs}

# 2) 번역 수행 (자동 길이 맞춤 옵션 포함)
@app.post("/translate/{job_id}")
def translate_endpoint(job_id: str, body: TranslateBody):
    workdir = f"/app/data/{job_id}"
    meta = load_meta(workdir)
    if "segments" not in meta:
        return JSONResponse({"error": "No ASR segments"}, status_code=400)

    translations = translate_stage(meta["segments"], src=body.src, tgt=body.tgt,
                                   length_mode=(body.length_mode or "auto"))
    meta["translations"] = translations
    meta.setdefault("options", {})["lang"] = {"src": body.src, "tgt": body.tgt}
    save_meta(workdir, meta)
    return {"ok": True, "translations": translations}

# 2.5) 번역 텍스트 수정
@app.patch("/translations/{job_id}")
def patch_translations(job_id: str, body: TranslationsPatch):
    workdir = f"/app/data/{job_id}"
    meta = load_meta(workdir)
    if "translations" not in meta:
        return JSONResponse({"error": "No translations"}, status_code=400)
    trs = meta["translations"]
    for item in body.items:
        if 0 <= item.i < len(trs):
            trs[item.i] = item.text.strip()
    meta["translations"] = trs
    save_meta(workdir, meta)
    return {"ok": True, "translations": trs}

# 3) TTS 실행 (ref_voice 선택)
@app.post("/tts/{job_id}")
async def tts_endpoint(
    job_id: str,
    target_lang: str = Form(..., description="'en' or 'ja'"),
    ref_voice: Optional[UploadFile] = File(None, description="Optional reference WAV (>=6s)")
):
    out = await tts_stage(job_id, target_lang=target_lang, ref_voice=ref_voice)
    return {"ok": True, **out}

# 4) 비디오와 합치기
@app.post("/mux/{job_id}")
def mux_endpoint(job_id: str):
    path = mux_stage(job_id)
    return {"ok": True, "output": path}
