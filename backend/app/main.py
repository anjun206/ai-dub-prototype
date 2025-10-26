# app/main.py
import os

# 🔴 TTS를 CPU로 돌릴 땐 cuDNN 로딩 자체를 사전에 차단해야 함
if os.getenv("TTS_DEVICE", "cpu").lower() == "cpu":
    import torch
    torch.backends.cudnn.enabled = False

# ✅ 추가: torchaudio가 torchcodec 대신 soundfile 백엔드를 쓰도록 강제
#    (requirements.txt에 soundfile 있고, 이미지에 libsndfile1 설치되어 있으므로 OK)
try:
    import torchaudio
    torchaudio.set_audio_backend("soundfile")
except Exception as e:
    print("WARN: torchaudio.set_audio_backend('soundfile') failed:", e)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# Swagger가 localhost/127.0.0.1 혼선을 만들지 않도록 상대 경로 서버로 고정
app = FastAPI(title="AI Dub Prototype", version="0.1.0", servers=[{"url": "/"}])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 필요시 특정 오리진만 허용해도 됨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⬇️ cuDNN 비활성화 후에 pipeline을 import (pipeline -> tts 경유)
from .pipeline import dub
import shutil

@app.get("/health")
def health():
    return {"ok": True}

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
