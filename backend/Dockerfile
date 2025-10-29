FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_PREFER_BINARY=1

# 시스템 패키지 + FFmpeg dev (PyAV 소스빌드 필요시 대비)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    build-essential git git-lfs curl ca-certificates \
    ffmpeg sox libsndfile1 libsndfile1-dev \
    libgl1 libglib2.0-0 \
    ninja-build cmake pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev \
 && rm -rf /var/lib/apt/lists/* \
 && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 \
 && python -m pip install --upgrade pip setuptools wheel

# ✅ Blackwell 대비: 확장 모듈 빌드 시 PTX 폴백 허용
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"

# ✅ CUDA 12.8용 PyTorch 스택
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# 프로젝트 의존성 먼저 설치 (여기에 faster-whisper==1.0.3 포함)
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- OpenVoice V2 + MeloTTS ----
# 핵심: OpenVoice를 --no-deps로 설치하여 faster-whisper==0.9.0/av==10.* 강제설치 차단
RUN git lfs install && \
    git clone --depth=1 https://github.com/myshell-ai/OpenVoice.git /tmp/OpenVoice && \
    pip install --no-cache-dir --no-deps -e /tmp/OpenVoice && \
    pip install --no-cache-dir git+https://github.com/myshell-ai/MeloTTS.git

# (안전하게) 한국어 g2p 보강
RUN pip install --no-cache-dir jamo==0.4.1 g2pkk>=0.1.1

# OpenVoice V2 체크포인트 미리 다운로드
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="myshell-ai/OpenVoiceV2",
    allow_patterns=["converter/*", "base_speakers/**"],
    local_dir="/app/assets/openvoice_v2/checkpoints_v2"
)
PY

COPY app /app/app

ENV HF_HOME=/app/cache/hf \
    TRANSFORMERS_CACHE=/app/cache/hf \
    HUGGINGFACE_HUB_CACHE=/app/cache/hf \
    TTS_HOME=/app/cache/tts \
    MT_KO_EN_MODEL="Helsinki-NLP/opus-mt-tc-big-ko-en"

RUN useradd -m appuser && mkdir -p /app/cache/hf /app/cache/tts && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
