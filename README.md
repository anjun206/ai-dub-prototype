# AI Dubbing Prototype (Docker)

한국어 영상 → 영어/일본어 더빙 **프로토타입**입니다.  
파이프라인: **FFmpeg 추출 → faster-whisper(STT) → facebook/m2m100_418M 번역 → Coqui XTTS v2(보이스 클로닝 TTS) → 타임스트레치/컨캣 → 영상에 오디오 교체**

## Docker Layout

```
.
|-- backend/
|   |-- app/                     # FastAPI application (shared source)
|   |-- Dockerfile.api           # API/ASR image definition
|   |-- Dockerfile.tts           # TTS-only image definition
|   |-- requirements.api.txt     # Dependencies for the API container
|   `-- requirements.tts.txt     # Dependencies for the TTS container
`-- docker-compose.yml           # Spins up api + tts services together
```

### Compose Services

| Service | Role | Ports | Notes |
|---------|------|-------|-------|
| `api` | Handles REST requests, runs ASR/translation/mixing, forwards synthesis to TTS over HTTP | `8000` | `TTS_URL` points to the `tts` service; defaults to GPU for Whisper/MT |
| `tts` | Hosts XTTS v2 endpoints (mainly `/tts-single`) | `9000` | Shares the same app code; optimized for GPU synthesis |

### Build & Run

```bash
docker compose build      # Build both images
docker compose up -d      # Launch api + tts containers
```

Interactive docs remain at http://localhost:8000/docs.

## Dependency Breakdown

### backend/requirements.api.txt

- `fastapi`, `uvicorn[standard]`, `python-multipart`: web server core and multipart form uploads.
- `faster-whisper`, `onnxruntime-gpu`: accelerated ASR inference on CUDA.
- `transformers`, `sentencepiece`, `huggingface_hub`: translation model runtime and tokenizer support.
- `soundfile`: WAV IO for intermediate audio assets.
- `sacremoses`, `cutlet`, `fugashi`, `unidic-lite`: Japanese tokenization/romanization helpers used in translation stages.
- `requests`: API container calls the remote TTS worker over HTTP.
- `webrtcvad`: Voice-activity detection fallback for silence trimming.
- `torchcodec`: referenced utilities for audio manipulation in the pipeline.

### backend/requirements.tts.txt

- `fastapi`, `uvicorn[standard]`, `python-multipart`: same FastAPI surface hosting `/tts-single`.
- `TTS`: Coqui XTTS v2 synthesis library.
- `soundfile`: read/write support for the generated waveforms.
- `requests`: keeps parity with the API environment when issuing auxiliary HTTP calls.

## docker-compose Overview

`docker-compose.yml` builds both images from the repository root, mounts the `backend/app` directory for live code reloads, and shares host caches (`./data/hf_cache`, `./data/tts_cache`, `./data/demucs_cache`) so model downloads persist between runs. The `api` service depends on `tts`, ensuring the synthesizer is ready before accepting external traffic.

Key environment variables:

- `TTS_URL` points the API container at the TTS worker (`http://tts:9000`) so synthesis happens remotely.
- `USE_GPU`, `MT_DEVICE`, `TTS_DEVICE`, `NVIDIA_VISIBLE_DEVICES` toggle CUDA usage; set explicit device IDs to split workloads across GPUs.
- `HF_HOME`, `TRANSFORMERS_CACHE`, `TTS_HOME`, `DEMUCS_CACHE` align cache paths with host bind mounts to avoid re-downloading weights.
- `MT_*` knobs control translation beam search and batching to balance speed and accuracy.

## API Quick Start

## 🧪 사용 방법

localhost:8000/docs에 들어가서 swagger를 통해 사용가능

- POST: asr<br>
mp4 파일 업로드시 배경음 추출, stt후 meta.json 생성
meta.json에 폴더 id 기록됨

- POST: translate<br>
meta.json의 원문 번역

PATCH 붙은 애들은 보통 수정인데 그냥 meta.json에서 직접 수정하는 거 권장

- POST: tts-probe<br>
tts 간단히 돌려 원문과 변역본 발화 시간 비교
(여러번 가능)

- POST: tts-finalize<br>
최종 tts로 어긋나는 발화시간은 공백 시간 활용, 음성 배속/감속으로 싱크 맞춤

- POST: tts-single<br>
간단하게 텍스트 넣고 tts만 사용가능

- POST: mux<br>
tts-finalize 이후 영상과 오디오 합성
 -> `output`

- POST: merge<br>
asr 이후 세그먼트 블록 병합 가능 (0번부터 시작)

- POST: voice-sample<br>
업로드한 파일에서 배경음 제거해 음성만 추출 .wav로 만듦

## Voice Sample Utility

- Call `POST /voice-sample` with any media file to generate a six-second reference clip stored at `./data/<job_id>/voice_sample_24k.wav`.
- Useful for creating voice references that can be reused with `/dub`, `/tts-probe`, or `/tts-single`.

```powershell
Invoke-WebRequest -Uri http://localhost:8000/voice-sample -Method Post -Form @{
  file = Get-Item .\sample.mp4
} -OutFile voice_sample.wav
```

## Storage Layout

- `./data`: per-job workspace (inputs, intermediates, final renders) shared across containers.
- `./data/hf_cache`, `./data/tts_cache`, `./data/demucs_cache`: model caches for Hugging Face, Coqui XTTS, and Demucs separation (safe to clear when reclaiming disk).
