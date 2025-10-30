# AI Dubbing Prototype (Docker)

Korean-to-English/Japanese dubbing pipeline built on faster-whisper (ASR), Helsinki-NLP MT, and Coqui XTTS v2 TTS. The repository is prepared for GPU workers where the API stack and the TTS engine run in separate containers but share the same FastAPI application code.

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

```powershell
# Full dub (Korean -> English)
Invoke-RestMethod -Uri http://localhost:8000/dub -Method Post -Form @{
  file        = Get-Item .\sample.mp4
  target_lang = "en"
}

# Dub with custom reference voice
Invoke-RestMethod -Uri http://localhost:8000/dub -Method Post -Form @{
  file        = Get-Item .\sample.mp4
  ref_voice   = Get-Item .\ref.wav
  target_lang = "ja"
}
```

Responses include a `job_id`; inspect `./data/<job_id>/output.mp4` or call `GET /download/{job_id}` to fetch the muxed video.

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
