# AI Dubbing Prototype (Docker)

한국어/영어/일본어 영상 더빙 **프로토타입**입니다.  
업데이트: **동물 울음/환호 등 비언어 음향을 보존**하도록 전처리 파이프라인을 개선했습니다.

## 🧭 파이프라인 개요
**FFmpeg 추출 → Demucs(보컬/배경 분리) → PANNs(동물 울음 구간 검출) → 말/동물/배경 마스킹 → faster-whisper(STT) → MT(번역) → Coqui XTTS v2(보이스 클로닝 TTS) → 길이 맞춤/타임스트레치 → (BGM + FX)와 사이드체인 믹스 → 영상에 오디오 교체**

- **말 전용 트랙**: `speech_only_48k.wav`
- **동물 울음 전용 트랙**: `animal_fx_48k.wav`
- **기타 비-스피치 FX**: `vocals_fx_48k.wav` (환호/웃음 등)
- 최종 믹스는 **BGM + (animal_fx ∪ vocals_fx)** 를 베드로 깔고, TTS에 **사이드체인 컴프레션**을 걸어 합칩니다.

---

## ⚙️ 요구사항
- Docker Desktop (Windows/WSL2, macOS, Linux)
- (옵션) NVIDIA GPU  
  - Windows/WSL2: WSL GPU 지원 + Docker Desktop GPU 패스스루  
  - Linux: NVIDIA 드라이버 + NVIDIA Container Toolkit
- 최초 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다.

---

## 🚀 실행
```bash
# 프로젝트 루트에서
docker compose build
docker compose up
````

서버: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔊 Voice Sample (무음 제거 음성 추출)

* Swagger에서 `POST /voice-sample` 로 `.mp4/.wav` 업로드
* 처리: **보컬/배경 분리 → STT 세그먼트로 무음 제거 → 말만 연결한 WAV 반환**
* 생성물: `./data/<job_id>/voice_sample_24k.wav`

예시 (PowerShell):

```powershell
Invoke-WebRequest -Uri http://localhost:8000/voice-sample -Method Post -Form @{
  file = Get-Item .\sample.mp4
} -OutFile voice_sample.wav
```

---

## 🧪 사용 예시 (PowerShell)

### 원샷 더빙

```powershell
# 영어 더빙
Invoke-RestMethod -Uri http://localhost:8000/dub -Method Post -Form @{
  file = Get-Item .\sample.mp4
  target_lang = "en"    # "en" | "ja" | "ko"
}

# 일본어 더빙 + 레퍼런스(원 화자 6초 이상 WAV)
Invoke-RestMethod -Uri http://localhost:8000/dub -Method Post -Form @{
  file = Get-Item .\sample.mp4
  ref_voice = Get-Item .\ref.wav
  target_lang = "ja"
}
```

### 단계형 파이프라인

```powershell
# 1) ASR
$res = Invoke-RestMethod -Uri http://localhost:8000/asr -Method Post -Form @{ file = Get-Item .\sample.mp4 }
$job = $res.job_id

# (선택) 세그먼트 수정 PATCH ...

# 2) 번역
Invoke-RestMethod -Uri "http://localhost:8000/translate/$job" -Method Post -Body (@{
  src = "ko"; tgt = "en"; length_mode = "off"
} | ConvertTo-Json) -ContentType "application/json"

# 3) TTS 길이 프로브 (보고서만)
Invoke-RestMethod -Uri "http://localhost:8000/tts-probe/$job" -Method Post -Form @{ target_lang = "en" }

# 4) TTS 최종합성
Invoke-RestMethod -Uri "http://localhost:8000/tts-finalize/$job" -Method Post -Form @{ target_lang = "en" }

# 5) MUX (오디오 교체)
Invoke-RestMethod -Uri "http://localhost:8000/mux/$job" -Method Post
```

응답 JSON의 `output`은 컨테이너 내부 경로입니다. 로컬에서는 `./data/<job_id>/output.mp4` 로 접근하거나, `GET /download/{job_id}` 로 직접 다운로드하세요.

---

## 📂 작업 폴더 구조 (`./data/<job_id>/`)

* 입력/중간 산출

  * `input.*` (원본)
  * `audio_full_48k.wav` (전체 오디오 48k)
  * `vocals_48k.wav` / `bgm_48k.wav` (Demucs 분리)
  * `speech_only_48k.wav` (**사람 말만**)
  * `animal_fx_48k.wav` (**동물 울음 전용**)
  * `vocals_fx_48k.wav` (**비-스피치 FX**)
  * `speech_16k.wav` (STT 입력)
  * `meta.json` (메타/리포트)
* 최종 산출

  * `dubbed.wav` (TTS 결합)
  * `final_mix.wav` (BGM+FX+TTS 믹스)
  * `output.mp4` (비디오 오디오 교체본)

---

## 🔧 환경변수 (docker-compose.yml)

* 분리/전처리

  * `SEPARATE_BGM="1"`: Demucs 분리 on/off
  * `VAD_AGGR="3"` / `VAD_FRAME_MS="30"`: webrtcvad 파라미터
  * `STT_INTERVAL_MARGIN="0.10"`: STT 세그먼트 양옆 마진(초)
* STT

  * `USE_GPU="1"`: GPU 사용(가능할 때)
  * `WHISPER_MODEL="large-v3"` 등
  * `WHISPER_LANG`: 힌트 언어(예: `ko`)
* 번역(MT)

  * `MT_DEVICE="cuda" | "cpu"`
  * `MT_FAST_ONLY="1"`: 빠른 모델 우선
  * `MT_NUM_BEAMS` / `MT_MAX_NEW_TOKENS` / 배치 관련 등
* TTS

  * `TTS_DEVICE="cuda" | "cpu"`
  * `TTS_MODEL="tts_models/multilingual/multi-dataset/xtts_v2"`

> **동물 울음 검출(PANNs) 튜닝**: `app/animal.py`의
> `th(기본 0.35)`, `min_dur(0.15s)`, `merge_gap(0.20s)`와
> `ANIMAL_LABELS`(AudioSet 라벨 집합)을 조정할 수 있습니다.

---

## 🧠 모델 스택

* **보컬/배경 분리**: Demucs (2-stems: vocals / no_vocals)
* **동물 울음 검출**: PANNs (AudioSet 527 클래스 프레임 단위 태깅)
  → 검출된 구간만 **타임라인 보존 마스킹**으로 `animal_fx_48k.wav` 생성
* **STT**: faster-whisper (GPU/CPU)
* **번역**: m2m100 / NLLB / opus-mt (환경변수로 선택/폴백)
* **TTS**: Coqui **XTTS v2**(멀티링구얼 + 보이스 클로닝)
* **믹스**: FFmpeg `sidechaincompress` + `amix`로 BGM/FX와 TTS 결합

---

## 🎛️ 믹스/레벨 조정 팁

* 동물/기타 FX가 너무 크거나 작다면 `app/utils.py`의
  `mix_bgm_fx_with_tts()` 필터 체인에서 `threshold/ratio` 조절.
* 동물만 별도 레벨로 다루고 싶으면 MUX 전 단계에서
  `animal_fx_48k`와 `vocals_fx_48k`를 분리/가중치 믹스해서 하나의 FX로 합치세요.

---

## ❗️문제 해결

* `animal_fx_48k.wav`가 비거나 과도하면

  * `animal.py`의 임계값 `th` ↑/↓, 최소 길이 `min_dur` 조정
  * `ANIMAL_LABELS`에 해당 종/소리를 추가(예: 고래/돌고래/맹금류 등)
* GPU 관련 에러

  * Windows/WSL2: WSL GPU, Docker Desktop GPU 사용 확인
  * Linux: NVIDIA 드라이버 + Container Toolkit 설치 확인
* Demucs 실패 시

  * 코드가 자동으로 폴백(원본=보컬, BGM=무음)합니다.

---

## 라이선스/크레딧

* 본 저장소 코드는 연구/프로토타입 용도입니다.
* 사용된 모델/체크포인트는 각 프로젝트의 라이선스를 따릅니다.
