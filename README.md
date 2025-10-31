# AI Dubbing Prototype (Docker)

한국어 영상 → 영어/일본어 더빙 **프로토타입**입니다.  
파이프라인: **FFmpeg 추출 → faster-whisper(STT) → facebook/m2m100_418M 번역 → Coqui XTTS v2(보이스 클로닝 TTS) → 타임스트레치/컨캣 → 영상에 오디오 교체**

## ⚙️ 요구사항
- Docker Desktop (Windows 가능)
- 최초 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다.

## 🚀 실행
```bash
# 프로젝트 루트에서
docker compose build
docker compose up
```

서버: http://localhost:8000/docs

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

## 📦 볼륨/캐시
- `./data` : 입력/출력 작업 폴더 (호스트에 그대로 저장)
- `hf_cache` : Hugging Face 캐시(모델)
- `tts_cache` : Coqui TTS 캐시(모델)

## 📝 참고
- STT: faster-whisper (CPU, int8)
- 번역: Helsinki-NLP/opus-mt-* (ko↔en, en↔ja 등)
- TTS: Coqui **XTTS v2** (멀티링구얼 + 보이스 클로닝)

> 품질을 올리려면: 화자 분리/참고 음성 추출(VAD/diarization), 용어집 기반 후편집, MFA(Forced Alignment) 추가 등을 고려하세요.
