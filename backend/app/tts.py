# app/tts.py
import os
from functools import lru_cache

LANG_MAP = {"en": "en", "ja": "ja", "ko": "ko"}
TTS_DEVICE = os.getenv("TTS_DEVICE", "cpu").lower()

def _prepare_torch_deserialization():
    """
    PyTorch 안전 직렬화 허용목록에 XTTS 관련 클래스를 추가.
    (필요 객체가 더 나오면 여기서 계속 추가 가능)
    """
    try:
        import torch.serialization as ts
        allow = []
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            allow.append(XttsConfig)
        except Exception:
            pass
        try:
            from TTS.tts.models.xtts import XttsAudioConfig
            allow.append(XttsAudioConfig)
        except Exception:
            pass
        if allow:
            ts.add_safe_globals(allow)
    except Exception:
        pass

@lru_cache(maxsize=1)
def _get_tts(model_name: str):
    _prepare_torch_deserialization()

    # 1차 시도: 안전 직렬화(allowlist 기반)만으로 로드
    from TTS.api import TTS
    import torch
    try:
        tts = TTS(model_name)
    except Exception as e:
        # 2차 폴백: 신뢰하는 체크포인트에 한해 weights_only=False로 재시도
        if "Weights only load failed" in str(e):
            orig_load = torch.load
            def _patched_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)  # 안전장치 완화 (공식 XTTS만 쓸 때)
                return orig_load(*args, **kwargs)
            try:
                torch.load = _patched_load
                tts = TTS(model_name)
            finally:
                torch.load = orig_load
        else:
            raise

    # 디바이스 이동
    use_cuda = (TTS_DEVICE == "cuda") and torch.cuda.is_available()
    try:
        tts.to("cuda" if use_cuda else "cpu")
    except Exception:
        pass
    return tts

# app/tts.py
def synthesize(text: str, ref_wav: str, language: str, out_path: str, model_name: str):
    if model_name.lower().startswith("openvoice_v2"):
        import torch
        from .utils import run  # 24k/mono 보장용 리샘플
        ckpt_root = os.getenv("OPENVOICE_V2_DIR", "/app/assets/openvoice_v2/checkpoints_v2")
        dev = os.getenv("TTS_DEVICE", "cpu").lower()
        speaker = os.getenv("OPENVOICE_SPEAKER", "default")
        speed = float(os.getenv("OPENVOICE_SPEED", "1.0"))

        ov = _OpenVoiceV2(ckpt_root, device_str=dev)
        tmp = out_path.replace(".wav", "_ov_final.wav")
        ov.synth(text, ref_wav, language, tmp, speaker=speaker, speed=speed)

        # 파이프라인이 24k/mono 가정 → 일치화
        run(f'ffmpeg -y -i "{tmp}" -ar 24000 -ac 1 "{out_path}"')
        return

    # --- 기존 XTTS(Coqui) 경로 (그대로 유지) ---
    tts = _get_tts(model_name)
    lang = LANG_MAP.get(language, "en")
    tts.tts_to_file(text=text, speaker_wav=ref_wav, language=lang, file_path=out_path)

def _ov_lang_label(lang: str) -> tuple[str, str]:
    """
    (OpenVoice base 스피커 폴더, base_speaker_tts의 language 문자열) 반환
    """
    l = {"en": ("EN", "English"),
         "ja": ("JP", "Japanese"),
         "ko": ("KO", "Korean")}.get(lang, ("EN", "English"))
    return l

class _OpenVoiceV2:
    """
    OpenVoice V2 파이프라인:
      1) BaseSpeakerTTS로 문장 합성 (스타일/속도 제어)
      2) ToneColorConverter로 참조 화자의 톤컬러로 변환
    """
    def __init__(self, ckpt_root: str, device_str: str = "cpu"):
        import torch
        from openvoice.api import BaseSpeakerTTS, ToneColorConverter

        self.torch = torch
        self.device = "cuda" if (device_str == "cuda" and torch.cuda.is_available()) else "cpu"

        # 체크포인트 경로
        self.ckpt_root = ckpt_root
        self.ckpt_converter = os.path.join(ckpt_root, "converter")

        self.ToneColorConverter = ToneColorConverter
        self.BaseSpeakerTTS = BaseSpeakerTTS

        # 변환기 로드(한 번)
        self.converter = ToneColorConverter(
            os.path.join(self.ckpt_converter, "config.json"),
            device=self.device
        )
        self.converter.load_ckpt(os.path.join(self.ckpt_converter, "checkpoint.pth"))

    def _load_base(self, lang_code: str):
        base_dir_name, lang_label = _ov_lang_label(lang_code)
        ckpt_base = os.path.join(self.ckpt_root, "base_speakers", base_dir_name)

        base = self.BaseSpeakerTTS(os.path.join(ckpt_base, "config.json"), device=self.device)
        base.load_ckpt(os.path.join(ckpt_base, "checkpoint.pth"))

        # 스타일 SE(언어별 기본 SE 제공됨)
        se_name_map = {"EN": "en_default_se.pth", "JP": "jp_default_se.pth", "KO": "ko_default_se.pth"}
        se_name = se_name_map.get(base_dir_name, "en_default_se.pth")
        source_se = self.torch.load(os.path.join(ckpt_base, se_name)).to(self.device)

        return base, lang_label, source_se

    def synth(self, text: str, ref_wav: str, lang_code: str, out_path: str, speaker: str = "default", speed: float = 1.0):
        from openvoice import se_extractor
        import os

        base, lang_label, source_se = self._load_base(lang_code)

        # 1) 베이스 합성 (24k로 내보낼 임시 파일)
        tmp_src = out_path.replace(".wav", "_ov_base.wav")
        base.tts(text, tmp_src, speaker=speaker, language=lang_label, speed=float(speed))

        # 2) 참조 화자 임베딩 추출 (작업 디렉토리별 캐시)
        proc_dir = os.path.join(os.path.dirname(out_path), "ov_proc")
        os.makedirs(proc_dir, exist_ok=True)
        target_se, _ = se_extractor.get_se(ref_wav, self.converter, target_dir=proc_dir, vad=True)

        # 3) 톤컬러 변환 → 최종 파일
        self.converter.convert(
            audio_src_path=tmp_src,
            src_se=source_se,
            tgt_se=target_se,
            output_path=out_path,
            message="@MyShell"  # 워터마크 메시지(옵션)
        )