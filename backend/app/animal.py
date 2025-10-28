# app/animal.py
import os
import numpy as np
import torch
import torchaudio
from panns_inference import SoundEventDetection, labels as PANN_LABELS

ANIMAL_LABELS = {
    "Dog", "Bark", "Howl", "Growling", "Cat", "Meow", "Horse",
    "Cattle", "Moo", "Pig", "Sheep", "Goat", "Roaring cats (lions, tigers)",
    "Roar", "Bird", "Bird vocalization, bird call, bird song",
    "Insect", "Cricket", "Frog", "Crow", "Owl", "Marine mammal",
    "Whale vocalization", "Dolphin", "Monkey", "Chimpanzee"
}

def _ma_smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(xp, ker, mode="valid").astype(np.float32)

def _load_mono_resample(path: str, target_sr: int = 32000) -> np.ndarray:
    wav, sr = torchaudio.load(path)  # [C, T]
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        res = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = res(wav)
    return wav.squeeze(0).cpu().numpy().astype(np.float32)

def detect_animal_intervals(
    wav_path: str,
    sr: int = 32000,
    th: float = float(os.getenv("ANIMAL_TH", "0.35")),
    min_dur: float = float(os.getenv("ANIMAL_MIN_DUR", "0.15")),
    merge_gap: float = float(os.getenv("ANIMAL_MERGE_GAP", "0.20")),
):
    """
    반환: [(start, end), ...] (초)
    """
    y = _load_mono_resample(wav_path, target_sr=sr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sed = SoundEventDetection(device=device)

    # framewise_output: (T, 527), 프레임 hop ≈ 10ms @32k
    framewise_output = sed.inference(y)[0]

    idxs = [i for i, name in enumerate(PANN_LABELS) if name in ANIMAL_LABELS]
    if not idxs:
        return []

    prob = framewise_output[:, idxs].max(axis=1).astype(np.float32)
    prob = _ma_smooth(prob, k=5)

    hop = 0.01  # ≈10ms
    intervals = []
    on = None
    for t, p in enumerate(prob):
        if p >= th and on is None:
            on = t * hop
        elif p < th and on is not None:
            off = t * hop
            if off - on >= min_dur:
                intervals.append((on, off))
            on = None
    if on is not None:
        off = len(prob) * hop
        if off - on >= min_dur:
            intervals.append((on, off))

    # 가까운 구간 병합
    merged = []
    for s, e in sorted(intervals):
        if not merged or s - merged[-1][1] > merge_gap:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(float(s), float(e)) for s, e in merged]
