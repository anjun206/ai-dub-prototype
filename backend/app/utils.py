import subprocess
import shlex
import os
from typing import List, Tuple, Optional
import re, tempfile, math

def run(cmd: str, cwd: Optional[str]=None) -> None:
    print(f"[RUN] {cmd}")
    proc = subprocess.run(shlex.split(cmd), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{proc.stdout}")

def ffprobe_duration(path: str) -> float:
    cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {shlex.quote(path)}"
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    try:
        return float(proc.stdout.strip())
    except:
        return 0.0

def time_stretch(in_path: str, out_path: str, ratio: float) -> None:
    # ffmpeg atempo supports 0.5~2.0 per filter; chain if needed
    def split_ratios(r):
        parts = []
        while r < 0.5 or r > 2.0:
            if r < 0.5:
                parts.append(0.5)
                r /= 0.5
            else:
                parts.append(2.0)
                r /= 2.0
        parts.append(r)
        return parts
    filters = ",".join([f"atempo={r:.6f}" for r in split_ratios(ratio)])
    run(f'ffmpeg -y -i {shlex.quote(in_path)} -filter:a "{filters}" -ar 24000 -ac 1 {shlex.quote(out_path)}')

def concat_audio(files: List[Tuple[str, float]]) -> str:
    # files: list of (audio_path, gap_after_seconds)
    # create a temp concat list
    concat_list = []
    tmp_parts = []
    for i, (audio, gap) in enumerate(files):
        tmp_parts.append(audio)
        if gap > 0.0001:
            gap_path = f"{os.path.splitext(audio)[0]}_gap.wav"
            run(f"ffmpeg -y -f lavfi -i anullsrc=r=24000:cl=mono -t {gap:.3f} {gap_path}")
            tmp_parts.append(gap_path)
    list_path = os.path.join(os.path.dirname(files[0][0]), "concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in tmp_parts:
            f.write(f"file '{os.path.basename(p)}'\n")
    out_path = os.path.join(os.path.dirname(files[0][0]), "dubbed.wav")
    run(f"ffmpeg -y -f concat -safe 0 -i {list_path} -c copy {out_path}")
    return out_path

def replace_audio_in_video(video_in: str, audio_in: str, video_out: str) -> None:
    run(f"ffmpeg -y -i {shlex.quote(video_in)} -i {shlex.quote(audio_in)} -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -b:a 192k -shortest {shlex.quote(video_out)}")

def detect_silences(wav_path: str, noise_db: str = "-35dB", min_s: float = 0.25):
    """
    ffmpeg silencedetect로 (start, end) 무음 구간 리스트 반환
    """
    cmd = (
        f'ffmpeg -hide_banner -nostats -i {shlex.quote(wav_path)} '
        f'-af silencedetect=noise={noise_db}:d={min_s} -f null -'
    )
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    txt = proc.stdout
    sil_starts, sil_ends = [], []
    for line in txt.splitlines():
        m1 = re.search(r"silence_start:\s*([0-9.]+)", line)
        m2 = re.search(r"silence_end:\s*([0-9.]+)", line)
        if m1: sil_starts.append(float(m1.group(1)))
        if m2: sil_ends.append(float(m2.group(1)))
    # 쌍 맞추기
    silences = []
    i = 0
    last = None
    for line in txt.splitlines():
        m1 = re.search(r"silence_start:\s*([0-9.]+)", line)
        m2 = re.search(r"silence_end:\s*([0-9.]+)", line)
        if m1:
            last = float(m1.group(1))
        if m2 and last is not None:
            silences.append((last, float(m2.group(1))))
            last = None
    return silences

def segment_boundaries(duration: float, silences):
    """
    전체 길이와 무음구간을 받아서 [발화][무음][발화]... 식의 경계 리스트 생성
    반환: [(seg_start, seg_end, is_silence: bool), ...]
    """
    pts = [0.0]
    for s,e in silences:
        pts += [s,e]
    pts.append(duration)
    pts = [max(0.0, p) for p in sorted(set(pts))]
    out = []
    for i in range(len(pts)-1):
        start, end = pts[i], pts[i+1]
        is_sil = False
        for s,e in silences:
            if abs(start - s) < 1e-3 and abs(end - e) < 1e-3:
                is_sil = True
                break
        out.append((start, end, is_sil))
    return out

def extract_wav_segment(src: str, dst: str, start: float, end: float):
    run(f'ffmpeg -y -ss {start:.3f} -to {end:.3f} -i {shlex.quote(src)} -ac 1 -ar 24000 -c:a pcm_s16le {shlex.quote(dst)}')

def safe_stretch(in_path: str, out_path: str, ratio: float, prefer_rb: bool = True):
    # 과도한 비율을 피하려고 clamp
    r = max(0.7, min(1.3, ratio))
    if prefer_rb:
        # ffmpeg rubberband가 빌드에 켜져 있음(로그상). 품질↑
        run(f'ffmpeg -y -i {shlex.quote(in_path)} -af "rubberband=tempo={r:.6f}:formant=1" -ar 24000 -ac 1 {shlex.quote(out_path)}')
    else:
        time_stretch(in_path, out_path, r)  # 기존 atempo 체인

def piecewise_fit(tts_wav: str, ref_wav: str, out_wav: str):
    """
    tts_wav를 ref_wav의 시간 구조(발화/무음 패턴)에 맞춰 '조각별'로 리타이밍.
    """
    ref_dur = ffprobe_duration(ref_wav)
    tts_dur = ffprobe_duration(tts_wav)
    ref_sil = detect_silences(ref_wav)
    tts_sil = detect_silences(tts_wav)

    ref_parts = segment_boundaries(ref_dur, ref_sil)
    tts_parts = segment_boundaries(tts_dur, tts_sil)

    # 파트 수가 다르면 근사 매칭: 개수를 맞추기 위해 더 세밀한 쪽을 병합
    def normalize_parts(parts, target_n):
        if len(parts) == target_n:
            return parts
        out = []
        acc = 0.0; start = parts[0][0]; sil = parts[0][2]
        chunks = []
        for i,p in enumerate(parts):
            chunks.append(p)
        # 단순 리스케일: 인덱스 매핑
        mapped = []
        for i in range(target_n):
            a = int(round(i * len(chunks) / target_n))
            b = int(round((i+1) * len(chunks) / target_n))
            a = min(max(a,0), len(chunks)-1)
            b = min(max(b, a+1), len(chunks))
            segs = chunks[a:b]
            s = segs[0][0]
            e = segs[-1][1]
            is_sil = all(x[2] for x in segs) if any(x[2] for x in segs) else False
            mapped.append((s,e,is_sil))
        return mapped

    n = min(len(ref_parts), len(tts_parts))
    ref_parts = normalize_parts(ref_parts, n)
    tts_parts = normalize_parts(tts_parts, n)

    tmp_dir = tempfile.mkdtemp()
    out_list = []
    for i, ((rs,re,rsil),(ts,te,tsil)) in enumerate(zip(ref_parts, tts_parts)):
        rdur = re - rs
        tdur = te - ts
        if tdur <= 0.01 or rdur <= 0.01:
            continue
        seg_in = os.path.join(tmp_dir, f"in_{i:04d}.wav")
        seg_out = os.path.join(tmp_dir, f"out_{i:04d}.wav")
        extract_wav_segment(tts_wav, seg_in, ts, te)
        ratio = rdur/tdur
        # 무음 조각은 정확 매칭 우선(큰 비율 허용), 발화 조각은 안전비율로 clamp
        safe_stretch(seg_in, seg_out, ratio, prefer_rb=True if not tsil else True)
        out_list.append(seg_out)

    # concat
    concat_file = os.path.join(tmp_dir, "c.txt")
    with open(concat_file, "w", encoding="utf-8") as f:
        for p in out_list:
            f.write(f"file '{os.path.basename(p)}'\n")
    run(f'ffmpeg -y -f concat -safe 0 -i {concat_file} -c copy {shlex.quote(out_wav)}')