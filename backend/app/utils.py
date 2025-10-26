import subprocess
import shlex
import os
from typing import List, Tuple, Optional, Dict
import re
import tempfile

# -------------------- 공용 실행/시간 유틸 --------------------

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

# -------------------- 오디오 가공 기본 --------------------

def make_silence(path: str, seconds: float, ar: int = 24000):
    if seconds <= 0.0001:
        # 0초에 가까우면 빈 파일을 만들어 concat 오류 방지
        open(path, "wb").close()
        return
    run(f"ffmpeg -y -f lavfi -i anullsrc=r={ar}:cl=mono -t {seconds:.6f} -ar {ar} -ac 1 {shlex.quote(path)}")

def time_stretch(in_path: str, out_path: str, tempo: float, ar: int = 24000):
    """
    tempo > 1.0 이면 빨라짐(길이 짧아짐), tempo < 1.0 이면 느려짐(길이 길어짐)
    rubberband가 있으면 품질↑. 없으면 atempo 체인.
    """
    # rubberband 우선
    try:
        run(f'ffmpeg -y -i {shlex.quote(in_path)} -af "rubberband=tempo={tempo:.6f}:formant=1" -ar {ar} -ac 1 {shlex.quote(out_path)}')
        return
    except Exception:
        pass
    # fallback: atempo(0.5~2.0 per filter) - 체인 분할
    def split_tempos(t):
        parts = []
        # atempo는 0.5~2.0 범위. 필요한 배속을 이 범위로 분해
        while t < 0.5 or t > 2.0:
            if t < 0.5:
                parts.append(0.5)
                t /= 0.5
            else:
                parts.append(2.0)
                t /= 2.0
        parts.append(t)
        return parts
    filters = ",".join([f"atempo={t:.6f}" for t in split_tempos(tempo)])
    run(f'ffmpeg -y -i {shlex.quote(in_path)} -af "{filters}" -ar {ar} -ac 1 {shlex.quote(out_path)}')

def concat_audio(files: List[str], out_path: str) -> str:
    """
    files: 24k/mono wav들 경로 순서. copy concat.
    """
    if not files:
        raise RuntimeError("concat_audio: no input files")
    list_path = os.path.join(os.path.dirname(out_path), "_concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in files:
            f.write(f"file '{os.path.basename(p)}'\n")
    run(f"ffmpeg -y -f concat -safe 0 -i {list_path} -c copy {shlex.quote(out_path)}")
    return out_path

def trim_or_pad_to_duration(in_path: str, out_path: str, target_dur: float, ar: int = 24000, fade_out_ms: int = 30):
    """
    입력을 target_dur로 정확히 맞춤.
    - 짧으면 뒤에 무음 붙임
    - 길면 살짝 페이드아웃하며 트림(클릭 방지)
    """
    cur = ffprobe_duration(in_path)
    if cur <= 0.0:
        # 비정상 파일이면 그냥 타겟 길이의 무음으로 대체
        make_silence(out_path, target_dur, ar=ar)
        return {"padded": target_dur, "trimmed": 0.0}
    if cur < target_dur - 1e-3:
        pad = target_dur - cur
        sil = os.path.join(os.path.dirname(out_path), "_pad.wav")
        make_silence(sil, pad, ar=ar)
        tmp = os.path.join(os.path.dirname(out_path), "_to_concat.wav")
        run(f"ffmpeg -y -i {shlex.quote(in_path)} -ar {ar} -ac 1 {shlex.quote(tmp)}")
        out24 = os.path.join(os.path.dirname(out_path), "_out24.wav")
        concat_audio([tmp, sil], out24)
        run(f"ffmpeg -y -i {out24} -ar {ar} -ac 1 {shlex.quote(out_path)}")
        return {"padded": pad, "trimmed": 0.0}
    elif cur > target_dur + 1e-3:
        # 끝 페이드 후 정확 트림
        st = max(0.0, target_dur - fade_out_ms/1000.0)
        run(
            f'ffmpeg -y -i {shlex.quote(in_path)} '
            f'-af "afade=t=out:st={st:.6f}:d={fade_out_ms/1000.0:.6f},atrim=0:{target_dur:.6f},asetpts=PTS-STARTPTS" '
            f'-ar {ar} -ac 1 {shlex.quote(out_path)}'
        )
        return {"padded": 0.0, "trimmed": cur - target_dur}
    else:
        # 거의 동일: 리샘플/모노만 보장
        run(f"ffmpeg -y -i {shlex.quote(in_path)} -ar {ar} -ac 1 {shlex.quote(out_path)}")
        return {"padded": 0.0, "trimmed": 0.0}

def replace_audio_in_video(video_in: str, audio_in: str, video_out: str) -> None:
    # 오디오 PTS 0부터, AAC 48k, faststart
    run(
        f'ffmpeg -y -i {shlex.quote(video_in)} -i {shlex.quote(audio_in)} '
        f'-map 0:v:0 -map 1:a:0 -c:v copy '
        f'-af "asetpts=PTS-STARTPTS" -c:a aac -ar 48000 -b:a 192k '
        f'-shortest -movflags +faststart {shlex.quote(video_out)}'
    )

# -------------------- (기존) 분석/도움 함수들 --------------------

def detect_silences(wav_path: str, noise_db: str = "-35dB", min_s: float = 0.25):
    cmd = (
        f'ffmpeg -hide_banner -nostats -i {shlex.quote(wav_path)} '
        f'-af silencedetect=noise={noise_db}:d={min_s} -f null -'
    )
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    txt = proc.stdout
    silences = []
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
    pts = [0.0]
    for s,e in silences:
        pts += [s,e]
    pts.append(duration)
    pts = [max(0.0, p) for p in sorted(set(pts))]
    out = []
    for i in range(len(pts)-1):
        start, end = pts[i], pts[i+1]
        is_sil = any(abs(start - s) < 1e-3 and abs(end - e) < 1e-3 for s,e in silences)
        out.append((start, end, is_sil))
    return out
