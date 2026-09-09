import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    duration: float
    has_audio: bool


def ffprobe_json(path: Path):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, text=True
    )
    return json.loads(p.stdout)


def probe(path: Path) -> VideoInfo:
    data = ffprobe_json(path)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v:
        raise ValueError("No video stream found.")
    fps = float(v.get("avg_frame_rate", "0/1").split('/')[0]) / max(1.0, float(v.get("avg_frame_rate", "0/1").split('/')[1])) if '/' in v.get("avg_frame_rate", "0/1") else 0
    if fps <= 0:
        fps = float(v.get("r_frame_rate", "24/1").split('/')[0]) / max(1.0, float(v.get("r_frame_rate", "24/1").split('/')[1]))
    duration = float(v.get("duration") or data.get("format", {}).get("duration") or 0)
    frames = int(v.get("nb_frames") or 0)
    if frames == 0 and duration and fps:
        frames = round(duration * fps)
    return VideoInfo(int(v["width"]), int(v["height"]), fps, frames, duration, any(s.get("codec_type") == "audio" for s in streams))


def remove_grok_watermark(input_path: Path, output_path: Path, mask_path: Path, max_seconds: float = 30.0):
    info = probe(input_path)
    if info.duration > max_seconds + 0.25:
        raise ValueError(f"Video is longer than the {int(max_seconds)} second limit.")
    if info.width < 200 or info.height < 200:
        raise ValueError("Video resolution is too small.")
    # Golden Grok mask was built for 720x1280 and is scaled to preserve relative placement.
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError("Golden watermark mask could not be loaded.")
    if mask.shape != (info.height, info.width):
        mask = cv2.resize(mask, (info.width, info.height), interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    # Restrict to the bottom-right region where the known Grok watermark is located.
    # The original 720x1280 mask bbox is x=628..712, y=1237..1265.
    # Preserve this geometry proportionally for other 9:16 resolutions.
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    tmp_dir = output_path.parent / "frames"
    tmp_dir.mkdir(exist_ok=True)
    clean_video = output_path.parent / "silent-clean.mp4"
    writer = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(clean_video), writer, info.fps, (info.width, info.height))
    if not vw.isOpened():
        cap.release()
        raise RuntimeError("Could not initialize video encoder.")

    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cleaned = cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)
        vw.write(cleaned)
        count += 1
    cap.release()
    vw.release()

    if count == 0:
        raise ValueError("No video frames could be decoded.")

    # Re-encode video with H.264 and mux/copy original audio when available.
    audio_args = ["-map", "1:a:0?", "-c:a", "copy"]
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(clean_video),
        "-i", str(input_path),
        "-map", "0:v:0", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-r", str(info.fps),
        *audio_args,
        "-shortest", str(output_path)
    ]
    subprocess.run(cmd, check=True)
    clean_video.unlink(missing_ok=True)

    out_info = probe(output_path)
    if out_info.width != info.width or out_info.height != info.height:
        raise RuntimeError("Output resolution changed unexpectedly.")
    if abs(out_info.fps - info.fps) > 0.01:
        raise RuntimeError("Output FPS changed unexpectedly.")
    return info
