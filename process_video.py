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
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(p.stdout)


def _rate(value: str, fallback: float = 24.0) -> float:
    try:
        if "/" in value:
            n, d = value.split("/", 1)
            d = float(d)
            return float(n) / d if d else fallback
        return float(value)
    except Exception:
        return fallback


def probe(path: Path) -> VideoInfo:
    data = ffprobe_json(path)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v:
        raise ValueError("No video stream found.")

    fps = _rate(v.get("avg_frame_rate", "0/1"), 0)
    if fps <= 0:
        fps = _rate(v.get("r_frame_rate", "24/1"), 24)

    duration = float(v.get("duration") or data.get("format", {}).get("duration") or 0)
    frames = int(v.get("nb_frames") or 0)
    if frames == 0 and duration and fps:
        frames = round(duration * fps)

    return VideoInfo(
        int(v["width"]),
        int(v["height"]),
        fps,
        frames,
        duration,
        any(s.get("codec_type") == "audio" for s in streams),
    )


def _scaled_box(info: VideoInfo):
    # Golden V3 watermark geometry at the tested 720x1280 reference.
    base_w, base_h = 720, 1280
    x1, x2, y1, y2 = 616, 708, 1234, 1268
    sx, sy = info.width / base_w, info.height / base_h
    return (
        round(x1 * sx),
        round(x2 * sx),
        round(y1 * sy),
        round(y2 * sy),
    )


def remove_grok_watermark(input_path: Path, output_path: Path, mask_path: Path | None = None, max_seconds: float = 30.0):
    """V4 fast engine: identical Golden V3 geometry/inpainting, but only on a small ROI.

    The previous V3 public implementation inpainted a full-frame mask for every frame and
    then encoded that intermediate video again with FFmpeg. V4 performs the exact same
    OpenCV Telea/NS-style operation only on the watermark ROI and streams raw cleaned
    frames directly into FFmpeg's H.264 encoder, eliminating the extra intermediate encode.
    """
    info = probe(input_path)
    if info.duration > max_seconds + 0.25:
        raise ValueError(f"Video is longer than the {int(max_seconds)} second limit.")
    if info.width < 200 or info.height < 200:
        raise ValueError("Video resolution is too small.")

    # Preserve the exact V3 rectangle. mask_path is retained for API compatibility, but
    # the tested V3 rectangle itself is the authoritative mask geometry.
    x1, x2, y1, y2 = _scaled_box(info)

    # A small context margin gives the inpaint algorithm enough surrounding pixels while
    # keeping the processed area tiny. Clamp to frame bounds.
    margin = max(24, round(32 * info.width / 720))
    rx1 = max(0, x1 - margin)
    rx2 = min(info.width, x2 + margin + 1)
    ry1 = max(0, y1 - margin)
    ry2 = min(info.height, y2 + margin + 1)

    mask = np.zeros((ry2 - ry1, rx2 - rx1), dtype=np.uint8)
    mx1, mx2 = x1 - rx1, x2 - rx1
    my1, my2 = y1 - ry1, y2 - ry1
    cv2.rectangle(mask, (mx1, my1), (mx2, my2), 255, -1)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    # Stream raw BGR frames directly into FFmpeg. This avoids the old mp4v intermediate
    # encode/decode and lets FFmpeg produce the final H.264 file in one video encode.
    encoder = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s:v", f"{info.width}x{info.height}",
        "-r", f"{info.fps:.12g}",
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    proc = subprocess.Popen(encoder, stdin=subprocess.PIPE)

    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            roi = frame[ry1:ry2, rx1:rx2]
            cleaned_roi = cv2.inpaint(roi, mask, 3, cv2.INPAINT_NS)
            frame[ry1:ry2, rx1:rx2] = cleaned_roi

            proc.stdin.write(frame.tobytes())
            count += 1
    except BrokenPipeError as exc:
        raise RuntimeError("FFmpeg video encoder stopped unexpectedly.") from exc
    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"FFmpeg video encoding failed with exit code {rc}.")
    if count == 0:
        raise ValueError("No video frames could be decoded.")

    # Mux/copy the original audio stream after the single H.264 encode.
    audio_mux = output_path.parent / "muxed.mp4"
    mux_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(output_path),
        "-i", str(input_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(audio_mux),
    ]
    subprocess.run(mux_cmd, check=True)
    audio_mux.replace(output_path)

    out_info = probe(output_path)
    if out_info.width != info.width or out_info.height != info.height:
        raise RuntimeError("Output resolution changed unexpectedly.")
    if abs(out_info.fps - info.fps) > 0.01:
        raise RuntimeError("Output FPS changed unexpectedly.")
    return info
