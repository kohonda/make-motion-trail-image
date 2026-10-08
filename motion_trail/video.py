"""Encode composites to a video file."""

from __future__ import annotations

import shutil
import subprocess
from itertools import chain
from pathlib import Path

import cv2
import numpy as np

VIDEO_OUT_EXTS = {".mp4", ".mov", ".mkv", ".avi"}


def write_video(frames_bgr, path: Path, fps: float = 10.0) -> str:
    """Encode BGR frames to *path*; return the codec written ("h264" / "mpeg4").

    ffmpeg (browser-playable H.264) is used when on PATH, else OpenCV's mpeg4.
    *frames_bgr* may be a one-shot generator, so it is streamed, not replayed.
    """
    frames = iter(frames_bgr)
    first = next(frames, None)
    if first is None:
        raise ValueError("no frames to write")
    frames = chain([first], frames)
    h, w = first.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(float(fps), 0.1)

    if shutil.which("ffmpeg"):
        _write_video_ffmpeg(frames, path, fps, w, h)
        return "h264"
    _write_video_opencv(frames, path, fps, w, h)
    return "mpeg4"


def _write_video_ffmpeg(frames, path: Path, fps: float, w: int, h: int) -> None:
    """Pipe raw BGR frames into ffmpeg and encode them as H.264."""
    # fmt: off
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}",
        "-i", "-",
        # yuv420p needs even dimensions, which an odd-sized frame would break
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
    ]
    # fmt: on
    if path.suffix.lower() in {".mp4", ".mov"}:
        cmd += ["-movflags", "+faststart"]  # only the mov muxer knows this one
    proc = subprocess.Popen(
        cmd + [str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass  # ffmpeg died early; its stderr below says why
    err = proc.stderr.read().decode("utf-8", "replace").strip()
    proc.stderr.close()
    if proc.wait() != 0:
        raise RuntimeError(err or "ffmpeg could not encode the video")


def _write_video_opencv(frames, path: Path, fps: float, w: int, h: int) -> None:
    """Fallback mpeg4 encoder (not browser-playable) for machines without ffmpeg."""
    # even size: this writer silently drops the last row / column of an odd one
    w, h = max(w - w % 2, 2), max(h - h % 2, 2)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open {path} for writing")
    try:
        for frame in frames:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
