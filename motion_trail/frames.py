"""Load frames from an image folder or a video."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def load_images(folder: Path) -> Tuple[List[np.ndarray], List[Path]]:
    """Load every .png / .jpg / .jpeg in *folder*, sorted by name."""
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    frames = [cv2.imread(str(p)) for p in paths]
    if frames:
        target_h, target_w = frames[0].shape[:2]
        frames = [
            cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA)
            if f.shape[:2] != (target_h, target_w)
            else f
            for f in frames
        ]
    return frames, paths


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def resize_to_first(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Resize every frame to match the first one's (H, W)."""
    if not frames:
        return frames
    h, w = frames[0].shape[:2]
    return [
        cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        if f.shape[:2] != (h, w)
        else f
        for f in frames
    ]


def load_video(
    path: Path,
    start_sec: float = 0.0,
    end_sec: float = 0.0,
    interval_sec: float = 1.0,
) -> List[np.ndarray]:
    """Extract one BGR frame every *interval_sec* over [start_sec, end_sec].

    ``end_sec <= 0`` means until the end. Returns [] if the video can't be read.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []

    interval_sec = max(float(interval_sec), 1e-3)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total > 0:
        indices = _interval_indices(start_sec, end_sec, interval_sec, fps, total)
        frames = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        cap.release()
        return resize_to_first(frames)

    # Frame count unknown (some codecs): read sequentially, then sample.
    all_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        all_frames.append(frame)
    cap.release()
    if not all_frames:
        return []
    indices = _interval_indices(start_sec, end_sec, interval_sec, fps, len(all_frames))
    return resize_to_first([all_frames[i] for i in indices])


def _interval_indices(
    start_sec: float,
    end_sec: float,
    interval_sec: float,
    fps: float,
    total: int,
) -> List[int]:
    """Frame indices at *interval_sec* steps across [start_sec, end_sec]."""
    duration = total / fps
    end_time = end_sec if (end_sec and end_sec > 0) else duration
    end_time = min(end_time, duration)
    start_time = max(min(start_sec, end_time), 0.0)

    indices: List[int] = []
    t = start_time
    while t <= end_time + 1e-9:
        fi = min(max(int(round(t * fps)), 0), total - 1)
        if not indices or fi != indices[-1]:
            indices.append(fi)
        t += interval_sec
    if not indices:
        indices = [min(max(int(round(start_time * fps)), 0), total - 1)]
    return indices
