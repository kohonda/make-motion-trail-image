"""Callbacks for loading frames, managing sets and annotating frames."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path

import cv2
import gradio as gr

from motion_trail.frames import VIDEO_EXTS, load_video, resize_to_first
from motion_trail.sam import run_predictor_on_frame
from motion_trail.ui.state import (
    current_views,
    extract_updates,
    label_to_index,
    new_set,
    next_color,
    parse_color,
    picker_hex,
    rgb_to_hex,
    selector_update,
    show_set,
)


def _parse_time(value) -> float:
    """Parse ``"12.5"``, ``"m:s"`` or ``"h:m:s"`` to seconds (blank / invalid -> 0)."""
    if value is None:
        return 0.0
    s = str(value).strip()
    if not s:
        return 0.0
    try:
        parts = [float(p) for p in s.split(":")]
    except ValueError:
        gr.Warning(f"Invalid time: {value}")
        return 0.0
    total = 0.0
    for p in parts:
        total = total * 60 + p
    return max(total, 0.0)


_BROWSER_CODECS = {"h264", "avc1", "vp8", "vp9", "av1"}


def playable_video(path: str):
    """A browser-playable path for *path*, transcoding to H.264 in the temp dir if needed."""
    src = Path(path)
    if not src.is_file():
        return gr.update()

    codec = ""
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(src),
            ],
            capture_output=True,
            text=True,
        )
        codec = probe.stdout.strip()
    except OSError:
        pass

    if codec in _BROWSER_CODECS and src.suffix.lower() in {".mp4", ".webm"}:
        return str(src)

    out_dir = Path(tempfile.gettempdir()) / "motion_trail_preview"
    out_dir.mkdir(exist_ok=True)
    # keyed on path + mtime + size so same-named videos never collide
    stat = src.stat()
    key = hashlib.md5(
        f"{src.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode()
    ).hexdigest()
    out = out_dir / f"{key}.mp4"
    if out.is_file():
        return str(out)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(out),
            ],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        gr.Warning("Could not prepare the video for in-browser playback")
        return gr.update()
    return str(out)


def add_set(sets: list):
    """Append a new empty set, select it, and clear the workspace."""
    color = next_color(len(sets))
    sets = sets + [new_set(color)]
    active = len(sets) - 1
    return (
        sets,  # st_sets
        active,  # st_active
        0,  # st_idx
        selector_update(sets, active),  # set_selector
        None,  # input_image
        None,  # preview_image
        gr.update(maximum=0, value=0),  # frame_slider
        rgb_to_hex(color),  # color_picker
        False,  # no_color_checkbox
    )


def remove_set(sets: list, active: int):
    """Remove the active set; keep at least one set."""
    sets = list(sets)
    if 0 <= active < len(sets):
        sets.pop(active)
    if not sets:
        sets = [new_set(next_color(0))]
        active = 0
    else:
        active = min(active, len(sets) - 1)

    s = sets[active]
    return (
        sets,  # st_sets
        active,  # st_active
        0,  # st_idx
        selector_update(sets, active),  # set_selector
        *show_set(s),  # input_image, preview_image, frame_slider
        picker_hex(s["color"], active),  # color_picker
        s["color"] is None,  # no_color_checkbox
        *extract_updates(s),  # start_sec, end_sec, interval_sec
    )


def move_set(sets: list, active: int, delta: int):
    """Swap the active set with its neighbour; the last set is drawn on top."""
    sets = list(sets)
    target = active + delta
    if not (0 <= active < len(sets)) or not (0 <= target < len(sets)):
        where = "last" if delta > 0 else "first"
        gr.Warning(f"The active set is already {where} in the order")
        return sets, active, gr.update(), gr.update()

    sets[active], sets[target] = sets[target], sets[active]
    active = target
    return (
        sets,  # st_sets
        active,  # st_active
        selector_update(sets, active),  # set_selector
        picker_hex(sets[active]["color"], active),  # color_picker
    )


def select_set(sets: list, label):
    """Switch the active set and repaint the workspace from its state."""
    active = label_to_index(label, sets)
    s = sets[active]
    return (
        active,  # st_active
        0,  # st_idx
        *show_set(s),  # input_image, preview_image, frame_slider
        picker_hex(s["color"], active),  # color_picker
        s["color"] is None,  # no_color_checkbox
        *extract_updates(s),  # start_sec, end_sec, interval_sec
    )


def set_color(sets: list, active: int, value):
    """Store a user-picked colour on the active set (clears 'no colour')."""
    rgb = parse_color(value)
    if rgb is not None and 0 <= active < len(sets):
        sets[active]["color"] = rgb
    return sets, False  # picking a colour implies the set is coloured


def toggle_no_color(sets: list, active: int, no_color: bool, picker_value):
    """Toggle tinting for the active set; 'no colour' keeps original pixels."""
    if 0 <= active < len(sets):
        if no_color:
            sets[active]["color"] = None
        else:
            sets[active]["color"] = parse_color(picker_value) or next_color(active)
    return sets


def set_background(sets: list, active: int, idx: int):
    """Capture the active set's current frame as the composite background."""
    if not (0 <= active < len(sets)):
        return None, None
    s = sets[active]
    if not s["frames_bgr"]:
        gr.Warning("Load images for this set first")
        return None, None
    idx = int(idx)
    if idx >= len(s["frames_bgr"]):
        idx = 0
    bg_bgr = s["frames_bgr"][idx].copy()
    bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
    return bg_bgr, bg_rgb


def _ingest_frames(
    frames_bgr: list,
    sets: list,
    active: int,
    source_label: str,
    video_out,
    extract: dict | None = None,
):
    """Store *frames_bgr* into the active set and return the standard outputs."""
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]

    if not sets:
        sets = [new_set(next_color(0))]
        active = 0
    s = sets[active]
    s["dir"] = source_label
    s["frames"] = frames_rgb
    s["frames_bgr"] = frames_bgr
    s["points_map"] = {}
    s["masks"] = [None] * len(frames_rgb)
    s["extract"] = extract

    first = frames_rgb[0]
    return (
        first,  # input_image
        first,  # preview_image
        gr.update(maximum=max(len(frames_rgb) - 1, 0), value=0),  # frame_slider
        0,  # st_idx
        sets,  # st_sets
        video_out,  # video_player (path for a movie, None for an image folder)
    )


def load_video_frames(
    video_path: str,
    sets: list,
    active: int,
    start_sec: str,
    end_sec: str,
    interval_sec: float,
):
    """Extract frames from the dropped video into the active set."""
    if not video_path:
        gr.Warning("Drop a video onto the player first")
        return None, None, gr.update(), 0, sets, gr.update()
    p = Path(video_path)
    if not (p.is_file() and p.suffix.lower() in VIDEO_EXTS):
        gr.Warning(f"Not a supported video file: {video_path}")
        return None, None, gr.update(), 0, sets, gr.update()
    frames_bgr = load_video(
        p,
        _parse_time(start_sec),
        _parse_time(end_sec),
        float(interval_sec or 1.0),
    )
    if not frames_bgr:
        gr.Warning("Could not read frames from video")
        return None, None, gr.update(), 0, sets, gr.update()
    return _ingest_frames(
        frames_bgr,
        sets,
        active,
        str(p),
        gr.update(),  # the player already shows the dropped video
        {
            "start_sec": start_sec,
            "end_sec": end_sec,
            "interval_sec": float(interval_sec or 1.0),
        },
    )


def load_image_files(files: list, sets: list, active: int):
    """Load dropped image files (e.g. a folder) into the active set."""
    if not files:
        return None, None, gr.update(), 0, sets, gr.update()
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(
        (Path(f) for f in files if Path(f).suffix.lower() in exts),
        key=lambda q: q.name,
    )
    frames = [cv2.imread(str(q)) for q in paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        gr.Warning("No images (.png/.jpg/.jpeg) found in the dropped folder")
        return None, None, gr.update(), 0, sets, gr.update()
    return _ingest_frames(
        resize_to_first(frames), sets, active, "(dropped folder)", None
    )


def on_video_drop(video_path):
    """Keep the original path for extraction and show a browser-playable copy."""
    if not video_path:
        return None, None
    if Path(video_path).suffix.lower() not in VIDEO_EXTS:
        gr.Warning(f"Not a supported video file: {video_path}")
        return None, None
    return video_path, playable_video(video_path)


def on_image_click(
    sets: list,
    active: int,
    current_idx: int,
    evt: gr.SelectData,
    mode: str,
):
    """Add a point to the active set's current frame and re-run SAM 3."""
    if not (0 <= active < len(sets)):
        return None, None, sets
    s = sets[active]
    if not s["frames"]:
        return None, None, sets

    label = 1 if mode == "Positive" else 0
    x, y = evt.index

    pts = s["points_map"].setdefault(current_idx, [])
    pts.append((x, y, label))

    rgb = s["frames"][current_idx]
    mask = run_predictor_on_frame(rgb, pts)
    s["masks"][current_idx] = mask
    return *current_views(s["frames"], s["points_map"], current_idx, s["masks"]), sets


def undo_point(sets: list, active: int, current_idx: int):
    """Remove the last point for the active set's current frame."""
    if not (0 <= active < len(sets)):
        return None, None, sets
    s = sets[active]
    pts = s["points_map"].get(current_idx, [])
    if pts:
        pts.pop()
        if pts:
            rgb = s["frames"][current_idx]
            s["masks"][current_idx] = run_predictor_on_frame(rgb, pts)
        else:
            s["masks"][current_idx] = None
    return *current_views(s["frames"], s["points_map"], current_idx, s["masks"]), sets


def clear_points(sets: list, active: int, current_idx: int):
    """Clear all points and the mask for the active set's current frame."""
    if not (0 <= active < len(sets)):
        return None, None, sets
    s = sets[active]
    s["points_map"][current_idx] = []
    if current_idx < len(s["masks"]):
        s["masks"][current_idx] = None
    return *current_views(s["frames"], s["points_map"], current_idx, s["masks"]), sets


def change_frame(sets: list, active: int, frame_idx: int):
    """Switch the displayed frame when the slider moves."""
    idx = int(frame_idx)
    if not (0 <= active < len(sets)) or not sets[active]["frames"]:
        return None, None, idx
    s = sets[active]
    return *current_views(s["frames"], s["points_map"], idx, s["masks"]), idx
