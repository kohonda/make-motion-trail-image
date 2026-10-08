"""Callbacks for generating the composite / video and saving / restoring sessions."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import gradio as gr

from motion_trail.compose import (
    compose_multi_set,
    compose_multi_set_progressive,
    pace_steps,
)
from motion_trail.session import (
    default_session_name,
    list_sessions,
    load_session,
    save_session,
)
from motion_trail.ui.edit import playable_video
from motion_trail.ui.state import (
    new_set,
    next_color,
    picker_hex,
    selector_update,
    show_set,
)
from motion_trail.video import VIDEO_OUT_EXTS, write_video

EMPHASIS_MODES = {
    "None": "none",
    "Last frame": "last",
    "First & last frames": "first_last",
}

OUTPUT_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
# displayable in the Result panel
BROWSER_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _resolve_output_path(output_path: str, exts: set, fallback: str) -> Path:
    """*output_path*, with an unsupported suffix replaced by *fallback*'s."""
    out = Path(str(output_path or "").strip() or fallback)
    if not out.name:
        out = Path(fallback)
    ext = out.suffix.lower()
    if ext not in exts:
        out = out.with_suffix(Path(fallback).suffix)
        gr.Warning(
            f"Unsupported output format '{ext or '(none)'}' – saved as {out.name}"
        )
    return out


def _trail_payload(sets: list, background, default_interval: float = 1.0):
    """Annotated sets in compose form + the background, or (None, None).

    Without a chosen background, the first annotated set's first frame is used.
    """
    usable = [
        s for s in sets if s["frames_bgr"] and any(m is not None for m in s["masks"])
    ]
    if not usable:
        gr.Warning("No sets with masks – annotate at least one set first")
        return None, None

    if background is None:
        background = usable[0]["frames_bgr"][0]

    payload = [
        {
            "frames_bgr": s["frames_bgr"],
            "masks": s["masks"],
            # RGB -> BGR, or None to keep the object's original colours
            "color_bgr": None if s["color"] is None else tuple(s["color"][::-1]),
            "interval_sec": (s.get("extract") or {}).get(
                "interval_sec", default_interval
            ),
        }
        for s in usable
    ]
    return payload, background


def generate_composite(
    sets: list,
    background,
    alpha: float,
    tint_strength: float,
    emphasis_label: str,
    output_path: str,
):
    """Overlay every annotated set's trail onto the chosen background."""
    payload, background = _trail_payload(sets, background)
    if payload is None:
        return None

    composite = compose_multi_set(
        payload,
        background,
        alpha=alpha,
        tint_strength=tint_strength,
        emphasis=EMPHASIS_MODES.get(emphasis_label, "last"),
    )

    out = _resolve_output_path(
        output_path, OUTPUT_EXTS, DEFAULT_SETTINGS["output_path"]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        written = cv2.imwrite(str(out), composite)
    except cv2.error as exc:
        gr.Warning(f"Could not write {out}: {exc}")
        return cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
    if not written:
        gr.Warning(f"Could not write {out}")
        return cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)

    gr.Info(f"Saved {out}")
    # a path, not an array: gr.Image would re-encode an array (as webp)
    if out.suffix.lower() in BROWSER_EXTS:
        return str(out.resolve())  # absolute: Gradio serves it regardless of cwd
    return cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)


def generate_video(
    sets: list,
    background,
    alpha: float,
    tint_strength: float,
    emphasis_label: str,
    video_output_path: str,
    fps: float,
    interval_sec: float,
):
    """Render the trail growing over time; *interval_sec* is for sets without one."""
    payload, background = _trail_payload(sets, background, interval_sec)
    if payload is None:
        return None

    out = _resolve_output_path(
        video_output_path, VIDEO_OUT_EXTS, DEFAULT_SETTINGS["video_output_path"]
    )
    steps = compose_multi_set_progressive(
        payload,
        background,
        alpha=alpha,
        tint_strength=tint_strength,
        emphasis=EMPHASIS_MODES.get(emphasis_label, "last"),
    )
    try:
        codec = write_video(pace_steps(steps, fps), out, fps)
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        gr.Warning(f"Could not write {out}: {exc}")
        return None

    if codec != "h264":
        gr.Warning(
            f"ffmpeg not found – wrote {out.name} as mpeg4, which desktop "
            "players handle but the preview below cannot show"
        )
    gr.Info(f"Saved {out}")
    return str(out.resolve())  # absolute: Gradio serves it regardless of cwd


# Initial widget values, and the fallback for settings a session lacks.
DEFAULT_SETTINGS = {
    "start_sec": "0",
    "end_sec": "0",
    "interval_sec": 1.0,
    "alpha": 0.7,
    "tint_strength": 0.5,
    "emphasis": "Last frame",
    "output_path": "outputs/sample_result.png",
    "video_output_path": "outputs/sample_result.mp4",
    "video_fps": 30.0,
}

# Number of outputs restore_session_cb feeds back into the UI.
_RESTORE_OUTPUTS = 23


def _no_restore():
    """Leave every restore output untouched."""
    return tuple(gr.update() for _ in range(_RESTORE_OUTPUTS))


def save_session_cb(
    sets: list,
    active: int,
    idx: int,
    background,
    video_path,
    name: str,
    start_sec: str,
    end_sec: str,
    interval_sec: float,
    alpha: float,
    tint_strength: float,
    emphasis_label: str,
    output_path: str,
    video_output_path: str,
    video_fps: float,
):
    """Write the current workspace (frames, masks, points, settings) to disk."""
    if not any(s["frames_bgr"] for s in sets):
        gr.Warning("Nothing to save – load frames into a set first")
        return gr.update(), gr.update()

    name = str(name or "").strip() or default_session_name()
    settings = {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "interval_sec": interval_sec,
        "alpha": alpha,
        "tint_strength": tint_strength,
        "emphasis": emphasis_label,
        "output_path": output_path,
        "video_output_path": video_output_path,
        "video_fps": video_fps,
    }
    try:
        path = save_session(
            name,
            sets,
            active=active,
            idx=idx,
            background_bgr=background,
            video_path=video_path,
            settings=settings,
        )
    except (OSError, ValueError) as exc:
        gr.Warning(f"Could not save the session: {exc}")
        return gr.update(), gr.update()

    gr.Info(f"Saved session to {path}")
    return gr.update(choices=list_sessions(), value=path.name), path.name


def _render_and_autosave(
    kind: str,
    sets: list,
    background,
    alpha: float,
    tint_strength: float,
    emphasis_label: str,
    output_path: str,
    video_output_path: str,
    video_fps: float,
    active: int,
    idx: int,
    video_path,
    name: str,
    start_sec: str,
    end_sec: str,
    interval_sec: float,
    autosave: bool,
):
    """Render the still or the video, then save the session if autosave is on."""
    if kind == "video":
        result = generate_video(
            sets,
            background,
            alpha,
            tint_strength,
            emphasis_label,
            video_output_path,
            video_fps,
            interval_sec,
        )
    else:
        result = generate_composite(
            sets, background, alpha, tint_strength, emphasis_label, output_path
        )
    if result is None or not autosave:
        return result, gr.update(), gr.update()
    selector, saved_name = save_session_cb(
        sets,
        active,
        idx,
        background,
        video_path,
        name,
        start_sec,
        end_sec,
        interval_sec,
        alpha,
        tint_strength,
        emphasis_label,
        output_path,
        video_output_path,
        video_fps,
    )
    return result, selector, saved_name


def generate_and_autosave(*render_inputs):
    return _render_and_autosave("image", *render_inputs)


def generate_video_and_autosave(*render_inputs):
    return _render_and_autosave("video", *render_inputs)


def restore_session_cb(name):
    """Repaint the whole workspace from a saved session."""
    if not name:
        gr.Warning("Select a saved session first")
        return _no_restore()
    try:
        data = load_session(name)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        gr.Warning(f"Could not restore session '{name}': {exc}")
        return _no_restore()

    sets = data["sets"] or [new_set(next_color(0))]
    active = min(max(data["active"], 0), len(sets) - 1)
    s = sets[active]

    # st_idx must match the slider: change_frame only fires on .release
    idx = min(max(data["idx"], 0), len(s["frames"]) - 1) if s["frames"] else 0

    bg = data["background_bgr"]
    bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB) if bg is not None else None

    # the uploaded video is usually gone by now; the frames are restored anyway
    video_path = data["video_path"]
    if video_path and Path(video_path).is_file():
        player = playable_video(video_path)
    else:
        video_path, player = None, None

    cfg = {**DEFAULT_SETTINGS, **data["settings"]}
    ex = s.get("extract") or {}
    emphasis = cfg["emphasis"]
    if emphasis not in EMPHASIS_MODES:
        emphasis = DEFAULT_SETTINGS["emphasis"]

    gr.Info(f"Restored session '{name}'")
    return (
        sets,  # st_sets
        active,  # st_active
        idx,  # st_idx
        bg,  # st_bg
        video_path,  # st_video
        selector_update(sets, active),  # set_selector
        *show_set(s, idx),  # input_image, preview_image, frame_slider
        picker_hex(s["color"], active),  # color_picker
        s["color"] is None,  # no_color_checkbox
        bg_rgb,  # bg_preview
        player,  # video_player
        ex.get("start_sec", cfg["start_sec"]),
        ex.get("end_sec", cfg["end_sec"]),
        ex.get("interval_sec", cfg["interval_sec"]),
        cfg["alpha"],
        cfg["tint_strength"],
        emphasis,
        cfg["output_path"],
        cfg["video_output_path"],
        cfg["video_fps"],
        name,  # session_name
    )
