"""Save and restore the work in progress as a session."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np

# sessions/<name>/
#     session.json          points, colours, widget settings
#     background.png        chosen background frame, if any
#     set00/frame_0000.png  the set's frames (stored, since uploads don't persist)
#     set00/mask_0000.png   masks, only for frames that have one

SESSIONS_DIR = Path(__file__).resolve().parents[1] / "sessions"
SESSION_VERSION = 1


def _sanitize_name(name) -> str:
    """Reduce user input to a safe single path component (may be empty)."""
    return re.sub(r"[^\w\-. ]", "_", str(name or "").strip()).strip(". ")


def _session_dir(name) -> Path:
    """Resolve ``sessions/<name>``, refusing anything outside SESSIONS_DIR."""
    safe = _sanitize_name(name)
    root = SESSIONS_DIR.resolve()
    path = (root / safe).resolve() if safe else root
    if not safe or path.parent != root:
        raise ValueError(f"Invalid session name: {name!r}")
    return path


def list_sessions() -> List[str]:
    """Names of saved sessions, most recently written first."""
    if not SESSIONS_DIR.is_dir():
        return []
    dirs = [p for p in SESSIONS_DIR.iterdir() if (p / "session.json").is_file()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in dirs]


def default_session_name() -> str:
    """Timestamped fallback name for a session saved without one."""
    return datetime.now().strftime("session_%Y%m%d_%H%M%S")


def _array_key(arr: np.ndarray) -> str:
    """Content hash of an image, used to skip re-encoding unchanged files."""
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _at(seq, i):
    """``seq[i]`` or None – reads a hash list that may be short or missing."""
    return seq[i] if seq and i < len(seq) else None


def _read_manifest(path: Path) -> dict:
    """The session's previous ``session.json``, or {} if absent / unreadable."""
    try:
        meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


# Only these names are pruned, so user files placed in a session dir survive.
_OWNED_TOP = re.compile(r"set\d+$")
_OWNED_SET = re.compile(r"(frame|mask)_\d+\.png$")


def _prune(directory: Path, keep: set, owned: re.Pattern) -> None:
    """Delete the session's own *owned* entries in *directory* unless kept."""
    for entry in directory.iterdir():
        if entry.name in keep or not owned.match(entry.name):
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def save_session(
    name: str,
    sets: List[dict],
    active: int = 0,
    idx: int = 0,
    background_bgr: np.ndarray | None = None,
    video_path: str | None = None,
    settings: dict | None = None,
) -> Path:
    """Write the workspace to ``sessions/<name>/`` and return that dir.

    Incremental: images whose content hash is unchanged are not re-encoded.
    """
    out = _session_dir(_sanitize_name(name) or default_session_name())
    prev_meta = _read_manifest(out) if out.is_dir() else {}
    previous = prev_meta.get("sets")
    previous = previous if isinstance(previous, list) else []
    out.mkdir(parents=True, exist_ok=True)

    meta_sets = []
    for i, s in enumerate(sets):
        sub = out / f"set{i:02d}"
        sub.mkdir(exist_ok=True)
        old = previous[i] if i < len(previous) else {}
        old_frames = old.get("frame_hashes")
        old_masks = old.get("mask_hashes")

        frames = list(s.get("frames_bgr") or [])
        masks = list(s.get("masks") or [])
        frame_hashes: List[str] = []
        mask_hashes: List[str | None] = []
        keep = set()
        for j, frame in enumerate(frames):
            key = _array_key(frame)
            frame_hashes.append(key)
            path = sub / f"frame_{j:04d}.png"
            keep.add(path.name)
            if key != _at(old_frames, j) or not path.is_file():
                cv2.imwrite(str(path), frame)

            mask = masks[j] if j < len(masks) else None
            if mask is None:
                mask_hashes.append(None)  # absence of a file means "no mask"
                continue
            binary = (mask.astype(np.uint8) > 0).astype(np.uint8)
            key = _array_key(binary)
            mask_hashes.append(key)
            path = sub / f"mask_{j:04d}.png"
            keep.add(path.name)
            if key != _at(old_masks, j) or not path.is_file():
                cv2.imwrite(str(path), binary * 255)  # 0/1 -> 0/255, viewable
        _prune(sub, keep, _OWNED_SET)

        extract = s.get("extract") or None
        if extract:
            extract = {
                "start_sec": str(extract.get("start_sec", "")),
                "end_sec": str(extract.get("end_sec", "")),
                "interval_sec": float(extract.get("interval_sec", 1.0)),
            }

        color = s.get("color")
        meta_sets.append(
            {
                "dir": s.get("dir", ""),
                "extract": extract,
                "color": [int(c) for c in color] if color is not None else None,
                "n_frames": len(frames),
                # JSON keys are strings; load_session converts them back to int
                "points_map": {
                    str(k): [[int(x), int(y), int(lab)] for x, y, lab in v]
                    for k, v in (s.get("points_map") or {}).items()
                },
                "frame_hashes": frame_hashes,
                "mask_hashes": mask_hashes,
            }
        )

    bg_path = out / "background.png"
    bg_hash = None
    if background_bgr is not None:
        bg_hash = _array_key(background_bgr)
        if bg_hash != prev_meta.get("background_hash") or not bg_path.is_file():
            cv2.imwrite(str(bg_path), background_bgr)
    elif bg_path.is_file():
        bg_path.unlink()

    meta = {
        "version": SESSION_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "active": int(active),
        "idx": int(idx),
        "video_path": str(video_path) if video_path else None,
        "has_background": background_bgr is not None,
        "background_hash": bg_hash,
        "settings": dict(settings or {}),
        "sets": meta_sets,
    }
    # Drop set dirs left behind by a previous save that had more sets.
    _prune(out, {f"set{i:02d}" for i in range(len(sets))}, _OWNED_TOP)
    (out / "session.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def load_session(name: str) -> dict:
    """Read a session back into app-shaped state."""
    path = _session_dir(name)
    meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
    version = int(meta.get("version", 0))
    if version > SESSION_VERSION:
        raise ValueError(
            f"session format v{version} is newer than this app (v{SESSION_VERSION})"
        )

    sets: List[dict] = []
    for i, m in enumerate(meta.get("sets", [])):
        sub = path / f"set{i:02d}"
        frames_bgr: List[np.ndarray] = []
        masks: List[np.ndarray | None] = []
        for j in range(int(m.get("n_frames", 0))):
            frame = cv2.imread(str(sub / f"frame_{j:04d}.png"))
            if frame is None:
                break  # truncate rather than skip, to keep frame indices aligned
            frames_bgr.append(frame)
            mask_path = sub / f"mask_{j:04d}.png"
            mask = (
                cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_path.is_file()
                else None
            )
            # None (never annotated) must stay distinct from an all-zero mask
            masks.append(None if mask is None else (mask > 127).astype(np.uint8))
        color = m.get("color")
        sets.append(
            {
                "dir": m.get("dir", ""),
                "frames": [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr],
                "frames_bgr": frames_bgr,
                # JSON stringified the int frame indices – convert them back
                "points_map": {
                    int(k): [tuple(p) for p in v]
                    for k, v in (m.get("points_map") or {}).items()
                },
                "masks": masks,
                "color": tuple(color) if color is not None else None,
                "extract": m.get("extract") or None,
            }
        )

    background = None
    if meta.get("has_background"):
        background = cv2.imread(str(path / "background.png"))

    return {
        "sets": sets,
        "active": int(meta.get("active", 0)),
        "idx": int(meta.get("idx", 0)),
        "background_bgr": background,
        "video_path": meta.get("video_path") or None,
        "settings": meta.get("settings") or {},
    }
