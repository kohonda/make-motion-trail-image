"""Alpha-composite segmented objects into a motion trail, as a still or a growing video."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def overlay_object_on_background(
    background: np.ndarray,
    object_layers: List[Tuple[np.ndarray, np.ndarray]],
    alpha: float = 0.5,
    opaque_indices: set = frozenset(),
) -> np.ndarray:
    """Alpha-blend object layers onto *background*; *opaque_indices* are pasted solid."""
    output = background.copy()
    for idx, (frame, mask) in enumerate(object_layers):
        m = mask.astype(bool)
        if m.sum() == 0:
            continue
        if idx in opaque_indices:
            output[m] = frame[m]
        else:
            output[m] = (
                (1 - alpha) * output[m].astype(np.float32)
                + alpha * frame[m].astype(np.float32)
            ).astype(np.uint8)
    return output


def tint(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: Tuple[int, int, int],
    strength: float = 0.5,
) -> np.ndarray:
    """Blend the masked pixels toward *color_bgr* (0 = original, 1 = flat colour)."""
    out = frame_bgr.copy()
    m = mask.astype(bool)
    if m.sum() == 0:
        return out
    color = np.array(color_bgr, dtype=np.float32)
    out[m] = (
        (1 - strength) * frame_bgr[m].astype(np.float32) + strength * color
    ).astype(np.uint8)
    return out


def resize_to_canvas(
    frame_bgr: np.ndarray,
    mask: np.ndarray | None,
    size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Resize a frame and its mask to *size* = (H, W); the mask stays binary."""
    h, w = size
    if frame_bgr.shape[:2] != (h, w):
        frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    if mask is not None and mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return frame_bgr, mask


def _set_layers(
    s: dict,
    size: Tuple[int, int],
    tint_strength: float,
) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """``(frame_index, frame_bgr, mask)`` for every annotated frame of a set."""
    layers = []
    for i, (frame, mask) in enumerate(zip(s["frames_bgr"], s["masks"])):
        if mask is None:
            continue
        frame, mask = resize_to_canvas(frame, mask, size)
        color = s.get("color_bgr")
        if color is not None:
            frame = tint(frame, mask, color, tint_strength)
        layers.append((i, frame, mask))
    return layers


def _opaque_indices(n: int, emphasis: str) -> set:
    """Layer positions painted opaque for *n* layers under *emphasis*."""
    if emphasis == "none":
        return set()
    if emphasis == "first_last":
        return {0, n - 1}
    return {n - 1}  # "last"


def compose_multi_set(
    sets: List[dict],
    background_bgr: np.ndarray,
    alpha: float = 0.5,
    tint_strength: float = 0.5,
    emphasis: str = "last",
) -> np.ndarray:
    """Overlay every set's trail onto the background, in list order.

    Each set has ``frames_bgr``, ``masks`` (entries may be None) and
    ``color_bgr`` (None keeps the original colours). *emphasis* is ``"none"``,
    ``"last"`` or ``"first_last"``: which frames of each set are opaque.
    """
    size = background_bgr.shape[:2]
    output = background_bgr.copy()
    for s in sets:
        layers = [(f, m) for _, f, m in _set_layers(s, size, tint_strength)]
        if not layers:
            continue
        output = overlay_object_on_background(
            output, layers, alpha, opaque_indices=_opaque_indices(len(layers), emphasis)
        )
    return output


def compose_multi_set_progressive(
    sets: List[dict],
    background_bgr: np.ndarray,
    alpha: float = 0.5,
    tint_strength: float = 0.5,
    emphasis: str = "last",
):
    """Yield ``(time_sec, composite)`` as the trails grow.

    Each set's clock starts at its own first annotated frame and advances by its
    ``interval_sec``. The final step equals :func:`compose_multi_set`.
    """
    size = background_bgr.shape[:2]
    per_set = []
    for s in sets:
        layers = _set_layers(s, size, tint_strength)
        if not layers:
            per_set.append([])
            continue
        first = layers[0][0]
        step = max(float(s.get("interval_sec") or 1.0), 0.0)
        # rounded so sets landing on the same moment share one step
        per_set.append([(round((i - first) * step, 6), f, m) for i, f, m in layers])

    for t in sorted({at for layers in per_set for at, _, _ in layers}):
        output = background_bgr.copy()
        for layers in per_set:
            visible = [(f, m) for at, f, m in layers if at <= t]
            if not visible:
                continue
            output = overlay_object_on_background(
                output,
                visible,
                alpha,
                opaque_indices=_opaque_indices(len(visible), emphasis),
            )
        yield t, output


def pace_steps(steps, fps: float):
    """Repeat each ``(time_sec, image)`` step until the next one's time at *fps*.

    Positions come from absolute times so rounding cannot drift; every step gets
    at least one frame, and the last is held as long as the one before it.
    """
    rate = max(float(fps), 0.1)
    first = None
    previous = None
    previous_at = None
    last_gap = None
    emitted = 0
    for at, image in steps:
        if first is None:
            first = at
        else:
            last_gap = at - previous_at
            target = max(int(round((at - first) * rate)), emitted + 1)
            for _ in range(target - emitted):
                yield previous
            emitted = target
        previous, previous_at = image, at
    if previous is not None:
        tail = last_gap if last_gap else 1.0 / rate
        for _ in range(max(int(round(tail * rate)), 1)):
            yield previous
