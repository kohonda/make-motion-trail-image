"""SAM 3 model management and per-frame interactive segmentation."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

_model = None
_processor: Sam3Processor | None = None


def _get_model_and_processor(device: str = None):
    """Build or return the cached SAM 3 model and processor."""
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Loading SAM 3 model …")
    _model = build_sam3_image_model(
        device=device,
        load_from_HF=True,
        enable_segmentation=True,
        enable_inst_interactivity=True,
    )
    _processor = Sam3Processor(_model, device=device)
    print("[INFO] SAM 3 model loaded")
    return _model, _processor


def run_predictor_on_frame(
    rgb: np.ndarray,
    points: list[tuple[int, int, int]],
) -> np.ndarray | None:
    """Run SAM 3 interactive predictor on a single frame.

    Returns (H, W) uint8 mask or None.
    """
    if not points:
        return None
    model, processor = _get_model_and_processor()

    # Encode image through the shared backbone (use PIL to avoid numpy shape bug)
    state = processor.set_image(Image.fromarray(rgb))

    # Run interactive point prediction
    coords = np.array([[x, y] for x, y, _ in points])
    labels = np.array([lab for _, _, lab in points])
    masks, scores, _ = model.predict_inst(
        state,
        point_coords=coords,
        point_labels=labels,
        multimask_output=True,
    )
    # masks: (3, H, W), scores: (3,)
    best = int(np.argmax(scores))
    mask = masks[best].astype(np.uint8)

    # Resize mask to match input image if dimensions differ
    h, w = rgb.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask
