#!/usr/bin/env python3
"""
Create a motion-trail composite from a set of images.

Steps
-----
1. Load all RGB frames in a directory.
2. For each frame, obtain the largest object mask using Segment Anything (SAM).
3. Save debug artefacts: binary mask and “object-only” crop.
4. Compute the median background across all frames.
5. Overlay the object layers on the background:
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import fire


def extract_largest_mask(masks: List[dict], min_area: int = 5_000) -> np.ndarray | None:
    """
    Pick the segmentation mask with the largest area that
    is at least `min_area` pixels; return *None* if nothing qualifies.
    """
    if not masks:
        return None
    largest = max(masks, key=lambda m: m["area"] if m["area"] > min_area else 0)
    return largest["segmentation"] if largest["area"] >= min_area else None


def load_images(folder: Path) -> Tuple[List[np.ndarray], List[Path]]:
    """
    Load every .png / .jpg / .jpeg in `folder` (non-recursive).

    Returns
    -------
    frames : list[np.ndarray]
        Images in BGR order, dtype=uint8.
    paths  : list[pathlib.Path]
        Matching list of file paths (sorted lexicographically).
    """
    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    frames = [cv2.imread(str(p)) for p in paths]
    return frames, paths


def generate_background(frames: List[np.ndarray]) -> np.ndarray:
    """
    Median pixel value across the time dimension → static background.
    """
    stack = np.stack(frames, axis=0).astype(np.uint8)
    return np.median(stack, axis=0).astype(np.uint8)


def overlay_object_on_background(
    background: np.ndarray,
    object_layers: List[Tuple[np.ndarray, np.ndarray]],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Compose the final image from background + object layers.

    Parameters
    ----------
    background : np.ndarray
        HxWx3 median background.
    object_layers : list[(frame, mask)]
        `frame`  : original image (HxWx3)
        `mask`   : binary mask where *True* marks object pixels
    alpha : float
        Blend ratio for intermediate frames.

    Returns
    -------
    output : np.ndarray
        Motion-trail composite.
    """
    output = background.copy()

    for idx, (frame, mask) in enumerate(object_layers):
        m = mask.astype(bool)
        if m.sum() == 0:  # skip empty masks
            continue

        if idx in (0, len(object_layers) - 1):  # paste opaquely
            output[m] = frame[m]
        else:  # alpha blend
            output[m] = (
                (1 - alpha) * output[m].astype(np.float32)
                + alpha * frame[m].astype(np.float32)
            ).astype(np.uint8)
    return output


def main(
    input_images_dir: str,
    model_path: str = "models/sam_vit_h_4b8939.pth",
    output_path: str = "outputs/sample_result.png",
    use_mask: bool = False,
    masks_dir: str = "data/masks/",
    min_area: int = 8_000,
    alpha: float = 0.7,
) -> None:
    """
    Parameters
    ----------
    input_images_dir : str
        Directory containing raw frames.
    model_path : str
        Segment-Anything checkpoint.
    output_path : str
        Where to save the composite PNG.
    use_mask : bool
        If *True*, reuse existing *_mask.png files instead of regenerating.
    masks_dir : str
        Folder for saving/loading per-frame masks & crops.
    min_area : int
        Minimum pixel area for candidate masks.
    alpha : float
        Blend factor for intermediate frames (0 = background, 1 = object).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Validate paths
    ckpt = Path(model_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    input_dir = Path(input_images_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    masks_root = Path(masks_dir)
    masks_root.mkdir(parents=True, exist_ok=True)

    # Load images
    frames, img_paths = load_images(input_dir)
    if not frames:
        print("[ERROR] No images found – abort.")
        return
    print(f"[INFO] {len(frames)} frames loaded")

    # Initialize SAM
    sam = sam_model_registry["vit_h"](checkpoint=str(ckpt)).to(device)
    mask_gen = SamAutomaticMaskGenerator(sam)

    # Generate or load masks
    object_layers: List[Tuple[np.ndarray, np.ndarray]] = []
    for frame, path in tqdm(
        list(zip(frames, img_paths)), desc="Masking frames", ncols=80
    ):
        stem = path.stem
        mask_png = masks_root / f"{stem}_mask.png"
        object_png = masks_root / f"{stem}_object_only.png"

        if use_mask and mask_png.is_file():
            mask = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8) if mask is not None else None
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            masks = mask_gen.generate(rgb)
            mask = extract_largest_mask(masks, min_area)
            if mask is not None:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                # Invert so *1 = background* / *0 = object*
                mask = np.logical_not(mask).astype(np.uint8)
                cv2.imwrite(str(mask_png), mask * 255)

        if mask is None:
            print(f"[WARN] No suitable mask for {path.name} - skipping")
            continue

        # Save a debug crop showing only the object
        mask3 = cv2.merge([mask] * 3)
        cv2.imwrite(str(object_png), cv2.bitwise_and(frame, mask3))

        object_layers.append((frame, mask))

    if not object_layers:
        print("[ERROR] No valid object layers - abort.")
        return

    # Compose final output
    background = generate_background(frames)
    composite = overlay_object_on_background(background, object_layers, alpha)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), composite)

    print(f"[INFO] Composite saved to {out_path}")
    print(f"[INFO] Masks and crops saved to {masks_root}/")


if __name__ == "__main__":
    fire.Fire(main)
