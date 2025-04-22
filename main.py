import os
import glob
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import fire


def extract_largest_mask(masks, min_area=5000):
    if len(masks) == 0:
        return None
    largest = max(masks, key=lambda x: x["area"] if x["area"] > min_area else 0)
    if largest["area"] < min_area:
        return None
    return largest["segmentation"]


def load_images(folder):
    paths = sorted(
        glob.glob(os.path.join(folder, "*.png"))
        + glob.glob(os.path.join(folder, "*.jpg"))
        + glob.glob(os.path.join(folder, "*.jpeg"))
    )
    frames = [cv2.imread(p) for p in paths]
    return frames, paths


def generate_background(frames):
    stack = np.stack(frames, axis=0).astype(np.uint8)
    bg = np.median(stack, axis=0).astype(np.uint8)
    return bg


def overlay_object_on_background(background, object_layers, alpha=0.5):
    output = background.copy()

    for i, (object, mask) in enumerate(object_layers):
        mask_bool = mask.astype(bool)

        if np.sum(mask_bool) == 0:
            print(f"[WARN] Skipping frame {i}: empty mask")
            continue

        if i == 0 or i == len(object_layers) - 1:
            for c in range(3):
                output[:, :, c][mask_bool] = object[:, :, c][mask_bool]
        else:
            for c in range(3):
                fg = object[:, :, c].astype(np.float32)
                bg = output[:, :, c].astype(np.float32)
                bg[mask_bool] = (1 - alpha) * bg[mask_bool] + alpha * fg[mask_bool]
                output[:, :, c] = bg.astype(np.uint8)
    return output


def main(
    input_images_dir: str,
    model_path: str = "models/sam_vit_h_4b8939.pth",
    output_path: str = "data/sample_result.png",
    use_mask: bool = False,
    masks_dir: str = "data/masks/",
    min_area: int = 8000,
    alpha: float = 0.5,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path(model_path)
    if not model_path.is_file():
        print(f"[ERROR] Model file does not exist: {model_path}")
        return
    sam = sam_model_registry["vit_h"](checkpoint=model_path).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    input_images_dir = Path(input_images_dir)
    # check if the input directory exists
    if not input_images_dir.is_dir():
        print(f"[ERROR] Input directory does not exist: {input_images_dir}")
        return

    frames, paths = load_images(Path(input_images_dir))
    print(f"[INFO] Loaded {len(frames)} images")

    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    object_layers = []
    for frame, path in tqdm(
        zip(frames, paths), desc="Processing frames", total=len(frames)
    ):
        basename = os.path.splitext(os.path.basename(path))[0]
        mask_png_path = os.path.join(masks_dir, f"{basename}_mask.png")
        masked_object_path = os.path.join(masks_dir, f"{basename}_object_only.png")

        if os.path.exists(mask_png_path) and use_mask:
            mask = cv2.imread(mask_png_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[WARN] Failed to load mask image: {mask_png_path}")
                continue
            mask = (mask > 127).astype(np.uint8)
        else:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            masks = mask_generator.generate(image_rgb)
            mask = extract_largest_mask(masks, min_area)
            if mask is None:
                print(f"[WARN] Skipping {basename} due to no valid mask")
                continue
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = np.logical_not(mask).astype(np.uint8)
            cv2.imwrite(mask_png_path, mask * 255)

        mask3 = cv2.merge([mask] * 3)
        object_only = cv2.bitwise_and(frame, mask3)
        cv2.imwrite(masked_object_path, object_only)

        object_layers.append((frame, mask))

    if len(object_layers) == 0:
        print("[ERROR] No valid object masks found.")
        return

    background = generate_background(frames)
    output = overlay_object_on_background(background, object_layers, alpha)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, output)
    print(f"[INFO] Saved final image to {output_path}")
    print(f"[INFO] Debug masks saved to {masks_dir}/")


if __name__ == "__main__":
    fire.Fire(main)
