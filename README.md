# Automatically Create Motion Trail Images with Segment Anything

<div style="text-align: center;">
  <img src="media/example.svg" width="500">
</div>

## Setup

Download the Segment Anything model from [here](https://github.com/facebookresearch/segment-anything).

For example:

```bash
cd make-motion-trail-image
mkdir -p models
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
  -o models/sam_vit_h_4b8939.pth
```

## Usage

Run the script with the required options:

```bash
uv run main.py \
  --input_images_dir <input_images_dir> \
  --output_path <output_path> \
  --model_path <model_path>
```

For example, to process the sample images:

```bash
uv run main.py --input_images_dir data/samples/
```

## Manual Mask Editing

You can manually edit the mask images in the `masks` directory using any image editing tool. After editing, regenerate the final motion-trail image by running:

```bash
uv run main.py \
  --input_images_dir <input_images_dir> \
  --output_path <output_path> \
  --model_path <model_path> \
  --masks_dir <masks_dir> \
  --use_masks True
```

