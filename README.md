# Automatically create motion trail images using Segment Anything

## Setup

Download Segment Anything model from [here](https://github.com/facebookresearch/segment-anything).

For example,

```bash
cd make-motion-trail-image
mkdir models && curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o models/sam_vit_h_4b8939.pth
```

## Run

```bash
uv run main.py --input_images_dir <input_images_dir> --output_path <output_path> --model_path <model_path> 
```

For example, using our sample images:

```bash
python main.py --input_images_dir data/samples/
```

## Manual adjustment

<!-- masksのimagesを微調整すると結果を修正できるよ -->

You can manually adjust the masks in the `masks` directory using any image editing software. After editing, run the following command to generate the final motion trail image:

```bash
uv run main.py --input_images_dir <input_images_dir> --output_path <output_path> --model_path <model_path> --masks_dir <masks_dir> --use_masks True
```