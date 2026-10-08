# Motion Trail Image Creator

Create motion-trail composite images interactively using **SAM 3** (Segment Anything Model 3).

Click on each frame to select the object you want to extract, then generate a single composite image showing the object's motion across all frames.

<p align="center">
  <img src="media/gui.png" width="800">
</p>

## Setup

```bash
git clone git@github.com:kohonda/make-motion-trail-image.git
cd make-motion-trail-image
uv sync
```

The SAM 3 model checkpoint is automatically downloaded from HuggingFace on first run. A GPU with at least 8 GB VRAM is recommended.

> **Note:** The SAM 3 model weights are hosted on a gated HuggingFace repo. Once accepted, you need to be authenticated to download the checkpoints. You can do this by running the following steps:
>
> 1. Create or log in to your [Hugging Face](https://huggingface.co/) account.
> 2. Go to the SAM 3 model page and accept the license agreement.
> 3. Generate an access token at [Hugging Face Settings](https://huggingface.co/settings/tokens).
> 4. Run `huggingface-cli login` and paste your token when prompted.

## Usage

```bash
uv run app.py
```

Open http://127.0.0.1:7860 in your browser.

### Workflow

1. **Load frames** -- Drag & drop into the active set:
   - **Image folder** -- Images (`.png`, `.jpg`, `.jpeg`) sorted by filename, so use zero-padded names (e.g. `frame_001.png`).
   - **Video** -- Set **Start** / **End** (seconds or `mm:ss`; End `0` = whole clip) and **Interval (sec)**, then click **Extract frames from video**.
2. **Annotate each frame** -- Use the frame slider to navigate between frames. For each frame:
   - Select **Positive** mode and click on the object to segment (green dots).
   - Select **Negative** mode and click on areas to exclude (red dots).
   - Use **Undo** to remove the last point or **Clear** to reset the current frame.
3. **Add more objects** (optional) -- **+ Add Set** adds another object with its own frames and colour. The last set is drawn on top; reorder with **◀ / ▶**.
4. **Generate** -- Pick a background with **Use current frame as background**, adjust **Alpha** / **Tint strength** / **Emphasize**, then click:
   - **Generate Motion Trail** for a still image (`.png`, `.jpg`, `.webp`, `.bmp`, `.tiff`).
   - **Generate Trail Video** for a video in which the trail grows over time (`.mp4`, `.mov`, `.mkv`, `.avi`; H.264 when `ffmpeg` is installed).

   The format follows the output path's extension.

### Saving and resuming work

The **Session** panel saves all sets, annotations and settings to `sessions/<name>/`, and restores them later -- even after an app restart. With **Autosave** on, the session is saved every time you generate. Sessions store every frame as PNG and can be large, so delete ones you no longer need.

## How it works

1. For each frame, SAM 3's interactive predictor segments the target object based on positive/negative point prompts.
2. The segmented objects are alpha-blended onto the chosen background frame to create the motion-trail effect; the frames selected under **Emphasize** are pasted opaquely.

## License

See [LICENSE](LICENSE).
**NOTE**: This project depends on Segment Anything Model 3 (SAM3) released by Meta under the SAM License.
