# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interactive Gradio GUI tool that creates motion-trail composite images using SAM 3 (Segment Anything Model 3). Users click on objects across a sequence of frames to segment them, then generate a single composite showing the object's motion trail over a background frame the user picks.

## Commands

```bash
# Install dependencies
uv sync

# Run the app (Gradio server at http://127.0.0.1:7860)
uv run app.py

# Install pre-commit hooks
make setup-hooks

# Run linting/formatting checks (ruff lint + format, plus general file checks)
make check-hooks
```

## Architecture

- **`app.py`** — entry point; `build_ui` lays out the Gradio Blocks and wires up the callbacks.
- **`motion_trail/`** — framework-independent logic:
  - `frames.py` (`load_images`, `load_video`) — load frames from an image folder or a video.
  - `compose.py` (`compose_multi_set`, `compose_multi_set_progressive`, `pace_steps`) — alpha-composite the segmented objects as a still, or as a trail growing over time.
  - `video.py` (`write_video`) — encode frames with ffmpeg H.264, falling back to OpenCV.
  - `session.py` (`save_session`, `load_session`, `list_sessions`) — save / restore the whole workspace under `sessions/<name>/`.
  - `sam.py` (`run_predictor_on_frame`) — lazily loads SAM 3 from HuggingFace and runs point-prompt segmentation per frame. The `sam3` package is installed from the Facebook Research GitHub repo.
- **`motion_trail/ui/`** — Gradio callbacks:
  - `state.py` — set records and preview drawing.
  - `edit.py` — set management, frame loading and annotation.
  - `render.py` — composite / video generation and session save / restore.

Key data flow: frames are stored in both RGB (for display/SAM) and BGR (for OpenCV compositing). Each set in `st_sets` holds its point prompts as `points_map: dict[int, list[(x, y, label)]]` and its masks as `masks: list[np.ndarray | None]`, where `None` means never annotated.

## Linting and Formatting

You DO NOT need to run any formatting or linting commands manually. The pre-commit hooks will automatically check and format code on commit.

## Requirements

- Python >=3.12 (pinned in `.python-version`)
- GPU with >=8GB VRAM recommended; falls back to CPU
- SAM 3 checkpoint auto-downloaded from HuggingFace on first run
