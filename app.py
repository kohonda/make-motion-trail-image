#!/usr/bin/env python3
"""Motion-trail image creator: Gradio GUI entry point."""

from __future__ import annotations

from contextlib import contextmanager

import gradio as gr

from motion_trail.frames import VIDEO_EXTS
from motion_trail.session import list_sessions
from motion_trail.ui.edit import (
    add_set,
    change_frame,
    clear_points,
    load_image_files,
    load_video_frames,
    move_set,
    on_image_click,
    on_video_drop,
    remove_set,
    select_set,
    set_background,
    set_color,
    toggle_no_color,
    undo_point,
)
from motion_trail.ui.render import (
    DEFAULT_SETTINGS,
    EMPHASIS_MODES,
    generate_and_autosave,
    generate_video_and_autosave,
    restore_session_cb,
    save_session_cb,
)
from motion_trail.ui.state import new_set, next_color, rgb_to_hex


# Each workflow step is a card with its own accent colour.
UI_CSS = """
.step {
    border: 1px solid var(--border-color-primary);
    border-left: 6px solid var(--step-color);
    border-radius: 12px;
    padding: 12px 16px !important;
    background: var(--block-background-fill);
}
.step-1 { --step-color: #6366f1; }
.step-2 { --step-color: #0ea5e9; }
.step-3 { --step-color: #10b981; }
.step-4 { --step-color: #f59e0b; }
.step-5 { --step-color: #ef4444; }
.step-title h3 {
    margin: 0;
    font-size: 1.3rem;
    color: var(--step-color);
}
.step-hint p { margin: 0; color: var(--body-text-color-subdued); }
"""


@contextmanager
def _step(n: int, title: str, hint: str = ""):
    """A workflow-step card: numbered title, optional hint, then its widgets."""
    with gr.Column(elem_classes=["step", f"step-{n}"]) as col:
        gr.Markdown(f"### {n}. {title}", elem_classes=["step-title"])
        if hint:
            gr.Markdown(hint, elem_classes=["step-hint"])
        yield col


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Motion Trail – SAM 3") as demo:
        gr.Markdown("# Motion Trail Image Creator (SAM 3)")

        init_color = next_color(0)

        # ---- state ----
        st_sets = gr.State([new_set(init_color)])  # list[set dict]
        st_active = gr.State(0)  # active set index
        st_idx = gr.State(0)  # current frame within active set
        st_bg = gr.State(None)  # chosen background frame (BGR)
        st_video = gr.State(None)  # original path of the dropped video

        with gr.Accordion("Session – save / restore work in progress", open=False):
            with gr.Row():
                session_name = gr.Textbox(
                    label="Session name",
                    placeholder="blank = timestamp",
                    scale=3,
                )
                save_session_btn = gr.Button("Save session", scale=1)
            autosave_checkbox = gr.Checkbox(
                label="Autosave on generate",
                value=True,
                info="Updates the session named above after every image or video.",
            )
            with gr.Row():
                session_selector = gr.Dropdown(
                    choices=list_sessions(),
                    value=None,
                    label="Saved sessions (newest first)",
                    scale=3,
                )
                refresh_sessions_btn = gr.Button("Refresh list", scale=1)
                restore_session_btn = gr.Button("Restore session", scale=1)

        with _step(
            1,
            "Choose a set",
            "One set per object. Sets are drawn in order, so the **last set is "
            "on top** where trails overlap.",
        ):
            with gr.Row():
                set_selector = gr.Radio(
                    choices=["Set 1"], value="Set 1", label="Active set", scale=4
                )
                add_btn = gr.Button("+ Add Set", scale=1)
                remove_btn = gr.Button("Remove Set", scale=1)
            with gr.Row():
                move_earlier_btn = gr.Button("◀ Move earlier (behind)")
                move_later_btn = gr.Button("▶ Move later (on top)")
            with gr.Row():
                color_picker = gr.ColorPicker(
                    label="Set colour", value=rgb_to_hex(init_color)
                )
                no_color_checkbox = gr.Checkbox(
                    label="No colour (keep original)", value=False
                )

        with _step(
            2,
            "Load frames",
            "Drop an image folder to load it immediately, or drop a video, set "
            "the range and click **Extract frames from video**.",
        ):
            with gr.Row():
                image_drop = gr.File(
                    label="Drop an image folder here",
                    file_count="directory",
                    height=120,
                )
                # a plain file box as the drop target, so the raw codec is never played
                video_drop = gr.File(
                    label="Drop a video here",
                    file_count="single",
                    file_types=sorted(VIDEO_EXTS),
                    height=120,
                )
            video_player = gr.Video(label="Video preview", interactive=False)
            with gr.Row():
                start_sec = gr.Textbox(
                    label="Start (video)",
                    value=DEFAULT_SETTINGS["start_sec"],
                    placeholder="sec or mm:ss.s, e.g. 1:23.5",
                )
                end_sec = gr.Textbox(
                    label="End (0 = until end)",
                    value=DEFAULT_SETTINGS["end_sec"],
                    placeholder="sec or mm:ss.s, e.g. 2:05",
                )
                interval_sec = gr.Number(
                    label="Interval (sec, video)",
                    value=DEFAULT_SETTINGS["interval_sec"],
                    minimum=0.01,
                )
                extract_btn = gr.Button("Extract frames from video", scale=1)

        with _step(
            3,
            "Annotate each frame",
            "Click the object (Positive) or areas to exclude (Negative); move "
            "between frames with the slider.",
        ):
            with gr.Row():
                input_image = gr.Image(label="Click to add points", interactive=False)
                preview_image = gr.Image(label="Mask preview", interactive=False)
            with gr.Row():
                mode_radio = gr.Radio(
                    ["Positive", "Negative"],
                    value="Positive",
                    label="Point mode",
                )
                undo_btn = gr.Button("Undo")
                clear_btn = gr.Button("Clear")
            frame_slider = gr.Slider(
                minimum=0,
                maximum=0,
                step=1,
                value=0,
                label="Frame",
            )

        with _step(
            4,
            "Pick a background",
            "Without one, the first frame of the first annotated set is used.",
        ):
            with gr.Row():
                bg_btn = gr.Button("Use current frame as background")
                bg_preview = gr.Image(label="Background", interactive=False)

        with _step(5, "Generate"):
            with gr.Row():
                alpha_slider = gr.Slider(
                    0.0, 1.0, value=DEFAULT_SETTINGS["alpha"], step=0.05, label="Alpha"
                )
                tint_slider = gr.Slider(
                    0.0,
                    1.0,
                    value=DEFAULT_SETTINGS["tint_strength"],
                    step=0.05,
                    label="Tint strength",
                )
                emphasis_radio = gr.Radio(
                    list(EMPHASIS_MODES),
                    value=DEFAULT_SETTINGS["emphasis"],
                    label="Emphasize (opaque) frames",
                )
            with gr.Row(equal_height=False):
                with gr.Column():
                    out_path = gr.Textbox(
                        label="Output path (.png / .jpg / .webp / .bmp / .tiff)",
                        value=DEFAULT_SETTINGS["output_path"],
                    )
                    gen_btn = gr.Button("Generate Motion Trail", variant="primary")
                    # format=png only applies to an array (a format the browser can't show)
                    result_image = gr.Image(
                        label="Result", interactive=False, format="png"
                    )
                with gr.Column():
                    with gr.Row():
                        video_fps = gr.Number(
                            label="Video FPS",
                            value=DEFAULT_SETTINGS["video_fps"],
                            minimum=0.1,
                            info="Encoding rate only — the pace comes from Interval (sec).",
                        )
                        video_out_path = gr.Textbox(
                            label="Video output path (.mp4 / .mov / .mkv / .avi)",
                            value=DEFAULT_SETTINGS["video_output_path"],
                        )
                    gen_video_btn = gr.Button("Generate Trail Video", variant="primary")
                    result_video = gr.Video(label="Trail video", interactive=False)

        # ---- wiring ----
        # .input / .release so programmatic updates don't re-trigger handlers
        set_selector.input(
            select_set,
            inputs=[st_sets, set_selector],
            outputs=[
                st_active,
                st_idx,
                input_image,
                preview_image,
                frame_slider,
                color_picker,
                no_color_checkbox,
                start_sec,
                end_sec,
                interval_sec,
            ],
        )

        add_btn.click(
            add_set,
            inputs=[st_sets],
            outputs=[
                st_sets,
                st_active,
                st_idx,
                set_selector,
                input_image,
                preview_image,
                frame_slider,
                color_picker,
                no_color_checkbox,
            ],
        )

        remove_btn.click(
            remove_set,
            inputs=[st_sets, st_active],
            outputs=[
                st_sets,
                st_active,
                st_idx,
                set_selector,
                input_image,
                preview_image,
                frame_slider,
                color_picker,
                no_color_checkbox,
                start_sec,
                end_sec,
                interval_sec,
            ],
        )

        move_earlier_btn.click(
            lambda sets, active: move_set(sets, active, -1),
            inputs=[st_sets, st_active],
            outputs=[st_sets, st_active, set_selector, color_picker],
        )

        move_later_btn.click(
            lambda sets, active: move_set(sets, active, 1),
            inputs=[st_sets, st_active],
            outputs=[st_sets, st_active, set_selector, color_picker],
        )

        color_picker.input(
            set_color,
            inputs=[st_sets, st_active, color_picker],
            outputs=[st_sets, no_color_checkbox],
        )

        no_color_checkbox.input(
            toggle_no_color,
            inputs=[st_sets, st_active, no_color_checkbox, color_picker],
            outputs=[st_sets],
        )

        image_drop.upload(
            load_image_files,
            inputs=[image_drop, st_sets, st_active],
            outputs=[
                input_image,
                preview_image,
                frame_slider,
                st_idx,
                st_sets,
                video_player,
            ],
        )

        video_drop.upload(
            on_video_drop,
            inputs=[video_drop],
            outputs=[st_video, video_player],
        )

        extract_btn.click(
            load_video_frames,
            inputs=[st_video, st_sets, st_active, start_sec, end_sec, interval_sec],
            outputs=[
                input_image,
                preview_image,
                frame_slider,
                st_idx,
                st_sets,
                video_player,
            ],
        )

        input_image.select(
            on_image_click,
            inputs=[st_sets, st_active, st_idx, mode_radio],
            outputs=[input_image, preview_image, st_sets],
        )

        undo_btn.click(
            undo_point,
            inputs=[st_sets, st_active, st_idx],
            outputs=[input_image, preview_image, st_sets],
        )

        clear_btn.click(
            clear_points,
            inputs=[st_sets, st_active, st_idx],
            outputs=[input_image, preview_image, st_sets],
        )

        frame_slider.release(
            change_frame,
            inputs=[st_sets, st_active, frame_slider],
            outputs=[input_image, preview_image, st_idx],
        )

        bg_btn.click(
            set_background,
            inputs=[st_sets, st_active, st_idx],
            outputs=[st_bg, bg_preview],
        )

        render_inputs = [
            st_sets,
            st_bg,
            alpha_slider,
            tint_slider,
            emphasis_radio,
            out_path,
            video_out_path,
            video_fps,
            st_active,
            st_idx,
            st_video,
            session_name,
            start_sec,
            end_sec,
            interval_sec,
            autosave_checkbox,
        ]

        gen_btn.click(
            generate_and_autosave,
            inputs=render_inputs,
            outputs=[result_image, session_selector, session_name],
        )

        gen_video_btn.click(
            generate_video_and_autosave,
            inputs=render_inputs,
            outputs=[result_video, session_selector, session_name],
        )

        save_session_btn.click(
            save_session_cb,
            inputs=[
                st_sets,
                st_active,
                st_idx,
                st_bg,
                st_video,
                session_name,
                start_sec,
                end_sec,
                interval_sec,
                alpha_slider,
                tint_slider,
                emphasis_radio,
                out_path,
                video_out_path,
                video_fps,
            ],
            outputs=[session_selector, session_name],
        )

        refresh_sessions_btn.click(
            lambda: gr.update(choices=list_sessions()),
            outputs=[session_selector],
        )

        restore_session_btn.click(
            restore_session_cb,
            inputs=[session_selector],
            outputs=[
                st_sets,
                st_active,
                st_idx,
                st_bg,
                st_video,
                set_selector,
                input_image,
                preview_image,
                frame_slider,
                color_picker,
                no_color_checkbox,
                bg_preview,
                video_player,
                start_sec,
                end_sec,
                interval_sec,
                alpha_slider,
                tint_slider,
                emphasis_radio,
                out_path,
                video_out_path,
                video_fps,
                session_name,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    # serve videos from anywhere: a local, single-user tool
    demo.launch(allowed_paths=["/"], css=UI_CSS)
