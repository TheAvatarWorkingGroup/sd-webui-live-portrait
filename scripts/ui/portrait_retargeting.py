import gradio as gr
from ..utils import add_source_image_buttons


def create_portrait_retargeting_tab(pipeline_handler):
    """Create the portrait retargeting tab UI."""
    with gr.Tab("Portrait Retargeting"):
        with gr.Row():
            with gr.Column():
                retargeting_input_image = gr.Image(
                    type="filepath", label="Source Image"
                )
                add_source_image_buttons(retargeting_input_image)

                with gr.Accordion(open=True, label="Retargeting Options"):
                    flag_do_crop_input_retargeting = gr.Checkbox(
                        value=True, label="Crop source"
                    )
                    flag_stitching_retargeting = gr.Checkbox(
                        value=True, label="Stitching"
                    )
                    flag_animate_transition = gr.Checkbox(
                        value=False, label="Animate transition"
                    )
                    retargeting_source_scale = gr.Number(
                        value=2.5,
                        label="Crop scale",
                        minimum=1.8,
                        maximum=3.2,
                        step=0.05,
                    )
                    eye_retargeting_slider = gr.Slider(
                        minimum=0,
                        maximum=0.8,
                        step=0.01,
                        label="Target eyes-open ratio",
                    )
                    lip_retargeting_slider = gr.Slider(
                        minimum=0,
                        maximum=0.8,
                        step=0.01,
                        label="Target lip-open ratio",
                    )

                with gr.Accordion(open=True, label="Facial Movement"):
                    with gr.Row():
                        head_pitch = gr.Slider(
                            minimum=-15.0,
                            maximum=15.0,
                            value=0,
                            step=1,
                            label="Relative pitch",
                        )
                        head_yaw = gr.Slider(
                            minimum=-25.0,
                            maximum=25.0,
                            value=0,
                            step=1,
                            label="Relative yaw",
                        )
                        head_roll = gr.Slider(
                            minimum=-15.0,
                            maximum=15.0,
                            value=0,
                            step=1,
                            label="Relative roll",
                        )
                    with gr.Row():
                        mov_x = gr.Slider(
                            minimum=-0.19,
                            maximum=0.19,
                            value=0.0,
                            step=0.01,
                            label="X-axis movement",
                        )
                        mov_y = gr.Slider(
                            minimum=-0.19,
                            maximum=0.19,
                            value=0.0,
                            step=0.01,
                            label="Y-axis movement",
                        )
                        mov_z = gr.Slider(
                            minimum=0.9,
                            maximum=1.2,
                            value=1.0,
                            step=0.01,
                            label="Z-axis movement",
                        )

                with gr.Accordion(open=True, label="Facial Expression"):
                    with gr.Row():
                        lip_variation_zero = gr.Slider(
                            minimum=-0.09,
                            maximum=0.09,
                            value=0,
                            step=0.01,
                            label="Pouting",
                        )
                        lip_variation_one = gr.Slider(
                            minimum=-20.0,
                            maximum=15.0,
                            value=0,
                            step=0.01,
                            label="Pursing 😐",
                        )
                        lip_variation_two = gr.Slider(
                            minimum=0.0,
                            maximum=15.0,
                            value=0,
                            step=0.01,
                            label="Grin 😁",
                        )
                    with gr.Row():
                        lip_variation_three = gr.Slider(
                            minimum=-90.0,
                            maximum=120.0,
                            value=0,
                            step=1.0,
                            label="Lip close <-> open",
                        )
                        smile = gr.Slider(
                            minimum=-0.3,
                            maximum=1.3,
                            value=0,
                            step=0.01,
                            label="Smile 😄",
                        )
                        wink = gr.Slider(
                            minimum=0,
                            maximum=39,
                            value=0,
                            step=0.01,
                            label="Wink 😉",
                        )
                    with gr.Row():
                        eyebrow = gr.Slider(
                            minimum=-30,
                            maximum=30,
                            value=0,
                            step=0.01,
                            label="Eyebrow 🤨",
                        )
                        eyeball_x = gr.Slider(
                            minimum=-30.0,
                            maximum=30.0,
                            value=0,
                            step=0.01,
                            label="Eye gaze (horizontal) 👀",
                        )
                        eyeball_y = gr.Slider(
                            minimum=-63.0,
                            maximum=63.0,
                            value=0,
                            step=0.01,
                            label="Eye gaze (vertical) 🙄",
                        )

            with gr.Column():
                with gr.Row():
                    retargeting_output = gr.Image(
                        type="numpy", label="Retargeting Result"
                    )
                    retargeting_output_paste_back = gr.Image(
                        type="numpy", label="Paste-back Result"
                    )
                with gr.Row():
                    retargeting_output_video = gr.Video(
                        autoplay=True, visible=False, label="Animation Result"
                    )
                    retargeting_output_video_paste_back = gr.Video(
                        autoplay=True,
                        visible=False,
                        label="Animation Result with Paste-back",
                    )

        with gr.Row():
            reset_retargeting_button = gr.Button("🔄 Reset")
            clear_retargeting_button = gr.Button("🧹 Clear")

        # Return all components that need to be accessed from outside
        return {
            "retargeting_input_image": retargeting_input_image,
            "flag_do_crop_input_retargeting": flag_do_crop_input_retargeting,
            "flag_stitching_retargeting": flag_stitching_retargeting,
            "flag_animate_transition": flag_animate_transition,
            "retargeting_source_scale": retargeting_source_scale,
            "eye_retargeting_slider": eye_retargeting_slider,
            "lip_retargeting_slider": lip_retargeting_slider,
            "head_pitch": head_pitch,
            "head_yaw": head_yaw,
            "head_roll": head_roll,
            "mov_x": mov_x,
            "mov_y": mov_y,
            "mov_z": mov_z,
            "lip_variation_zero": lip_variation_zero,
            "lip_variation_one": lip_variation_one,
            "lip_variation_two": lip_variation_two,
            "lip_variation_three": lip_variation_three,
            "smile": smile,
            "wink": wink,
            "eyebrow": eyebrow,
            "eyeball_x": eyeball_x,
            "eyeball_y": eyeball_y,
            "retargeting_output": retargeting_output,
            "retargeting_output_paste_back": retargeting_output_paste_back,
            "retargeting_output_video": retargeting_output_video,
            "retargeting_output_video_paste_back": retargeting_output_video_paste_back,
            "reset_retargeting_button": reset_retargeting_button,
            "clear_retargeting_button": clear_retargeting_button,
        }
