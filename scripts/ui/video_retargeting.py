import gradio as gr


def create_video_retargeting_tab(pipeline_handler):
    """Create the video retargeting tab UI."""
    with gr.Tab("Video Retargeting"):
        with gr.Row():
            with gr.Column():
                retargeting_input_video = gr.Video(label="Source Video")
                with gr.Accordion(open=True, label="Video Retargeting Options"):
                    flag_do_crop_input_retargeting_video = gr.Checkbox(
                        value=True, label="Crop source"
                    )
                    video_retargeting_source_scale = gr.Number(
                        value=2.3,
                        label="Crop scale",
                        minimum=1.8,
                        maximum=3.2,
                        step=0.05,
                    )
                    video_lip_retargeting_slider = gr.Slider(
                        minimum=0,
                        maximum=0.8,
                        step=0.01,
                        label="Target lip-open ratio",
                    )
                    driving_smooth_observation_variance_retargeting = gr.Number(
                        value=3e-6,
                        label="Motion smooth strength",
                        minimum=1e-11,
                        maximum=1e-2,
                        step=1e-8,
                    )
                    video_retargeting_silence = gr.Checkbox(
                        value=False, label="Keep lip silent"
                    )

            with gr.Column():
                with gr.Row():
                    video_retargeting_output = gr.Video(
                        autoplay=False, label="Retargeting Result"
                    )
                    video_retargeting_output_paste_back = gr.Video(
                        autoplay=False, label="Paste-back Result"
                    )

        with gr.Row():
            process_video_retargeting_button = gr.Button(
                "🚗 Retarget Video", variant="primary"
            )
            clear_video_retargeting_button = gr.Button("🧹 Clear")

        # Return all components that need to be accessed from outside
        return {
            "retargeting_input_video": retargeting_input_video,
            "flag_do_crop_input_retargeting_video": flag_do_crop_input_retargeting_video,
            "video_retargeting_source_scale": video_retargeting_source_scale,
            "video_lip_retargeting_slider": video_lip_retargeting_slider,
            "driving_smooth_observation_variance_retargeting": driving_smooth_observation_variance_retargeting,
            "video_retargeting_silence": video_retargeting_silence,
            "video_retargeting_output": video_retargeting_output,
            "video_retargeting_output_paste_back": video_retargeting_output_paste_back,
            "process_video_retargeting_button": process_video_retargeting_button,
            "clear_video_retargeting_button": clear_video_retargeting_button,
        }
