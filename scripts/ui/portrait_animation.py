import gradio as gr
from modules import shared
from ..utils import add_source_image_buttons


def create_portrait_animation_tab(pipeline_handler):
    """Create the portrait animation tab UI."""
    with gr.Tab("Portrait Animation"):
        with gr.Row(equal_height=True):
            with gr.Column():
                # Source input tabs
                with gr.Tabs() as source_tabs:
                    with gr.Tab("Image", id="source_image_tab"):
                        source_image_input = gr.Image(
                            type="filepath", label="Source Image"
                        )
                        add_source_image_buttons(source_image_input)
                    with gr.Tab("Video", id="source_video_tab"):
                        source_video_input = gr.Video(label="Source Video")

                # Source cropping options
                with gr.Accordion(open=True, label="Cropping Options (Source)"):
                    flag_do_crop_input = gr.Checkbox(value=True, label="Crop (source)")
                    source_face_index = gr.Number(
                        value=0,
                        label="Face Index",
                        minimum=0,
                        maximum=999,
                        step=1,
                    )
                    scale = gr.Number(
                        value=2.3,
                        label="Crop Scale",
                        minimum=1.8,
                        maximum=3.2,
                        step=0.05,
                    )
                    vx_ratio = gr.Number(
                        value=0.0,
                        label="Crop X",
                        minimum=-0.5,
                        maximum=0.5,
                        step=0.01,
                    )
                    vy_ratio = gr.Number(
                        value=-0.125,
                        label="Crop Y",
                        minimum=-0.5,
                        maximum=0.5,
                        step=0.01,
                    )

                # Driving cropping options
                with gr.Accordion(open=True, label="Cropping Options (Driving)"):
                    flag_crop_driving_video_input = gr.Checkbox(
                        value=False, label="Crop (driving)"
                    )
                    driving_face_index = gr.Number(
                        value=0,
                        label="Face Index",
                        minimum=0,
                        maximum=999,
                        step=1,
                    )
                    scale_crop_driving_video = gr.Number(
                        value=2.2,
                        label="Crop Scale",
                        minimum=1.8,
                        maximum=3.2,
                        step=0.05,
                    )
                    vx_ratio_crop_driving_video = gr.Number(
                        value=0.0,
                        label="Crop X",
                        minimum=-0.5,
                        maximum=0.5,
                        step=0.01,
                    )
                    vy_ratio_crop_driving_video = gr.Number(
                        value=-0.1,
                        label="Crop Y",
                        minimum=-0.5,
                        maximum=0.5,
                        step=0.01,
                    )

            with gr.Column():
                # Animation options
                with gr.Accordion(open=True, label="Animation Options"):
                    flag_normalize_lip = gr.Checkbox(value=False, label="normalize lip")
                    flag_relative_input = gr.Checkbox(
                        value=True, label="relative motion"
                    )
                    flag_remap_input = gr.Checkbox(value=True, label="paste-back")
                    flag_stitching_input = gr.Checkbox(value=True, label="stitching")
                    animation_region = gr.Radio(
                        ["exp", "pose", "lip", "eyes", "all"],
                        value="all",
                        label="animation region",
                    )
                    driving_option_input = gr.Radio(
                        ["expression-friendly", "pose-friendly"],
                        value="expression-friendly",
                        label="driving option (i2v)",
                    )
                    driving_multiplier = gr.Number(
                        value=1.0,
                        label="driving multiplier (i2v)",
                        minimum=0.0,
                        maximum=2.0,
                        step=0.02,
                    )
                    driving_smooth_observation_variance = gr.Number(
                        value=3e-7,
                        label="motion smooth strength (v2v)",
                        minimum=1e-11,
                        maximum=1e-2,
                        step=1e-8,
                    )
                    flag_eye_retargeting = gr.Checkbox(
                        value=False, label="Eye retargeting"
                    )
                    flag_lip_retargeting = gr.Checkbox(
                        value=False, label="Lip retargeting"
                    )
                    flag_source_video_eye_retargeting = gr.Checkbox(
                        value=False, label="Source video eye retargeting"
                    )

        with gr.Row():
            process_button_animation = gr.Button("🚀 Animate", variant="primary")
            process_button_reset = gr.Button("🧹 Clear")

        # Define driving inputs
        with gr.Tabs() as driving_tabs:
            with gr.Tab("Video", id="driving_video_tab"):
                driving_video_input = gr.Video(label="Driving Video")
            with gr.Tab("Image", id="driving_image_tab"):
                driving_image_input = gr.Image(type="filepath", label="Driving Image")
            with gr.Tab("Webcam", id="driving_webcam_tab"):
                driving_image_webcam_input = gr.Image(
                    source="webcam", type="filepath", label="Webcam"
                )
            with gr.Tab("Pickle", id="driving_pickle_tab"):
                driving_video_pickle_input = gr.File(label="Driving Pickle")

        with gr.Row():
            output_video_i2v = gr.Video(autoplay=False, label="Output Video")
            output_video_concat_i2v = gr.Video(
                autoplay=False, label="Output Video with Paste-back"
            )

        # Return all components that need to be accessed from outside
        return {
            "source_tabs": source_tabs,
            "driving_tabs": driving_tabs,
            "source_image_input": source_image_input,
            "source_video_input": source_video_input,
            "driving_video_input": driving_video_input,
            "driving_image_input": driving_image_input,
            "driving_image_webcam_input": driving_image_webcam_input,
            "driving_video_pickle_input": driving_video_pickle_input,
            "flag_normalize_lip": flag_normalize_lip,
            "flag_relative_input": flag_relative_input,
            "flag_do_crop_input": flag_do_crop_input,
            "flag_remap_input": flag_remap_input,
            "flag_stitching_input": flag_stitching_input,
            "animation_region": animation_region,
            "driving_option_input": driving_option_input,
            "driving_multiplier": driving_multiplier,
            "flag_crop_driving_video_input": flag_crop_driving_video_input,
            "source_face_index": source_face_index,
            "scale": scale,
            "vx_ratio": vx_ratio,
            "vy_ratio": vy_ratio,
            "driving_face_index": driving_face_index,
            "scale_crop_driving_video": scale_crop_driving_video,
            "vx_ratio_crop_driving_video": vx_ratio_crop_driving_video,
            "vy_ratio_crop_driving_video": vy_ratio_crop_driving_video,
            "driving_smooth_observation_variance": driving_smooth_observation_variance,
            "flag_eye_retargeting": flag_eye_retargeting,
            "flag_lip_retargeting": flag_lip_retargeting,
            "flag_source_video_eye_retargeting": flag_source_video_eye_retargeting,
            "process_button_animation": process_button_animation,
            "process_button_reset": process_button_reset,
            "output_video_i2v": output_video_i2v,
            "output_video_concat_i2v": output_video_concat_i2v,
        }
