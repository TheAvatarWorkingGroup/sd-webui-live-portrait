import datetime
import os
import os.path as osp
from pathlib import Path
from typing import cast, Literal
from packaging.version import parse

import gradio as gr
import gradio.components
import modules.scripts as scripts
from modules import (
    devices,
    restart,
    script_callbacks,
    shared,
    generation_parameters_copypaste,
)
from modules.paths_internal import data_path

from liveportrait.utils.helper import load_description
from liveportrait.config.argument_config import ArgumentConfig
from liveportrait.config.crop_config import CropConfig
from liveportrait.config.inference_config import InferenceConfig
from liveportrait.gradio_pipeline import GradioPipeline, GradioPipelineAnimal

from internal_liveportrait.utils import (
    download_insightface_models,
    download_liveportrait_animals_models,
    download_liveportrait_models,
    download_liveportrait_landmark_model,
    IS_MACOS,
    has_xpose_lib,
    del_xpose_lib_dir,
)

repo_root = Path(__file__).parent.parent

gradio_pipeline: GradioPipeline | None = None
gradio_pipeline_animal: GradioPipelineAnimal | None = None

gradio_version = parse(gr.__version__)

if gradio_version.major > 3:
    try:

        def save_pil_to_file_patched(*args, **kwargs):
            from modules import ui_tempdir

            kwargs = {k: v for k, v in kwargs.items() if k != "name"}
            return ui_tempdir.save_pil_to_file(*args, **kwargs)

        gradio.processing_utils.save_pil_to_cache = save_pil_to_file_patched
    except Exception:
        pass


class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "Live Portrait"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return ()


def create_send_image_button():
    return gr.Button("Send to Live Portrait", variant="secondary")


def get_image_from_tab(img):
    if img is None:
        return None
    if isinstance(img, list) and len(img) > 0:
        # Handle gallery images (txt2img/img2img output)
        return img[0] if isinstance(img[0], dict) else img[0][0]
    elif isinstance(img, dict):
        # Handle single image (img2img input)
        return img.get("image", None)
    return img


def add_source_image_buttons(source_image_component):
    with gr.Row():
        img2img_source = gr.Button("Get from Image-to-Image")
        txt2img_source = gr.Button("Get from Text-to-Image")
        extras_source = gr.Button("Get from Extras")

    def get_img2img_image():
        return get_image_from_tab(getattr(shared, "img2img_image", None))

    def get_txt2img_image():
        return get_image_from_tab(getattr(shared, "txt2img_gallery", None))

    def get_extras_image():
        return get_image_from_tab(getattr(shared, "extras_image", None))

    # Get the current image from img2img tab
    img2img_source.click(
        fn=get_img2img_image,
        inputs=[],
        outputs=[source_image_component],
    )

    # Get the current image from txt2img gallery
    txt2img_source.click(
        fn=get_txt2img_image,
        inputs=[],
        outputs=[source_image_component],
    )

    # Get the current image from extras tab
    extras_source.click(
        fn=get_extras_image,
        inputs=[],
        outputs=[source_image_component],
    )


def on_ui_tabs():
    if shared.cmd_opts.nowebui:
        return

    def clear_model_cache():
        global gradio_pipeline, gradio_pipeline_animal
        gradio_pipeline = None
        gradio_pipeline_animal = None
        devices.torch_gc()

    def get_crop_config():
        # Get settings from shared options but don't use them as model parameters
        # since they're not part of CropConfig
        device = str(
            shared.opts.data.get("live_portrait_face_alignment_detector_device", "cuda")
        ).lower()

        return CropConfig(
            device_id=0 if device == "cuda" else -1,
            flag_force_cpu=(device == "cpu"),
            det_thresh=0.1,  # Default detection threshold
            dsize=512,  # Default crop size
            scale=2.3,  # Default scale value
            vx_ratio=0.0,  # Default x ratio
            vy_ratio=-0.125,  # Default y ratio
            max_face_num=0,  # No limit
            flag_do_rot=True,  # Enable rotation
            scale_crop_driving_video=2.2,
            vx_ratio_crop_driving_video=0.0,
            vy_ratio_crop_driving_video=-0.1,
            direction="large-small",
        )

    def get_inference_config():
        # Disable torch compilation by default to avoid CUDA graph issues
        device = str(
            shared.opts.data.get("live_portrait_face_alignment_detector_device", "cuda")
        ).lower()

        return InferenceConfig(
            flag_do_torch_compile=False,  # Forcibly disable compilation
            flag_use_half_precision=True,
            flag_crop_driving_video=False,
            device_id=0 if device == "cuda" else -1,
            flag_normalize_lip=False,
            flag_source_video_eye_retargeting=False,
            flag_eye_retargeting=False,
            flag_lip_retargeting=False,
            flag_stitching=True,
            flag_relative_motion=True,  # This was previously flag_relative_input
            flag_pasteback=True,  # This was previously flag_remap_input
            flag_do_crop=True,
            flag_do_rot=True,
            flag_force_cpu=(device == "cpu"),
            driving_option="expression-friendly",
            driving_multiplier=1.0,
            driving_smooth_observation_variance=3e-7,
            animation_region="all",
            # Experimental Global Dynamics Parameters
            flag_camera_shake=False,
            camera_shake_amplitude=0.005,
            flag_inertial_lag=False,
            inertial_lag_strength=0.3,
            flag_global_transform=False,
            global_transform_scale=0.2,
            # Additional required parameters with default values
            source_max_dim=1280,
            source_division=2,
        )

    def get_argument_config():
        default_output_dir = osp.join(
            data_path, "outputs", "live-portrait", f"{datetime.date.today()}"
        )
        config_output_dir = shared.opts.data.get("live_portrait_output_dir", "")
        device = str(
            shared.opts.data.get("live_portrait_face_alignment_detector_device", "cuda")
        ).lower()

        return ArgumentConfig(
            output_dir=(config_output_dir or default_output_dir),
            flag_use_half_precision=True,
            flag_crop_driving_video=False,
            device_id=0 if device == "cuda" else -1,
            flag_force_cpu=(device == "cpu"),
            flag_normalize_lip=False,
            flag_source_video_eye_retargeting=False,
            flag_eye_retargeting=False,
            flag_lip_retargeting=False,
            flag_stitching=True,
            flag_relative_motion=True,
            flag_pasteback=True,
            flag_do_crop=True,
            flag_do_rot=True,
            driving_option="expression-friendly",
            driving_multiplier=1.0,
            driving_smooth_observation_variance=3e-7,
            animation_region="all",
            det_thresh=0.15,
            scale=2.3,
            vx_ratio=0.0,
            vy_ratio=-0.125,
            source_max_dim=1280,
            source_division=2,
            scale_crop_driving_video=2.2,
            vx_ratio_crop_driving_video=0.0,
            vy_ratio_crop_driving_video=-0.1,
            server_port=8890,
            share=False,
            server_name="127.0.0.1",
            flag_do_torch_compile=False,
            gradio_temp_dir=None,
        )

    def init_gradio_pipeline():
        global gradio_pipeline, gradio_pipeline_animal
        inference_cfg = get_inference_config()
        crop_cfg = get_crop_config()
        argument_cfg = get_argument_config()
        if not gradio_pipeline:
            clear_model_cache()
            download_liveportrait_models()
            download_insightface_models()
            gradio_pipeline = GradioPipeline(
                inference_cfg=inference_cfg, crop_cfg=crop_cfg, args=argument_cfg
            )
        else:
            gradio_pipeline.cropper.update_config(crop_cfg.__dict__)
            gradio_pipeline.live_portrait_wrapper.update_config(inference_cfg.__dict__)
            gradio_pipeline.args.output_dir = argument_cfg.output_dir
        return gradio_pipeline

    def init_gradio_pipeline_animal():
        global gradio_pipeline, gradio_pipeline_animal
        inference_cfg = get_inference_config()
        crop_cfg = CropConfig()
        argument_cfg = get_argument_config()
        if not gradio_pipeline_animal:
            clear_model_cache()
            download_liveportrait_landmark_model()
            download_liveportrait_animals_models()
            download_insightface_models()
            gradio_pipeline_animal = GradioPipelineAnimal(
                inference_cfg=inference_cfg, crop_cfg=crop_cfg, args=argument_cfg
            )
        else:
            gradio_pipeline_animal.live_portrait_wrapper_animal.update_config(
                inference_cfg.__dict__
            )
            gradio_pipeline_animal.args.output_dir = argument_cfg.output_dir
        return gradio_pipeline_animal

    def gpu_wrapped_init_retargeting_image(*args, **kwargs):
        pipeline = init_gradio_pipeline()
        # Extract the parameters from args or kwargs
        retargeting_source_scale = (
            args[0] if len(args) > 0 else kwargs.get("retargeting_source_scale", 2.5)
        )
        eye_ratio = (
            args[1] if len(args) > 1 else kwargs.get("eye_retargeting_slider", 0.0)
        )
        lip_ratio = (
            args[2] if len(args) > 2 else kwargs.get("lip_retargeting_slider", 0.0)
        )
        image_input = (
            args[3] if len(args) > 3 else kwargs.get("retargeting_input_image", None)
        )

        # Skip if no image is provided
        if image_input is None:
            return 0.0, 0.0

        # Handle the image path
        image_path = (
            image_input.name if hasattr(image_input, "name") else str(image_input)
        )
        if not os.path.exists(image_path):
            print(f"Warning: Image path does not exist: {image_path}")
            return 0.0, 0.0

        # Call init_retargeting_image with the correct parameter names
        src_eye_ratio, src_lip_ratio = pipeline.init_retargeting_image(
            scale=retargeting_source_scale,
            source_eye_ratio=eye_ratio,
            source_lip_ratio=lip_ratio,
            image=image_path,
        )
        return float(src_eye_ratio), float(src_lip_ratio)

    def gpu_wrapped_execute_image_retargeting(*args, **kwargs):
        pipeline = init_gradio_pipeline()
        # Extract and validate the image input
        image_input = kwargs.get("retargeting_input_image", None)
        if image_input is None:
            print("Warning: No image provided for retargeting")
            return None, None

        # Handle the image path
        if hasattr(image_input, "name"):
            kwargs["retargeting_input_image"] = image_input.name
        else:
            kwargs["retargeting_input_image"] = str(image_input)

        out, out_to_ori_blend = pipeline.execute_image_retargeting(*args, **kwargs)
        # Convert the output to the format expected by Gradio's Gallery
        if isinstance(out, dict) and "image" in out:
            out = out["image"]
        if isinstance(out_to_ori_blend, dict) and "image" in out_to_ori_blend:
            out_to_ori_blend = out_to_ori_blend["image"]
        return out, out_to_ori_blend

    def gpu_wrapped_execute_video(*args, **kwargs):
        return init_gradio_pipeline().execute_video(*args, **kwargs)

    def gpu_wrapped_execute_video_retargeting(*args, **kwargs):
        return init_gradio_pipeline().execute_video_retargeting(*args, **kwargs)

    def gpu_wrapped_execute_video_animal(*args, **kwargs):
        return init_gradio_pipeline_animal().execute_video(*args, **kwargs)

    def reset_sliders(*args, **kwargs):
        return (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.5,
            True,
            True,
        )

    def reinstall_xpose(*args, **kwargs):
        del_xpose_lib_dir()
        if restart.is_restartable:
            restart.restart_program()
        else:
            restart.stop_program()

    with gr.Blocks(analytics_enabled=False) as live_portrait:
        with gr.Tabs():
            with gr.Tab("Portrait Animation"):
                with gr.Row():
                    with gr.Column():
                        source_image_input = gr.Image(
                            type="filepath", label="Source Image"
                        )
                        add_source_image_buttons(source_image_input)
                        with gr.Accordion(open=True, label="Cropping Options (Source)"):
                            flag_do_crop_input = gr.Checkbox(
                                value=True, label="Crop (source)"
                            )
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

                    with gr.Column():
                        with gr.Tabs():
                            with gr.TabItem("Driving Video"):
                                driving_video_input = gr.Video()
                            with gr.TabItem(
                                "Driving Video (Webcam)",
                                visible=gradio_version.major < 4,
                            ):
                                driving_video_webcam_input = gr.Video(
                                    format="mp4", include_audio=True
                                )
                            with gr.TabItem("Driving Image"):
                                driving_image_input = gr.Image(type="filepath")
                            with gr.TabItem(
                                "Driving Image (Webcam)",
                                visible=gradio_version.major < 4,
                            ):
                                driving_image_webcam_input = gr.Image(type="filepath")
                            with gr.TabItem("Driving Pickle"):
                                driving_video_pickle_input = gr.File(
                                    type=(
                                        "filepath"
                                        if gradio_version.major >= 4
                                        else "file"
                                    ),
                                    file_types=[".pkl"],
                                )

                with gr.Row():
                    with gr.Accordion(open=True, label="Animation Options"):
                        flag_normalize_lip = gr.Checkbox(
                            value=False, label="normalize lip"
                        )
                        flag_relative_input = gr.Checkbox(
                            value=True, label="relative motion"
                        )
                        flag_remap_input = gr.Checkbox(value=True, label="paste-back")
                        flag_stitching_input = gr.Checkbox(
                            value=True, label="stitching"
                        )
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

                with gr.Row():
                    process_button_animation = gr.Button(
                        "🚀 Animate", variant="primary"
                    )
                    process_button_reset = gr.Button("🧹 Clear")

                with gr.Row():
                    output_video_i2v = gr.Video(autoplay=False, label="Output Video")
                    output_video_concat_i2v = gr.Video(
                        autoplay=False, label="Output Video with Paste-back"
                    )

            with gr.Tab("Image Retargeting"):
                with gr.Row():
                    with gr.Column():
                        retargeting_input_image = gr.Image(
                            type="filepath", label="Source Image"
                        )
                        add_source_image_buttons(retargeting_input_image)

                        with gr.Row():
                            flag_do_crop_input_retargeting_image = gr.Checkbox(
                                value=True, label="do crop (source)"
                            )
                            flag_stitching_retargeting_input = gr.Checkbox(
                                value=True, label="stitching"
                            )
                            face_index = gr.Number(
                                value=0,
                                label="face index",
                                minimum=0,
                                maximum=999,
                                step=1,
                            )
                            retargeting_source_scale = gr.Number(
                                minimum=1.8,
                                maximum=3.2,
                                value=2.5,
                                step=0.05,
                                label="crop scale",
                            )
                            eye_retargeting_slider = gr.Slider(
                                minimum=0,
                                maximum=0.8,
                                step=0.01,
                                label="target eye-open ratio",
                                visible=False,
                            )
                            lip_retargeting_slider = gr.Slider(
                                minimum=0,
                                maximum=0.8,
                                step=0.01,
                                label="target lip-open ratio",
                                visible=False,
                            )

                with gr.Row():
                    with gr.Column():
                        with gr.Accordion(open=True, label="Facial movement sliders"):
                            head_pitch_slider = gr.Slider(
                                minimum=-15.0,
                                maximum=15.0,
                                value=0,
                                step=1,
                                label="relative pitch",
                            )
                            head_yaw_slider = gr.Slider(
                                minimum=-25,
                                maximum=25,
                                value=0,
                                step=1,
                                label="relative yaw",
                            )
                            head_roll_slider = gr.Slider(
                                minimum=-15.0,
                                maximum=15.0,
                                value=0,
                                step=1,
                                label="relative roll",
                            )
                            mov_x = gr.Slider(
                                minimum=-0.19,
                                maximum=0.19,
                                value=0.0,
                                step=0.01,
                                label="x-axis movement",
                            )
                            mov_y = gr.Slider(
                                minimum=-0.19,
                                maximum=0.19,
                                value=0.0,
                                step=0.01,
                                label="y-axis movement",
                            )
                            mov_z = gr.Slider(
                                minimum=0.9,
                                maximum=1.2,
                                value=1.0,
                                step=0.01,
                                label="z-axis movement",
                            )

                    with gr.Column():
                        with gr.Accordion(open=True, label="Facial expression sliders"):
                            lip_variation_zero = gr.Slider(
                                minimum=-0.09,
                                maximum=0.09,
                                value=0,
                                step=0.01,
                                label="pouting",
                            )
                            lip_variation_one = gr.Slider(
                                minimum=-20.0,
                                maximum=15.0,
                                value=0,
                                step=0.01,
                                label="pursing 😐",
                            )
                            lip_variation_two = gr.Slider(
                                minimum=0.0,
                                maximum=15.0,
                                value=0,
                                step=0.01,
                                label="grin 😁",
                            )
                            lip_variation_three = gr.Slider(
                                minimum=-90.0,
                                maximum=120.0,
                                value=0,
                                step=1.0,
                                label="lip close <-> open",
                            )
                            smile = gr.Slider(
                                minimum=-0.3,
                                maximum=1.3,
                                value=0,
                                step=0.01,
                                label="smile 😄",
                            )
                            wink = gr.Slider(
                                minimum=0,
                                maximum=39,
                                value=0,
                                step=0.01,
                                label="wink 😉",
                            )
                            eyebrow = gr.Slider(
                                minimum=-30,
                                maximum=30,
                                value=0,
                                step=0.01,
                                label="eyebrow 🤨",
                            )
                            eyeball_direction_x = gr.Slider(
                                minimum=-30.0,
                                maximum=30.0,
                                value=0,
                                step=0.01,
                                label="eye gaze (horizontal) 👀",
                            )
                            eyeball_direction_y = gr.Slider(
                                minimum=-63.0,
                                maximum=63.0,
                                value=0,
                                step=0.01,
                                label="eye gaze (vertical) 🙄",
                            )

                with gr.Row():
                    reset_button = gr.Button("🔄 Reset")
                    process_button_reset_retargeting = gr.Button("🧹 Clear")

                with gr.Row():
                    retargeting_output_image = gr.Gallery(
                        preview=True,
                        selected_index=0,
                        object_fit="contain",
                        label="Output",
                    )
                    retargeting_output_image_paste_back = gr.Gallery(
                        preview=True,
                        selected_index=0,
                        object_fit="contain",
                        height=512,
                        label="Output with Paste-back",
                    )

                # Add buttons to send images to other tabs
                with gr.Row():
                    with gr.Column():
                        send_to_extras = gr.Button(
                            "Send to Extras", variant="secondary"
                        )
                        send_to_img2img = gr.Button(
                            "Send to Image-to-Image", variant="secondary"
                        )
                    with gr.Column():
                        send_pasteback_to_extras = gr.Button(
                            "Send Paste-back to Extras", variant="secondary"
                        )
                        send_pasteback_to_img2img = gr.Button(
                            "Send Paste-back to Image-to-Image", variant="secondary"
                        )

            with gr.Tab("Video Retargeting"):
                with gr.Row():
                    with gr.Column():
                        retargeting_input_video = gr.Video(label="Source Video")
                        with gr.Row():
                            flag_do_crop_input_retargeting_video = gr.Checkbox(
                                value=True, label="do crop (source)"
                            )
                            video_face_index = gr.Number(
                                value=0,
                                label="face index",
                                minimum=0,
                                maximum=999,
                                step=1,
                            )
                            video_retargeting_source_scale = gr.Number(
                                minimum=1.8,
                                maximum=3.2,
                                value=2.3,
                                step=0.05,
                                label="crop scale",
                            )
                            video_lip_retargeting_slider = gr.Slider(
                                minimum=0,
                                maximum=0.8,
                                step=0.01,
                                label="target lip-open ratio",
                            )
                            driving_smooth_observation_variance_retargeting = gr.Number(
                                value=3e-7,
                                label="motion smooth strength (v2v)",
                                minimum=1e-11,
                                maximum=1e-2,
                                step=1e-8,
                            )
                            video_retargeting_silence = gr.Checkbox(
                                value=False, label="keeping the lip silent"
                            )

                with gr.Row():
                    process_button_retargeting_video = gr.Button(
                        "🍄 Retargeting Video", variant="primary"
                    )
                    process_button_reset_retargeting = gr.Button("🧹 Clear")

                with gr.Row():
                    output_video = gr.Video(autoplay=False, label="Output Video")
                    output_video_paste_back = gr.Video(
                        autoplay=False, label="Output Video with Paste-back"
                    )

            if not IS_MACOS:
                with gr.Tab("Animals"):
                    if not has_xpose_lib():
                        gr.Markdown(
                            "The XPose model, required to generate animal videos, is not installed or could not be installed correctly. Try to reinstall it by following instructions in this extension's README."
                        )
                        reinstall_xpose_button = gr.Button(
                            "Reinstall XPose and Restart UI", variant="primary"
                        )
                        reinstall_xpose_button.click(
                            fn=reinstall_xpose,
                            _js="restart_reload",
                            inputs=[],
                            outputs=[],
                        )
                    else:
                        with gr.Row():
                            with gr.Column():
                                source_image_input = gr.Image(
                                    type="filepath", label="Source Animal Image"
                                )
                                add_source_image_buttons(source_image_input)

                                with gr.Accordion(open=True, label="Cropping Options"):
                                    flag_do_crop_input = gr.Checkbox(
                                        value=True, label="do crop (source)"
                                    )
                                    scale = gr.Number(
                                        value=2.3,
                                        label="source crop scale",
                                        minimum=1.8,
                                        maximum=3.2,
                                        step=0.05,
                                    )
                                    vx_ratio = gr.Number(
                                        value=0.0,
                                        label="source crop x",
                                        minimum=-0.5,
                                        maximum=0.5,
                                        step=0.01,
                                    )
                                    vy_ratio = gr.Number(
                                        value=-0.125,
                                        label="source crop y",
                                        minimum=-0.5,
                                        maximum=0.5,
                                        step=0.01,
                                    )

                            with gr.Column():
                                with gr.Tabs():
                                    with gr.TabItem("Driving Pickle"):
                                        driving_video_pickle_input = gr.File(
                                            type=(
                                                "filepath"
                                                if gradio_version.major >= 4
                                                else "file"
                                            ),
                                            file_types=[".pkl"],
                                        )
                                    with gr.TabItem("Driving Video"):
                                        driving_video_input = gr.Video()

                        with gr.Row():
                            with gr.Accordion(open=False, label="Animation Options"):
                                flag_stitching = gr.Checkbox(
                                    value=False, label="stitching (not recommended)"
                                )
                                flag_remap_input = gr.Checkbox(
                                    value=False, label="paste-back (not recommended)"
                                )
                                driving_multiplier = gr.Number(
                                    value=1.0,
                                    label="driving multiplier",
                                    minimum=0.0,
                                    maximum=2.0,
                                    step=0.02,
                                )

                        with gr.Row():
                            process_button_animation = gr.Button(
                                "🚀 Animate", variant="primary"
                            )
                            process_button_reset = gr.Button("🧹 Clear")

                        with gr.Row():
                            output_video_animal_i2v = gr.Video(
                                autoplay=False, label="Output Video"
                            )
                            output_video_animal_i2v_gif = gr.Image(
                                type="numpy", label="Output GIF"
                            )
                            output_video_animal_concat_i2v = gr.Video(
                                autoplay=False, label="Output Video with Paste-back"
                            )

        # Add missing variable definitions
        flag_crop_driving_video_input = gr.Checkbox(
            value=True, label="Crop (driving)", visible=False
        )
        driving_face_index = gr.Number(
            value=0,
            label="Driving Face Index",
            minimum=0,
            maximum=999,
            step=1,
            visible=False,
        )
        scale_crop_driving_video = gr.Number(
            value=2.3,
            label="Driving Crop Scale",
            minimum=1.8,
            maximum=3.2,
            step=0.05,
            visible=False,
        )
        vx_ratio_crop_driving_video = gr.Number(
            value=0.0,
            label="Driving Crop X",
            minimum=-0.5,
            maximum=0.5,
            step=0.01,
            visible=False,
        )
        vy_ratio_crop_driving_video = gr.Number(
            value=-0.125,
            label="Driving Crop Y",
            minimum=-0.5,
            maximum=0.5,
            step=0.01,
            visible=False,
        )

        # Add click handlers for sending images to other tabs
        def send_image_to_extras(gallery):
            if not gallery:
                return
            selected = gallery[0] if isinstance(gallery[0], dict) else gallery[0][0]
            setattr(shared, "extras_image", selected)

        def send_image_to_img2img(gallery):
            if not gallery:
                return
            selected = gallery[0] if isinstance(gallery[0], dict) else gallery[0][0]
            setattr(shared, "img2img_image", {"image": selected})

        send_to_extras.click(
            fn=send_image_to_extras,
            inputs=[retargeting_output_image],
            outputs=[],
        )
        send_to_img2img.click(
            fn=send_image_to_img2img,
            inputs=[retargeting_output_image],
            outputs=[],
        )
        send_pasteback_to_extras.click(
            fn=send_image_to_extras,
            inputs=[retargeting_output_image_paste_back],
            outputs=[],
        )
        send_pasteback_to_img2img.click(
            fn=send_image_to_img2img,
            inputs=[retargeting_output_image_paste_back],
            outputs=[],
        )

        # Wire up all the event handlers
        process_button_animation.click(
            fn=gpu_wrapped_execute_video,
            inputs=[
                source_image_input,
                driving_video_input,
                driving_video_webcam_input,
                driving_image_input,
                driving_image_webcam_input,
                driving_video_pickle_input,
                flag_normalize_lip,
                flag_relative_input,
                flag_do_crop_input,
                flag_remap_input,
                flag_stitching_input,
                animation_region,
                driving_option_input,
                driving_multiplier,
                flag_crop_driving_video_input,
                source_face_index,
                scale,
                vx_ratio,
                vy_ratio,
                driving_face_index,
                scale_crop_driving_video,
                vx_ratio_crop_driving_video,
                vy_ratio_crop_driving_video,
                driving_smooth_observation_variance,
            ],
            outputs=[output_video_i2v, output_video_concat_i2v],
            show_progress="full",
        )

        retargeting_input_image.change(
            fn=gpu_wrapped_init_retargeting_image,
            inputs=[
                retargeting_source_scale,
                eye_retargeting_slider,
                lip_retargeting_slider,
                retargeting_input_image,
            ],
            outputs=[eye_retargeting_slider, lip_retargeting_slider],
        )

        for slider in [
            head_pitch_slider,
            head_yaw_slider,
            head_roll_slider,
            mov_x,
            mov_y,
            mov_z,
            lip_variation_zero,
            lip_variation_one,
            lip_variation_two,
            lip_variation_three,
            smile,
            wink,
            eyebrow,
            eyeball_direction_x,
            eyeball_direction_y,
        ]:
            slider.change(
                fn=gpu_wrapped_execute_image_retargeting,
                inputs=[
                    eye_retargeting_slider,
                    lip_retargeting_slider,
                    head_pitch_slider,
                    head_yaw_slider,
                    head_roll_slider,
                    mov_x,
                    mov_y,
                    mov_z,
                    lip_variation_zero,
                    lip_variation_one,
                    lip_variation_two,
                    lip_variation_three,
                    smile,
                    wink,
                    eyebrow,
                    eyeball_direction_x,
                    eyeball_direction_y,
                    retargeting_input_image,
                    face_index,
                    retargeting_source_scale,
                    flag_stitching_retargeting_input,
                    flag_do_crop_input_retargeting_image,
                ],
                outputs=[retargeting_output_image, retargeting_output_image_paste_back],
            )

        process_button_retargeting_video.click(
            fn=gpu_wrapped_execute_video_retargeting,
            inputs=[
                video_lip_retargeting_slider,
                retargeting_input_video,
                video_face_index,
                video_retargeting_source_scale,
                driving_smooth_observation_variance_retargeting,
                video_retargeting_silence,
                flag_do_crop_input_retargeting_video,
            ],
            outputs=[output_video, output_video_paste_back],
            show_progress="full",
        )

        if not IS_MACOS and has_xpose_lib():
            process_button_animation.click(
                fn=gpu_wrapped_execute_video_animal,
                inputs=[
                    source_image_input,
                    driving_video_input,
                    driving_video_pickle_input,
                    flag_do_crop_input,
                    flag_remap_input,
                    driving_multiplier,
                    flag_stitching,
                    flag_crop_driving_video_input,
                    scale,
                    vx_ratio,
                    vy_ratio,
                    scale_crop_driving_video,
                    vx_ratio_crop_driving_video,
                    vy_ratio_crop_driving_video,
                ],
                outputs=[
                    output_video_animal_i2v,
                    output_video_animal_concat_i2v,
                    output_video_animal_i2v_gif,
                ],
                show_progress="full",
            )

    return [(live_portrait, "Live Portrait", "live_portrait")]


def on_ui_settings():
    section = ("live_portrait", "Live Portrait")
    shared.opts.add_option(
        "live_portrait_human_face_detector",
        shared.OptionInfo(
            default="InsightFace",
            label="Human face detector",
            component=gr.Radio,
            component_args={"choices": ["InsightFace", "MediaPipe", "FaceAlignment"]},
            section=section,
        ),
    )
    shared.opts.add_option(
        "live_portrait_face_alignment_detector",
        shared.OptionInfo(
            default="BlazeFace Back Camera",
            label="Face alignment detector",
            component=gr.Radio,
            component_args={
                "choices": ["BlazeFace", "BlazeFace Back Camera", "RetinaFace", "SFD"]
            },
            section=section,
        ),
    )
    shared.opts.add_option(
        "live_portrait_face_alignment_detector_device",
        shared.OptionInfo(
            default="CUDA",
            label="Face alignment detector device",
            component=gr.Radio,
            component_args={"choices": ["CUDA", "CPU", "MPS"]},
            section=section,
        ),
    )
    shared.opts.add_option(
        "live_portrait_face_alignment_detector_dtype",
        shared.OptionInfo(
            default="fp16",
            label="Face alignment detector dtype",
            component=gr.Radio,
            component_args={"choices": ["fp16", "bf16", "fp32"]},
            section=section,
        ),
    )
    shared.opts.add_option(
        "live_portrait_flag_do_torch_compile",
        shared.OptionInfo(
            False, "Enable torch.compile for faster inference", section=section
        ),
    )
    shared.opts.add_option(
        "live_portrait_output_dir",
        shared.OptionInfo(
            "", "Live portrait generation output directory", section=section
        ),
    )
    shared.opts.add_option(
        "img2img_editor_height",
        shared.OptionInfo(
            default=720,
            label="Image editor height",
            component=gr.Slider,
            component_args={"minimum": 80, "maximum": 2160, "step": 1},
            section=section,
        ),
    )
    shared.opts.add_option(
        "img2img_inpaint_mask_brush_color",
        shared.OptionInfo(
            default="#ffffff",
            label="Inpaint mask brush color",
            component=gr.ColorPicker,
            section=section,
        ),
    )


try:
    script_callbacks.on_ui_tabs(on_ui_tabs)
    script_callbacks.on_ui_settings(on_ui_settings)
except:
    print("Live Portrait UI failed to initialize")
