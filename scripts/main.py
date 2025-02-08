import datetime
import os.path as osp
from pathlib import Path
from packaging.version import parse

import gradio as gr
import gradio.components
import modules.scripts as scripts
from modules import (
    devices,
    restart,
    script_callbacks,
    shared,
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

from scripts.ui import (
    create_portrait_animation_tab,
    create_portrait_retargeting_tab,
    create_video_retargeting_tab,
)

repo_root = Path(__file__).parent.parent
gradio_pipeline: GradioPipeline | None = None
gradio_pipeline_animal: GradioPipelineAnimal | None = None

# Patch Gradio's file saving for newer versions
if parse(gr.__version__).major > 3:
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


def get_image_from_tab(img):
    """Helper function to extract image from various tab formats."""
    if img is None:
        return None
    if isinstance(img, list) and len(img) > 0:
        return img[0] if isinstance(img[0], dict) else img[0][0]
    elif isinstance(img, dict):
        return img.get("image", None)
    return img


def add_source_image_buttons(source_image_component):
    """Add buttons to get images from other tabs."""
    with gr.Row():
        img2img_source = gr.Button("Get from Image-to-Image")
        txt2img_source = gr.Button("Get from Text-to-Image")
        extras_source = gr.Button("Get from Extras")

    img2img_source.click(
        fn=lambda: get_image_from_tab(getattr(shared, "img2img_image", None)),
        inputs=[],
        outputs=[source_image_component],
    )
    txt2img_source.click(
        fn=lambda: get_image_from_tab(getattr(shared, "txt2img_gallery", None)),
        inputs=[],
        outputs=[source_image_component],
    )
    extras_source.click(
        fn=lambda: get_image_from_tab(getattr(shared, "extras_image", None)),
        inputs=[],
        outputs=[source_image_component],
    )


def clear_model_cache():
    """Clear GPU memory and reset pipelines."""
    global gradio_pipeline, gradio_pipeline_animal
    gradio_pipeline = None
    gradio_pipeline_animal = None
    devices.torch_gc()


def reinstall_xpose(*args, **kwargs):
    """Reinstall xpose library and restart if needed."""
    del_xpose_lib_dir()
    if restart.is_restartable:
        restart.restart_program()
    else:
        restart.stop_program()


def create_send_image_button():
    return gr.Button("Send to Live Portrait", variant="secondary")


def on_ui_tabs():
    if shared.cmd_opts.nowebui:
        return

    def get_crop_config():
        # Get settings from shared options but don't use them as model parameters
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

    def gpu_wrapped_execute_image_retargeting(*args, **kwargs):
        pipeline = init_gradio_pipeline()
        # Extract and validate the image input - it's the 18th argument (0-based index 17)
        image_input = args[17] if len(args) > 17 else None
        if image_input is None:
            print("Warning: No image provided for retargeting")
            return (
                None,  # retargeting_output
                None,  # retargeting_output_paste_back
                gr.update(visible=False),  # retargeting_output_video
                gr.update(visible=False),  # retargeting_output_video_paste_back
            )

        # Handle the image path
        if hasattr(image_input, "name"):
            args = list(args)
            args[17] = image_input.name
        else:
            args = list(args)
            args[17] = str(image_input)

        try:
            result = pipeline.execute_image_retargeting(*args, **kwargs)
            if len(result) == 2:  # Image output
                out, out_paste_back = result
                # Ensure we're returning single images, not lists
                if isinstance(out, list):
                    out = out[0] if out else None
                if isinstance(out_paste_back, list):
                    out_paste_back = out_paste_back[0] if out_paste_back else None

                # Handle dict outputs (e.g. {"image": img})
                if isinstance(out, dict) and "image" in out:
                    out = out["image"]
                if isinstance(out_paste_back, dict) and "image" in out_paste_back:
                    out_paste_back = out_paste_back["image"]

                return (
                    out,  # Single image for retargeting output
                    out_paste_back,  # Single image for paste-back output
                    gr.update(visible=False),  # Hide video output
                    gr.update(visible=False),  # Hide video paste-back output
                )
            else:  # Video output
                out_video, out_video_paste_back = result
                return (
                    gr.update(visible=False),  # Hide image output
                    gr.update(visible=False),  # Hide image paste-back output
                    out_video,  # Video output
                    out_video_paste_back,  # Video paste-back output
                )
        except Exception as e:
            print(f"Error in execute_image_retargeting: {str(e)}")
            return None, None, gr.update(visible=False), gr.update(visible=False)

    def gpu_wrapped_execute_video(*args, **kwargs):
        pipeline = init_gradio_pipeline()

        # Extract source inputs and handle potential dict inputs
        source_image = args[0] if len(args) > 0 else None
        source_video = args[1] if len(args) > 1 else None

        # Extract driving inputs
        driving_video = args[2] if len(args) > 2 else None
        driving_image = args[3] if len(args) > 3 else None
        driving_image_webcam = args[4] if len(args) > 4 else None
        driving_pickle = args[5] if len(args) > 5 else None

        # Handle various input types
        def process_input(inp):
            if isinstance(inp, (list, tuple)):
                inp = inp[0] if inp else None
            if isinstance(inp, dict):
                if "image" in inp:
                    return inp["image"]
                if "video" in inp:
                    return inp["video"]
                if "name" in inp:
                    return inp["name"]
            return inp

        # Process all inputs
        source_image = process_input(source_image)
        source_video = process_input(source_video)
        driving_video = process_input(driving_video)
        driving_image = process_input(driving_image)
        driving_image_webcam = process_input(driving_image_webcam)
        driving_pickle = process_input(driving_pickle)

        # Extract tab states from the end of args
        source_tab = args[-2] if len(args) > len(args) - 2 else "source_image_tab"
        driving_tab = args[-1] if len(args) > len(args) - 1 else "driving_video_tab"

        # Convert file inputs to paths
        if hasattr(source_image, "name"):
            source_image = source_image.name
        if hasattr(source_video, "name"):
            source_video = source_video.name
        if hasattr(driving_video, "name"):
            driving_video = driving_video.name
        if hasattr(driving_image, "name"):
            driving_image = driving_image.name
        if hasattr(driving_image_webcam, "name"):
            driving_image_webcam = driving_image_webcam.name
        if hasattr(driving_pickle, "name"):
            driving_pickle = driving_pickle.name

        # Create new args list without tab states
        new_args = [
            source_image,
            source_video,
            driving_video,
            driving_image,
            driving_image_webcam,
            driving_pickle,
        ]

        # Map tab IDs to the expected values
        kwargs["tab_selection"] = "Image" if "image" in source_tab.lower() else "Video"
        kwargs["v_tab_selection"] = (
            "Video"
            if "video" in driving_tab.lower()
            else (
                "Image"
                if "image" in driving_tab.lower()
                else ("Webcam" if "webcam" in driving_tab.lower() else "Pickle")
            )
        )

        # Debug print
        print(f"Source tab: {kwargs['tab_selection']}")
        print(f"Driving tab: {kwargs['v_tab_selection']}")
        print(f"Source image: {source_image}")
        print(f"Source video: {source_video}")
        print(f"Driving video: {driving_video}")
        print(f"Driving image: {driving_image}")
        print(f"Driving webcam: {driving_image_webcam}")
        print(f"Driving pickle: {driving_pickle}")

        return pipeline.execute_video(*new_args, **kwargs)

    def gpu_wrapped_execute_video_retargeting(*args, **kwargs):
        pipeline = init_gradio_pipeline()
        try:
            # Extract and process video input
            video_input = args[1] if len(args) > 1 else None
            if video_input is None:
                print("Warning: No video provided for retargeting")
                return None, None

            # Handle file path
            if hasattr(video_input, "name"):
                args = list(args)
                args[1] = video_input.name

            return pipeline.execute_video_retargeting(*args, **kwargs)
        except Exception as e:
            print(f"Error in execute_video_retargeting: {str(e)}")
            return None, None

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

    with gr.Blocks(analytics_enabled=False) as live_portrait:
        # Add tab state tracking
        source_tab_state = gr.State(value="Image")  # Default to Image tab
        driving_tab_state = gr.State(value="Video")  # Default to Video tab

        # Now build the UI layout
        with gr.Tabs():
            # Create the main tabs
            animation_components = create_portrait_animation_tab(init_gradio_pipeline)
            retargeting_components = create_portrait_retargeting_tab(
                init_gradio_pipeline
            )
            video_retargeting_components = create_video_retargeting_tab(
                init_gradio_pipeline
            )

            # Wire up the animation tab handlers
            animation_components["source_tabs"].select(
                fn=lambda evt: (
                    "Image"
                    if evt is None or not hasattr(evt, "selected")
                    else (
                        evt.selected.id
                        if hasattr(evt.selected, "id")
                        else "source_image_tab"
                    )
                ),
                inputs=None,
                outputs=source_tab_state,
            )

            animation_components["driving_tabs"].select(
                fn=lambda evt: (
                    "Video"
                    if evt is None or not hasattr(evt, "selected")
                    else (
                        evt.selected.id
                        if hasattr(evt.selected, "id")
                        else "driving_video_tab"
                    )
                ),
                inputs=None,
                outputs=driving_tab_state,
            )

            # Wire up the animation button handlers
            animation_components["process_button_animation"].click(
                fn=gpu_wrapped_execute_video,
                inputs=[
                    animation_components["source_image_input"],
                    animation_components["source_video_input"],
                    animation_components["driving_video_input"],
                    animation_components["driving_image_input"],
                    animation_components["driving_image_webcam_input"],
                    animation_components["driving_video_pickle_input"],
                    animation_components["flag_normalize_lip"],
                    animation_components["flag_relative_input"],
                    animation_components["flag_do_crop_input"],
                    animation_components["flag_remap_input"],
                    animation_components["flag_stitching_input"],
                    animation_components["animation_region"],
                    animation_components["driving_option_input"],
                    animation_components["driving_multiplier"],
                    animation_components["flag_crop_driving_video_input"],
                    animation_components["source_face_index"],
                    animation_components["scale"],
                    animation_components["vx_ratio"],
                    animation_components["vy_ratio"],
                    animation_components["driving_face_index"],
                    animation_components["scale_crop_driving_video"],
                    animation_components["vx_ratio_crop_driving_video"],
                    animation_components["vy_ratio_crop_driving_video"],
                    animation_components["driving_smooth_observation_variance"],
                    source_tab_state,
                    driving_tab_state,
                ],
                outputs=[
                    animation_components["output_video_i2v"],
                    animation_components["output_video_concat_i2v"],
                ],
                show_progress="full",
            )

            # Wire up the retargeting handlers
            retargeting_components["retargeting_input_image"].change(
                fn=init_gradio_pipeline().init_retargeting_image,
                inputs=[
                    retargeting_components["retargeting_source_scale"],
                    retargeting_components["eye_retargeting_slider"],
                    retargeting_components["lip_retargeting_slider"],
                    retargeting_components["retargeting_input_image"],
                ],
                outputs=[
                    retargeting_components["eye_retargeting_slider"],
                    retargeting_components["lip_retargeting_slider"],
                ],
            )

            # Add handlers for all retargeting sliders
            retargeting_sliders = [
                retargeting_components["eye_retargeting_slider"],
                retargeting_components["lip_retargeting_slider"],
                retargeting_components["head_pitch"],
                retargeting_components["head_yaw"],
                retargeting_components["head_roll"],
                retargeting_components["mov_x"],
                retargeting_components["mov_y"],
                retargeting_components["mov_z"],
                retargeting_components["lip_variation_zero"],
                retargeting_components["lip_variation_one"],
                retargeting_components["lip_variation_two"],
                retargeting_components["lip_variation_three"],
                retargeting_components["smile"],
                retargeting_components["wink"],
                retargeting_components["eyebrow"],
                retargeting_components["eyeball_x"],
                retargeting_components["eyeball_y"],
            ]

            for slider in retargeting_sliders:
                slider.change(
                    fn=gpu_wrapped_execute_image_retargeting,
                    inputs=[
                        retargeting_components["eye_retargeting_slider"],
                        retargeting_components["lip_retargeting_slider"],
                        retargeting_components["head_pitch"],
                        retargeting_components["head_yaw"],
                        retargeting_components["head_roll"],
                        retargeting_components["mov_x"],
                        retargeting_components["mov_y"],
                        retargeting_components["mov_z"],
                        retargeting_components["lip_variation_zero"],
                        retargeting_components["lip_variation_one"],
                        retargeting_components["lip_variation_two"],
                        retargeting_components["lip_variation_three"],
                        retargeting_components["smile"],
                        retargeting_components["wink"],
                        retargeting_components["eyebrow"],
                        retargeting_components["eyeball_x"],
                        retargeting_components["eyeball_y"],
                        retargeting_components["retargeting_input_image"],
                        retargeting_components["retargeting_source_scale"],
                        retargeting_components["flag_stitching_retargeting"],
                        retargeting_components["flag_animate_transition"],
                        retargeting_components["flag_do_crop_input_retargeting"],
                    ],
                    outputs=[
                        retargeting_components["retargeting_output"],
                        retargeting_components["retargeting_output_paste_back"],
                        retargeting_components["retargeting_output_video"],
                        retargeting_components["retargeting_output_video_paste_back"],
                    ],
                )

            # Wire up video retargeting handlers
            video_retargeting_components["process_video_retargeting_button"].click(
                fn=gpu_wrapped_execute_video_retargeting,
                inputs=[
                    video_retargeting_components["video_lip_retargeting_slider"],
                    video_retargeting_components["retargeting_input_video"],
                    video_retargeting_components["video_retargeting_source_scale"],
                    video_retargeting_components[
                        "driving_smooth_observation_variance_retargeting"
                    ],
                    video_retargeting_components["video_retargeting_silence"],
                    video_retargeting_components[
                        "flag_do_crop_input_retargeting_video"
                    ],
                ],
                outputs=[
                    video_retargeting_components["video_retargeting_output"],
                    video_retargeting_components["video_retargeting_output_paste_back"],
                ],
                show_progress=True,
            )

            # Add reset handlers
            retargeting_components["reset_retargeting_button"].click(
                fn=reset_sliders,
                inputs=[],
                outputs=retargeting_sliders
                + [
                    retargeting_components["retargeting_source_scale"],
                    retargeting_components["flag_stitching_retargeting"],
                    retargeting_components["flag_animate_transition"],
                    retargeting_components["flag_do_crop_input_retargeting"],
                ],
            )

            # Add clear handlers
            retargeting_components["clear_retargeting_button"].click(
                fn=None,
                inputs=[],
                outputs=[
                    retargeting_components["retargeting_input_image"],
                    retargeting_components["retargeting_output"],
                    retargeting_components["retargeting_output_paste_back"],
                    retargeting_components["retargeting_output_video"],
                    retargeting_components["retargeting_output_video_paste_back"],
                ],
                _js="() => [null, null, null, null, null]",
            )

            video_retargeting_components["clear_video_retargeting_button"].click(
                fn=None,
                inputs=[],
                outputs=[
                    video_retargeting_components["retargeting_input_video"],
                    video_retargeting_components["video_retargeting_output"],
                    video_retargeting_components["video_retargeting_output_paste_back"],
                ],
                _js="() => [null, null, null]",
            )

            animation_components["process_button_reset"].click(
                fn=reset_sliders,
                inputs=[],
                outputs=[
                    animation_components["vx_ratio"],
                    animation_components["vy_ratio"],
                    animation_components["scale"],
                    animation_components["vx_ratio_crop_driving_video"],
                    animation_components["vy_ratio_crop_driving_video"],
                    animation_components["scale_crop_driving_video"],
                    animation_components["flag_normalize_lip"],
                    animation_components["flag_relative_input"],
                    animation_components["flag_remap_input"],
                    animation_components["flag_stitching_input"],
                    animation_components["flag_do_crop_input"],
                    animation_components["flag_crop_driving_video_input"],
                    animation_components["source_face_index"],
                    animation_components["driving_face_index"],
                    animation_components["driving_multiplier"],
                    animation_components["driving_smooth_observation_variance"],
                    animation_components["flag_eye_retargeting"],
                    animation_components["flag_lip_retargeting"],
                    animation_components["flag_source_video_eye_retargeting"],
                ],
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
