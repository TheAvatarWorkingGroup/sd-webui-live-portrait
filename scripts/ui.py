import gradio as gr


def add_source_image_buttons(image_component):
    """Add buttons for sending images to/from img2img, txt2img, and extras tabs."""
    with gr.Row():
        send_to_img2img = gr.Button("Send to img2img", variant="secondary")
        send_to_txt2img = gr.Button("Send to txt2img", variant="secondary")
        send_to_extras = gr.Button("Send to extras", variant="secondary")

    return {
        "img2img": send_to_img2img,
        "txt2img": send_to_txt2img,
        "extras": send_to_extras,
    }


def create_portrait_animation_tab(init_pipeline_fn):
    """Create the portrait animation tab with all its components."""
    components = {}

    with gr.Row(equal_height=True):
        with gr.Column():
            # Source input tabs
            with gr.Tabs() as source_tabs:
                with gr.Tab("Image", id="source_image_tab"):
                    source_image_input = gr.Image(type="filepath", label="Source Image")
                    add_source_image_buttons(source_image_input)
                with gr.Tab("Video", id="source_video_tab"):
                    source_video_input = gr.Video(label="Source Video")

            components["source_tabs"] = source_tabs
            components["source_image_input"] = source_image_input
            components["source_video_input"] = source_video_input

            with gr.Accordion(open=True, label="Cropping Options (Source)"):
                flag_do_crop_input = gr.Checkbox(value=True, label="Crop (source)")
                source_face_index = gr.Number(
                    value=0, label="Face Index", minimum=0, maximum=999, step=1
                )
                scale = gr.Number(
                    value=2.3, label="Crop Scale", minimum=1.8, maximum=3.2, step=0.05
                )
                vx_ratio = gr.Number(
                    value=0.0, label="Crop X", minimum=-0.5, maximum=0.5, step=0.01
                )
                vy_ratio = gr.Number(
                    value=-0.125, label="Crop Y", minimum=-0.5, maximum=0.5, step=0.01
                )

            components.update(
                {
                    "flag_do_crop_input": flag_do_crop_input,
                    "source_face_index": source_face_index,
                    "scale": scale,
                    "vx_ratio": vx_ratio,
                    "vy_ratio": vy_ratio,
                }
            )

            with gr.Accordion(open=True, label="Cropping Options (Driving)"):
                flag_crop_driving_video_input = gr.Checkbox(
                    value=False, label="Crop (driving)"
                )
                driving_face_index = gr.Number(
                    value=0, label="Face Index", minimum=0, maximum=999, step=1
                )
                scale_crop_driving_video = gr.Number(
                    value=2.2, label="Crop Scale", minimum=1.8, maximum=3.2, step=0.05
                )
                vx_ratio_crop_driving_video = gr.Number(
                    value=0.0, label="Crop X", minimum=-0.5, maximum=0.5, step=0.01
                )
                vy_ratio_crop_driving_video = gr.Number(
                    value=-0.1, label="Crop Y", minimum=-0.5, maximum=0.5, step=0.01
                )

            components.update(
                {
                    "flag_crop_driving_video_input": flag_crop_driving_video_input,
                    "driving_face_index": driving_face_index,
                    "scale_crop_driving_video": scale_crop_driving_video,
                    "vx_ratio_crop_driving_video": vx_ratio_crop_driving_video,
                    "vy_ratio_crop_driving_video": vy_ratio_crop_driving_video,
                }
            )

        with gr.Column():
            # Animation options
            with gr.Accordion(open=True, label="Animation Options"):
                flag_normalize_lip = gr.Checkbox(value=False, label="normalize lip")
                flag_relative_input = gr.Checkbox(value=True, label="relative motion")
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
                flag_eye_retargeting = gr.Checkbox(value=False, label="Eye retargeting")
                flag_lip_retargeting = gr.Checkbox(value=False, label="Lip retargeting")
                flag_source_video_eye_retargeting = gr.Checkbox(
                    value=False, label="Source video eye retargeting"
                )

            components.update(
                {
                    "flag_normalize_lip": flag_normalize_lip,
                    "flag_relative_input": flag_relative_input,
                    "flag_remap_input": flag_remap_input,
                    "flag_stitching_input": flag_stitching_input,
                    "animation_region": animation_region,
                    "driving_option_input": driving_option_input,
                    "driving_multiplier": driving_multiplier,
                    "driving_smooth_observation_variance": driving_smooth_observation_variance,
                    "flag_eye_retargeting": flag_eye_retargeting,
                    "flag_lip_retargeting": flag_lip_retargeting,
                    "flag_source_video_eye_retargeting": flag_source_video_eye_retargeting,
                }
            )

    with gr.Row():
        process_button_animation = gr.Button("🚀 Animate", variant="primary")
        process_button_reset = gr.Button("🧹 Clear")

    components.update(
        {
            "process_button_animation": process_button_animation,
            "process_button_reset": process_button_reset,
        }
    )

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

    components.update(
        {
            "driving_tabs": driving_tabs,
            "driving_video_input": driving_video_input,
            "driving_image_input": driving_image_input,
            "driving_image_webcam_input": driving_image_webcam_input,
            "driving_video_pickle_input": driving_video_pickle_input,
        }
    )

    with gr.Row():
        output_video_i2v = gr.Video(autoplay=False, label="Output Video")
        output_video_concat_i2v = gr.Video(
            autoplay=False, label="Output Video with Paste-back"
        )

    components.update(
        {
            "output_video_i2v": output_video_i2v,
            "output_video_concat_i2v": output_video_concat_i2v,
        }
    )

    return components


def create_portrait_retargeting_tab(init_pipeline_fn):
    """Create the portrait retargeting tab with all its components."""
    components = {}

    with gr.Row():
        with gr.Column():
            retargeting_input_image = gr.Image(type="filepath", label="Source Image")
            add_source_image_buttons(retargeting_input_image)

            components["retargeting_input_image"] = retargeting_input_image

            with gr.Accordion(open=True, label="Retargeting Options"):
                flag_do_crop_input_retargeting = gr.Checkbox(
                    value=True, label="Crop source"
                )
                flag_stitching_retargeting = gr.Checkbox(value=True, label="Stitching")
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

            components.update(
                {
                    "flag_do_crop_input_retargeting": flag_do_crop_input_retargeting,
                    "flag_stitching_retargeting": flag_stitching_retargeting,
                    "flag_animate_transition": flag_animate_transition,
                    "retargeting_source_scale": retargeting_source_scale,
                    "eye_retargeting_slider": eye_retargeting_slider,
                    "lip_retargeting_slider": lip_retargeting_slider,
                }
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

            components.update(
                {
                    "head_pitch": head_pitch,
                    "head_yaw": head_yaw,
                    "head_roll": head_roll,
                    "mov_x": mov_x,
                    "mov_y": mov_y,
                    "mov_z": mov_z,
                }
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

            components.update(
                {
                    "lip_variation_zero": lip_variation_zero,
                    "lip_variation_one": lip_variation_one,
                    "lip_variation_two": lip_variation_two,
                    "lip_variation_three": lip_variation_three,
                    "smile": smile,
                    "wink": wink,
                    "eyebrow": eyebrow,
                    "eyeball_x": eyeball_x,
                    "eyeball_y": eyeball_y,
                }
            )

        with gr.Column():
            with gr.Row():
                retargeting_output = gr.Image(type="numpy", label="Retargeting Result")
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

            components.update(
                {
                    "retargeting_output": retargeting_output,
                    "retargeting_output_paste_back": retargeting_output_paste_back,
                    "retargeting_output_video": retargeting_output_video,
                    "retargeting_output_video_paste_back": retargeting_output_video_paste_back,
                }
            )

    with gr.Row():
        reset_retargeting_button = gr.Button("🔄 Reset")
        clear_retargeting_button = gr.Button("🧹 Clear")

    components.update(
        {
            "reset_retargeting_button": reset_retargeting_button,
            "clear_retargeting_button": clear_retargeting_button,
        }
    )

    return components


def create_video_retargeting_tab(init_pipeline_fn):
    """Create the video retargeting tab with all its components."""
    components = {}

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

            components.update(
                {
                    "retargeting_input_video": retargeting_input_video,
                    "flag_do_crop_input_retargeting_video": flag_do_crop_input_retargeting_video,
                    "video_retargeting_source_scale": video_retargeting_source_scale,
                    "video_lip_retargeting_slider": video_lip_retargeting_slider,
                    "driving_smooth_observation_variance_retargeting": driving_smooth_observation_variance_retargeting,
                    "video_retargeting_silence": video_retargeting_silence,
                }
            )

        with gr.Column():
            with gr.Row():
                video_retargeting_output = gr.Video(
                    autoplay=False, label="Retargeting Result"
                )
                video_retargeting_output_paste_back = gr.Video(
                    autoplay=False, label="Paste-back Result"
                )

            components.update(
                {
                    "video_retargeting_output": video_retargeting_output,
                    "video_retargeting_output_paste_back": video_retargeting_output_paste_back,
                }
            )

    with gr.Row():
        process_video_retargeting_button = gr.Button(
            "🚗 Retarget Video", variant="primary"
        )
        clear_video_retargeting_button = gr.Button("🧹 Clear")

    components.update(
        {
            "process_video_retargeting_button": process_video_retargeting_button,
            "clear_video_retargeting_button": clear_video_retargeting_button,
        }
    )

    return components
