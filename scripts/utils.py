import gradio as gr
from modules import shared


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
