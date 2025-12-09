import os
from PIL import Image
import numpy as np
import cv2
import gradio as gr

from style_transfer_cartoon import run_style_transfer
from cartoon_filter import cartoonize_opencv  # our OpenCV cartoon effect

# Map style names -> style image paths (for NST only)
STYLE_FILES = {
    "Van Gogh - Starry Night": "style/style.png",
}

CONTENT_PATH = "input/temp_content.png"
OUTPUT_PATH_NST = "output/temp_result_nst.png"
OUTPUT_PATH_CARTOON = "output/temp_result_cartoon.png"


def stylize_image(
    input_image: Image.Image,
    effect_name: str,
    progress=gr.Progress(track_tqdm=False),
):
    """
    Gradio callback: takes uploaded image + selected effect,
    runs either NST or Cartoon filter, returns stylized image.
    """
    if input_image is None:
        return None

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # --- Neural Style Transfer mode ---
    if effect_name == "Neural Style Transfer (Starry Night)":
        # Save uploaded content for NST
        input_image.save(CONTENT_PATH)

        style_path = STYLE_FILES["Van Gogh - Starry Night"]

        progress(0, desc="Starting neural style transfer...")
        run_style_transfer(
            content_path=CONTENT_PATH,
            style_path=style_path,
            output_path=OUTPUT_PATH_NST,
            num_steps=100,
            style_weight=8e4,
            content_weight=20.0,
            progress=progress,
        )

        output_image = Image.open(OUTPUT_PATH_NST).convert("RGB")
        return output_image

    # --- Cartoon / Disney-ish mode ---
    elif effect_name == "Cartoon / Disney-ish Filter (OpenCV)":
        progress(0, desc="Applying cartoon filter...")

        # Convert PIL -> OpenCV BGR
        img_rgb = np.array(input_image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        cartoon_bgr = cartoonize_opencv(img_bgr)

        # Convert back BGR -> RGB -> PIL
        cartoon_rgb = cv2.cvtColor(cartoon_bgr, cv2.COLOR_BGR2RGB)
        cartoon_pil = Image.fromarray(cartoon_rgb)

        cartoon_pil.save(OUTPUT_PATH_CARTOON)
        progress(1.0, desc="Done!")
        return cartoon_pil

    else:
        # Fallback (shouldn't happen)
        return input_image


with gr.Blocks(title="Image Stylization Demo") as demo:
    # Title banner
    gr.Markdown(
        """
# 🎨 Image Stylization – Deep Learning Mini Project

Upload any photo and generate:
- 🌌 A **Neural Style Transfer** output using Van Gogh's *Starry Night*  
- 🧸 A **Cartoon / Disney-ish** version using an OpenCV-based filter  
        """
    )

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(
                type="pil",
                label="Upload content image",
            )
            effect_dropdown = gr.Dropdown(
                choices=[
                    "Neural Style Transfer (Starry Night)",
                    "Cartoon / Disney-ish Filter (OpenCV)",
                ],
                value="Neural Style Transfer (Starry Night)",
                label="Choose Effect",
            )
            generate_btn = gr.Button("✨ Generate Stylized Image")

        with gr.Column():
            output_img = gr.Image(
                type="pil",
                label="Stylized output",
            )

    generate_btn.click(
        fn=stylize_image,
        inputs=[input_img, effect_dropdown],
        outputs=output_img,
    )

if __name__ == "__main__":
    demo.launch(share=True)
