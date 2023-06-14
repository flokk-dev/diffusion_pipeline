"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import gradio as gr

# IMPORT: project
from paths import FAVICON
from src.frontend import \
    StableDiffusionPage, ControlNetPage, \
    ImageInpaintingPage, Image2ImagePage, Pix2PixPage, \
    Image2PromptPage, \
    Text2PromptPage

with gr.Blocks() as demo:
    # ----- Components ----- #
    # Instantiates the Text-to-Image page
    with gr.Tab("Text-to-Image"):
        with gr.Tab("Diffusion"):
            StableDiffusionPage()
        with gr.Tab("ControlNet"):
            ControlNetPage()

    # Instantiates the Image-to-Image page
    with gr.Tab("Image-to-Image"):
        with gr.Tab("Inpainting"):
            ImageInpaintingPage()
        with gr.Tab("Image2Image"):
            Image2ImagePage()
        with gr.Tab("Pix2Pix"):
            Pix2PixPage()

    # Instantiates the Image-to-Prompt page
    with gr.Tab("Image-to-Prompt"):
        Image2PromptPage()

    # Instantiates the Text-to-Prompt page
    with gr.Tab("Text-to-Prompt"):
        Text2PromptPage()

if __name__ == "__main__":
    demo.launch(favicon_path=FAVICON, server_port=8080)
