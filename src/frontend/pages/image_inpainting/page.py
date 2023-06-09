"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import gradio as gr

import numpy as np

# IMPORT: utils
from src.frontend.component import Prompts, Hyperparameters, RankingFeedback
from src.backend.deep_learning import StableInpaintDiffuser


class ImageInpainting:
    """ Represents the page allowing to write a prompt describing an image. """
    def __init__(self):
        """ Initializes the page allowing to write a prompt describing an image. """
        # ----- Components ----- #
        with gr.Accordion(label="Images", open=True):
            with gr.Row():
                # Creates the component allowing to upload an image
                image_to_mask = gr.Image(label="Image", tool="sketch").style(height=350)

                # Creates the component allowing to display the prompt
                modified_image = gr.Image(label="Modified image").style(height=350)

        # Creates the component allowing to specify the prompt/negative prompt
        self.prompts: Prompts = Prompts(parent=self)

        # Creates the component allowing to adjust the hyperparameters
        self.hyperparameters: Hyperparameters = Hyperparameters(parent=self)

        # Creates the object allowing to generate images
        self.diffusion: type | StableInpaintDiffuser = StableInpaintDiffuser

        self.button = gr.Button("Inpaint the image")
        self.button.click(
            fn=self.on_click,
            inputs=[
                image_to_mask,
                *self.prompts.retrieve_info(),
                *self.hyperparameters.retrieve_info()
            ],
            outputs=[modified_image]
        )

    def on_click(
            self,
            image_to_mask: Dict[str, np.ndarray],
            prompt: str,
            negative_prompt: str = "",
            num_images: int = 1,
            width: int = 512,
            height: int = 512,
            num_steps: int = 50,
            guidance_scale: float = 7.5,
            seed: int = None
    ):
        # Creates the dictionary of arguments
        self.args = {
            "prompt": prompt,
            "image": image_to_mask["image"],
            "mask": image_to_mask["mask"],
            "negative_prompt": negative_prompt,
            "num_images": int(num_images) if num_images > 0 else 1,
            "width": width,
            "height": height,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "seed": int(seed) if seed >= 0 else None,
        }

        # Instantiates the StableDiffusion pipeline if needed
        if isinstance(self.diffusion, type):
            self.diffusion = self.diffusion()

        self.latents, generated_images = self.diffusion(**self.args)
        return generated_images
