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
import torch

# IMPORT: project
from src.frontend.component import Prompts, Hyperparameters
from src.backend.deep_learning.diffusion import Image2ImageDiffuser


class Image2ImageSubPage:
    """ Represents the page allowing to process images. """
    def __init__(self):
        """ Initializes the page allowing to process images. """

        # ----- Attributes ----- #
        # Creates the object allowing to generate images
        self.diffuser: Image2ImageDiffuser = None

        # Creates the arguments of the diffusion
        self.args: dict = None
        self.latents: torch.Tensor = None

        # ----- Components ----- #
        with gr.Accordion(label="Image", open=True):
            # Creates the component allowing to upload an image
            image = gr.Image(label="Image").style(height=350)

        # Creates the component allowing to specify the prompt/negative prompt
        self.prompts: Prompts = Prompts(parent=self)

        # Creates the component allowing to adjust the hyperparameters
        self.hyperparameters: Hyperparameters = Hyperparameters(
            parent=self,
            components=["num_images", "seed", "guidance_scale", "strength", "num_steps"]
        )

        # Creates the component allowing to generate and display images
        with gr.Accordion(label="Generation", open=True):
            # Creates the carousel containing the generated images
            self.generated_images: gr.Gallery = gr.Gallery(label="Images").style(grid=3)

            with gr.Row():
                # Creates the list of the available diffusion models
                pipeline_id: gr.Dropdown = gr.Dropdown(
                    label="Diffusion model",
                    choices=Image2ImageDiffuser.PIPELINES.keys()
                )

                # Creates the button allowing to generate images
                button: gr.Button = gr.Button("Generate new images").style(full_width=True)
                button.click(
                        fn=self.on_click,
                        inputs=[
                            pipeline_id,
                            image,
                            *self.prompts.retrieve_info(),
                            *self.hyperparameters.retrieve_info()
                        ],
                        outputs=[self.generated_images]
                    )

    def on_click(
            self,
            pipeline_id: str,
            image: np.ndarray,
            prompt: str,
            negative_prompt: str,
            num_images: int,
            seed: int,
            strength: float,
            guidance_scale: float,
            num_steps: int
    ):
        # Creates the dictionary of arguments
        self.args = {
            "prompt": prompt,
            "image": image,
            "strength": strength,
            "negative_prompt": negative_prompt,
            "num_images": int(num_images) if num_images > 0 else 1,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "seed": int(seed) if seed >= 0 else None
        }

        # Instantiates the StableDiffusion pipeline if needed
        if self.diffuser is None or self.diffuser.is_different(pipeline_id):
            self.diffuser = Image2ImageDiffuser(pipeline_id)

        return self.diffuser(**self.args)