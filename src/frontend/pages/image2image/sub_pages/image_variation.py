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
from src.frontend.component import Prompts, Hyperparameters, RankingFeedback
from src.backend.deep_learning.diffusion import ImageVariationDiffuser


class ImageVariationSubPage:
    """ Represents the page allowing to process images. """
    def __init__(self):
        """ Initializes the page allowing to process images. """

        # ----- Attributes ----- #
        # Creates the object allowing to generate images
        self.diffuser: ImageVariationDiffuser = None

        # Creates the arguments of the diffusion
        self.args: dict = None
        self.latents: torch.Tensor = None

        # ----- Components ----- #
        with gr.Accordion(label="Image", open=True):
            # Creates the component allowing to upload an image
            image = gr.Image(label="Image").style(height=350)

        # Creates the component allowing to adjust the hyperparameters
        self.hyperparameters: Hyperparameters = Hyperparameters(
            parent=self,
            components=["num_images", "seed", "width", "height", "guidance_scale", "num_steps"]
        )

        # Creates the component allowing to generate and display images
        with gr.Accordion(label="Generation", open=True):
            # Creates the carousel containing the generated images
            self.generated_images: gr.Gallery = gr.Gallery(label="Images").style(grid=3)

            with gr.Row():
                # Creates the list of the available diffusion models
                pipeline_id: gr.Dropdown = gr.Dropdown(
                    label="Diffusion model",
                    choices=ImageVariationDiffuser.PIPELINES.keys()
                )

                # Creates the button allowing to generate images
                button: gr.Button = gr.Button("Generate new images").style(full_width=True)

        # Creates the component allowing the user to give its feedback
        self.ranking_feedback: RankingFeedback = RankingFeedback(parent=self)
        button.click(
                fn=self.on_click,
                inputs=[
                    pipeline_id,
                    image,
                    *self.hyperparameters.retrieve_info()
                ],
                outputs=[
                    self.generated_images,
                    self.ranking_feedback.container,
                    self.ranking_feedback.row_1,
                    self.ranking_feedback.row_2,
                    self.ranking_feedback.row_3
                ]
            )

    def on_click(
            self,
            pipeline_id: str,
            image: np.ndarray,
            num_images: int = 1,
            seed: int = None,
            width: int = 512,
            height: int = 512,
            guidance_scale: float = 7.5,
            num_steps: int = 50
    ):
        # Creates the dictionary of arguments
        self.args = {
            "image": image,
            "num_images": int(num_images) if num_images > 0 else 1,
            "width": width,
            "height": height,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "seed": int(seed) if seed >= 0 else None,
        }

        # Instantiates the StableDiffusion pipeline if needed
        if self.diffuser is None or self.diffuser.need_instantiation(pipeline_id):
            self.diffuser = ImageVariationDiffuser(pipeline_id)

        self.latents, generated_images = self.diffuser(**self.args)
        return generated_images, \
            gr.update(open=False, visible=True), \
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
