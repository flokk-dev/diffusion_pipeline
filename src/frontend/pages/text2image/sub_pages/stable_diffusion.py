"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import gradio as gr

import torch

# IMPORT: project
from src.frontend.component import Prompts, Hyperparameters, RankingFeedback
from src.backend.deep_learning.diffusion import StableDiffuser


class StableDiffusionSubPage:
    """ Represents the page allowing to process images. """
    def __init__(self):
        """ Initializes the page allowing to process images. """

        # ----- Components ----- #
        # Creates the component allowing to specify the prompt/negative prompt
        self.prompts: Prompts = Prompts(parent=self)

        # Creates the component allowing to adjust the hyperparameters
        self.hyperparameters: Hyperparameters = Hyperparameters(parent=self)

        # ----- Attributes ----- #
        # Creates the object allowing to generate images
        self.diffusion: type | StableDiffuser = StableDiffuser

        # Creates the arguments of the diffusion
        self.args: dict = None
        self.latents: torch.Tensor = None

        # ----- Components ----- #
        with gr.Accordion(label="Generation", open=True):
            # Creates the carousel containing the generated images
            self.generated_images: gr.Gallery = gr.Gallery(label="Images").style(grid=3)

            # Creates the button allowing to generate images
            button: gr.Button = gr.Button("Generate new images").style(full_width=True)

        # Creates the component allowing the user to give its feedback
        self.ranking_feedback: RankingFeedback = RankingFeedback(parent=self)
        button.click(
                fn=self.on_click,
                inputs=[
                    *self.prompts.retrieve_info(),
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
        return generated_images, \
            gr.update(open=True, visible=True), \
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
