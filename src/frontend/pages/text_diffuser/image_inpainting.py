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
import numpy as np

# IMPORT: project
import utils

from src.backend.text_diffuser import ImageInpaintDiffuser
from src.frontend.component import \
    Component, Prompts, Hyperparameters, ImageGeneration, RankingFeedback


class ImageInPaintPage:
    """ Allows to generate images. """

    def __init__(self):
        """ Allows to generate images. """
        # ----- Attributes ----- #
        self.diffuser: Any = None

        self.latents: torch.Tensor = None
        self.args: Dict[str, Any] = dict()

        # ----- Components ----- #
        # Creates the component allowing to create the input images
        self.image_painter: ImagePainter = ImagePainter(parent=self)

        # Creates the component allowing to specify the prompt/negative prompt
        self.prompts: Prompts = Prompts(parent=self)

        # Creates the component allowing to adjust the hyperparameters
        self.hyperparameters: Hyperparameters = Hyperparameters(parent=self)

        # Creates the component allowing to generate and display images
        self.image_generation: ImageGeneration = ImageGeneration(
            parent=self, diffuser_type=ImageInpaintDiffuser
        )

        # Creates the component allowing the user to give its feedback
        self.ranking_feedback: RankingFeedback = RankingFeedback(parent=self)

        # Defines the image generation inputs and outputs
        self.image_generation.button.click(
            fn=self.on_click,
            inputs=[
                *self.image_generation.retrieve_info(),
                self.image_painter.image,
                self.image_painter.mask,
                *self.prompts.retrieve_info(),
                *self.hyperparameters.retrieve_info()
            ],
            outputs=[
                self.image_generation.generated_images,
                self.ranking_feedback.row_1,
                self.ranking_feedback.row_2,
                self.ranking_feedback.row_3
            ]
        )

    def on_click(
            self,
            pipeline_id: str,
            image: np.ndarray,
            mask: np.ndarray,
            prompt: str,
            negative_prompt: str,
            num_images: int,
            seed: int,
            guidance_scale: float,
            num_steps: int
    ):
        # Creates the dictionary of arguments
        self.args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "mask": mask,
            "num_images": int(num_images) if num_images > 0 else 1,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "seed": int(seed) if seed >= 0 else None,
        }

        # Verifies if an instantiation of the diffuser is needed
        if self.image_generation.diffuser is None:
            self.image_generation.diffuser = ImageInpaintDiffuser(pipeline_id)

        self.latents, generated_images = self.image_generation.diffuser(**self.args)
        return generated_images, \
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


class ImagePainter(Component):
    """ Allows to paint an image. """

    def __init__(self, parent: Any):
        """
        Allows to paint an image.

        Parameters
        ----------
            parent: Any
                parent of the component
        """
        super(ImagePainter, self).__init__(parent)

        # ----- Attributes ----- #
        # Images
        self.image: gr.Image = None
        self.mask: gr.Image = None

        # ----- Components ----- #
        with gr.Accordion(label="Images", open=True):
            with gr.Row():
                # Creates the component allowing to upload an image
                self.image = gr.Image(label="Image").style(height=350)

                # Creates the component allowing to display the prompt
                self.mask = gr.Image(label="Mask").style(height=350)
