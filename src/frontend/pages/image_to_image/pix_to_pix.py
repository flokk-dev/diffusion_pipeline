"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import gradio as gr
import numpy as np

# IMPORT: project
import utils

from src.backend.image_generation import Pix2PixDiffuser
from src.frontend.component import Prompts, Hyperparameters, ImageGeneration, RankingFeedback


class Pix2PixPage:
    """ Allows to generate modify an image. """

    def __init__(self):
        """ Allows to generate modify an image. """
        # ----- Components ----- #
        with gr.Accordion(label="Image", open=True):
            # Creates the component allowing to upload an image
            image = gr.Image(label="Image", height=350)

        # Creates the component allowing to specify the prompt/negative prompt
        self.prompts: Prompts = Prompts(parent=self)

        # Creates the component allowing to adjust the hyperparameters
        self.hyperparameters: Hyperparameters = Hyperparameters(parent=self)

        # Creates the component allowing to generate and display images
        self.image_generation: ImageGeneration = ImageGeneration(
            parent=self, diffuser_type=Pix2PixDiffuser
        )

        # Creates the component allowing the user to give its feedback
        self.ranking_feedback: RankingFeedback = RankingFeedback(parent=self)

        # Defines the image generation inputs and outputs
        self.image_generation.button.click(
            fn=self.on_click,
            inputs=[
                *self.image_generation.retrieve_info(),
                image,
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
            "image": utils.resize_image(image, resolution=512),
            "negative_prompt": negative_prompt,
            "num_images": int(num_images) if num_images > 0 else 1,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "seed": int(seed) if seed >= 0 else None,
        }

        # Verifies if an instantiation of the diffuser is needed
        if self.image_generation.diffuser is None or self.image_generation.diffuser.is_different(
            pipeline_path=pipeline_id
        ):
            self.image_generation.diffuser = Pix2PixDiffuser(pipeline_id)

        self.latents, generated_images = self.image_generation.diffuser(**self.args)
        return generated_images, \
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
