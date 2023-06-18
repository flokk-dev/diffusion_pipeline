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
import utils

from src.backend.image_generation import Image2ImageDiffuser
from src.frontend.component import Prompts, Hyperparameters, ImageGeneration


class Image2ImagePage:
    """ Allows to generate an image using another image. """

    def __init__(self):
        """ Allows to generate an image using another image. """
        # ----- Components ----- #
        with gr.Accordion(label="Image", open=True):
            # Creates the component allowing to upload an image
            image = gr.Image(label="Image", height=350)

        # Creates the component allowing to specify the prompt/negative prompt
        self.prompts: Prompts = Prompts(parent=self)

        # Creates the component allowing to adjust the hyperparameters
        self.hyperparameters: Hyperparameters = CustomHyperparameters(parent=self)

        # Creates the component allowing to generate and display images
        self.image_generation: ImageGeneration = ImageGeneration(
            parent=self, diffuser_type=Image2ImageDiffuser
        )

        # Defines the image generation inputs and outputs
        self.image_generation.button.click(
            fn=self.on_click,
            inputs=[
                *self.image_generation.retrieve_info(),
                image,
                *self.prompts.retrieve_info(),
                *self.hyperparameters.retrieve_info()
            ],
            outputs=[self.image_generation.generated_images]
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
            "image": utils.resize_image(image, resolution=512),
            "strength": strength,
            "negative_prompt": negative_prompt,
            "num_images": int(num_images) if num_images > 0 else 1,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "seed": int(seed) if seed >= 0 else None
        }

        # Verifies if an instantiation of the diffuser is needed
        if self.image_generation.diffuser is None or self.image_generation.diffuser.is_different(
            pipeline_path=pipeline_id
        ):
            self.image_generation.diffuser = Image2ImageDiffuser(pipeline_id)

        return self.image_generation.diffuser(**self.args)


class CustomHyperparameters(Hyperparameters):
    """ Allows to adjust the hyperparameters. """

    def __init__(self, parent: Any):
        """
        Allows to adjust the hyperparameters.

        Parameters
        ----------
            parent: Any
                parent of the component
        """
        super(CustomHyperparameters, self).__init__(parent)

    def _create_row_2(self):
        """ Creates the components of the second row. """
        # Creates the slider allowing to specify the strength of an input image
        self.components["strength"]: gr.Slider = gr.Slider(
            label="Strength of the input image",
            minimum=0.0, maximum=1.0,
            value=0.2,
            step=0.01,
            interactive=True
        )

        # Creates the other hyperparameters of the row n°2
        super()._create_row_2()

    def retrieve_info(self) -> List[Any]:
        """
        Retrieves the component information.

        Returns
        ----------
            List[Any]
                info within the component
        """
        return list(self.components.values())
