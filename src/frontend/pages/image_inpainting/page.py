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

# IMPORT: utils
from src.frontend.component import Component, Prompts, Hyperparameters, RankingFeedback
from src.backend.deep_learning import ImageInpaintDiffuser


class ImageInpainting:
    """ Represents the page allowing to write a prompt describing an image. """
    def __init__(self):
        """ Initializes the page allowing to write a prompt describing an image. """

        # ----- Attributes ----- #
        # Creates the object allowing to generate images
        self.diffuser: ImageInpaintDiffuser = None

        # Creates the arguments of the diffusion
        self.args: dict = None
        self.latents: torch.Tensor = None

        # ----- Components ----- #
        # Creates
        self.image_painter: ImagePainter = ImagePainter(parent=self)

        # Creates the component allowing to specify the prompt/negative prompt
        self.prompts: Prompts = Prompts(parent=self)

        # Creates the component allowing to adjust the hyperparameters
        self.hyperparameters: Hyperparameters = Hyperparameters(
            parent=self,
            components=["num_images", "seed", "guidance_scale", "num_steps"]
        )

        # Creates the component allowing to generate and display images
        with gr.Accordion(label="Generation", open=True):
            # Creates the carousel containing the generated images
            self.generated_images: gr.Gallery = gr.Gallery(label="Images").style(grid=3)

            with gr.Row():
                # Creates the list of the available diffusion models
                pipeline_id: gr.Dropdown = gr.Dropdown(
                    label="Diffusion model",
                    choices=ImageInpaintDiffuser.PIPELINES.keys()
                )

                # Creates the button allowing to generate images
                button: gr.Button = gr.Button("Generate new images").style(full_width=True)
                button.click(
                    fn=self.on_click,
                    inputs=[
                        pipeline_id,
                        self.image_painter.image,
                        *self.prompts.retrieve_info(),
                        *self.hyperparameters.retrieve_info()
                    ],
                    outputs=[self.generated_images]
                )

    def on_change(self, image_to_mask: Dict[str, np.ndarray]):
        return image_to_mask["mask"]

    def on_click(
            self,
            pipeline_id: str,
            image_to_mask: Dict[str, np.ndarray],
            prompt: str,
            negative_prompt: str = "",
            num_images: int = 1,
            seed: int = None,
            guidance_scale: float = 7.5,
            num_steps: int = 50
    ):
        # Creates the dictionary of arguments
        self.args = {
            "prompt": prompt,
            "image": image_to_mask["image"],
            "mask": image_to_mask["mask"],
            "negative_prompt": negative_prompt,
            "num_images": int(num_images) if num_images > 0 else 1,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "seed": int(seed) if seed >= 0 else None,
        }

        # Instantiates the StableInpaintDiffuser pipeline if needed
        if self.diffuser is None or self.diffuser.need_instantiation(pipeline_id):
            self.diffuser = ImageInpaintDiffuser(pipeline_id)

        self.latents, generated_images = self.diffuser(**self.args)
        return generated_images


class ImagePainter(Component):
    """ Represents the component allowing to paint an image. """
    def __init__(self, parent: Any):
        """
        Initializes the component allowing to paint an image.

        Parameters
        ----------
            parent: Any
                parent of the component
        """
        super(ImagePainter, self).__init__(parent=parent)

        # ----- Attributes ----- #
        # Images
        self.image: gr.Image = None
        self.mask: gr.Image = None

        # ----- Components ----- #
        with gr.Accordion(label="Images", open=True):
            with gr.Row():
                # Creates the component allowing to upload an image
                self.image = gr.Image(label="Image", tool="sketch").style(height=350)

                # Creates the component allowing to display the prompt
                self.mask = gr.Image(label="Mask").style(height=350)

            # Creates the component allowing to display the mask
            button = gr.Button("Display the mask")
            button.click(
                fn=self.on_click,
                inputs=[self.image],
                outputs=[self.mask]
            )

    @staticmethod
    def on_click(image):
        return image["mask"]
