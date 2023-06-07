"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import gradio as gr

# IMPORT: project
from src.frontend.component import Component, Prompts, Hyperparameters
from src.backend.deep_learning.diffusion import StableDiffusion


class StableDiffusionSubPage:
    """ Represents the page allowing to process images. """
    def __init__(self):
        """ Initializes the page allowing to process images. """

        # ----- Components ----- #
        # Creates the component allowing to specify the prompt/negative prompt
        self.prompts = Prompts(parent=self)

        # Creates the component allowing to adjust the hyperparameters
        self.hyperparameters = Hyperparameters(parent=self)

        # Creates the component allowing to generate and display images
        self.image_generation = ImageGeneration(parent=self)


class ImageGeneration(Component):
    """ Represents the component allowing to generate and display images. """
    def __init__(self, parent: Any):
        """
        Initializes the component allowing to generate and display images.

        Parameters
        ----------
            parent: Any
                parent of the component
        """
        super(ImageGeneration, self).__init__(parent=parent)

        # ----- Attributes ----- #
        # Creates the object allowing to generate images
        self.diffusion = StableDiffusion

        # Creates the carousel containing the generated images
        self.generated_images: gr.Gallery = gr.Gallery(label="Generated images").style(grid=4)

        # Creates the button allowing to generate images
        self.generation: gr.Button = gr.Button("Generate images").style(full_width=True)
        self.generation.click(
            fn=self.on_click,
            inputs=[
                *self.parent.prompts.retrieve_info(),
                *self.parent.hyperparameters.retrieve_info()
            ],
            outputs=[self.generated_images]
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
        args: Dict[Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_images": int(num_images) if num_images > 0 else 1,
            "width": width,
            "height": height,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "seed": int(seed) if seed >= 0 else None,
        }

        if isinstance(self.diffusion, type):
            self.diffusion = self.diffusion()
        _, generated_images = self.diffusion(**args)

        return generated_images
