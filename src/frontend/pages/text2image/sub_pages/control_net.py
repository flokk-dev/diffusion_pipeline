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

# IMPORT: project
from src.frontend.component import Component, Prompts, Hyperparameters

from src.backend.image_processing import ImageProcessingManager
from src.backend.deep_learning.diffusion import ControlNetStableDiffusion


class ControlNetSubPage:
    """ Represents the page allowing to process images. """
    def __init__(self):
        """ Initializes the page allowing to process images. """

        # ----- Components ----- #
        # Creates the component allowing to select the ControlNets to use
        self.controlnet = ControlNet(parent=self)

        # Creates the component allowing to specify the prompt/negative prompt
        self.prompts = Prompts(parent=self)

        # Creates the component allowing to adjust the hyperparameters
        self.hyperparameters = Hyperparameters(parent=self)

        # Creates the component allowing to generate and display images
        self.image_generation = ImageGeneration(parent=self)


class ControlNet(Component):
    """ Represents the component allowing to select the ControlNets to use. """
    def __init__(self, parent: Any):
        """
        Initializes the component allowing to select the ControlNets to use.

        Parameters
        ----------
            parent: Any
                parent of the component
        """
        super(ControlNet, self).__init__(parent=parent)

        # ----- Attributes ----- #
        # Images
        self.images: List[gr.Image] = list()
        self.masks: List[gr.Image] = list()

        # Select boxes
        self.processing: List[gr.Dropdown] = list()
        self.weights: List[gr.Slider] = list()

        # Image processing
        self.image_processing_manager = ImageProcessingManager()

        # ----- Components ----- #
        with gr.Accordion(label="ControlNets", open=False):
            # Creates 3 ControlNet slots
            for idx in range(3):
                # Creates a tab for each ControlNet slot
                with gr.Tab(f"n°{idx}"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Creates the component allowing to upload an image
                            self.images.append(gr.Image(label="Image").style(height=350))

                        with gr.Column(scale=1):
                            # Creates the component allowing to display the mask of the image
                            self.masks.append(gr.Image(label="Mask").style(height=350))

                    with gr.Row():
                        with gr.Column(scale=1):
                            # Creates the select box allowing to select the image processing
                            self.processing.append(
                                gr.Dropdown(
                                    label="Processing",
                                    choices=self.image_processing_manager.keys()
                                )
                            )

                        with gr.Column(scale=1):
                            # Creates the select box allowing to select the ControlNet
                            self.weights.append(
                                gr.Slider(
                                    label="Weight",
                                    minimum=0.0, maximum=1.0,
                                    value=1.0,
                                    step=0.01,
                                    interactive=True
                                )
                            )

                    # Creates the button allowing to run the image processing
                    button = gr.Button("Process image")
                    button.click(
                        fn=self.on_click,
                        inputs=[self.processing[idx], self.images[idx]],
                        outputs=[self.masks[idx]]
                    )

    def on_click(self, processing, image):
        return self.image_processing_manager(processing, image)


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
        self.diffusion = ControlNetStableDiffusion

        # Creates the carousel containing the generated images
        self.generated_images: gr.Gallery = gr.Gallery(label="Generated images").style(grid=4)

        # Creates the button allowing to generate images
        self.generation: gr.Button = gr.Button("Generate images").style(full_width=True)
        self.generation.click(
            fn=self.on_click,
            inputs=[
                # *self.parent.controlnet.retrieve_info(),
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
        masks: List[gr.Image] = list()
        weights: List[gr.Slider] = list()
        processing: List[gr.Dropdown] = list()

        for idx, mask in enumerate(self.parent.controlnet.masks):
            print(self.parent.controlnet.images[idx].shape)
            print(mask.shape)
            if isinstance(mask, gr.Image):
                masks.append(mask)
                weights.append(self.parent.controlnet.weights[idx])
                processing.append(self.parent.controlnet.processing[idx])

        # If not instantiated or ControlNet ids have been modified
        if isinstance(self.diffusion, type) or processing != self.diffusion.controlnet_ids:
            self.diffusion = self.diffusion(controlnet_ids=processing)

        args: Dict[Any] = {
            "images": masks,
            "weights": weights,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_images": int(num_images) if num_images > 0 else 1,
            "width": width,
            "height": height,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "seed": seed if seed >= 0 else None,
        }

        _, generated_images = self.diffusion(**args)

        return generated_images
