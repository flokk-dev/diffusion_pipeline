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
from src.frontend.component import Component, Prompts, Hyperparameters, RankingFeedback

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

        # ----- Attributes ----- #
        # Creates the object allowing to generate images
        self.diffusion: type | ControlNetStableDiffusion = ControlNetStableDiffusion

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
                *self.controlnet.retrieve_info(),
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
            mask_0: np.ndarray, processing_0: str, weight_0: float,
            mask_1: np.ndarray, processing_1: str, weight_1: float,
            mask_2: np.ndarray, processing_2: str, weight_2: float,
            prompt: str,
            negative_prompt: str = "",
            num_images: int = 1,
            width: int = 512,
            height: int = 512,
            num_steps: int = 50,
            guidance_scale: float = 7.5,
            seed: int = None
    ):
        # Aggregates the ControlNet elements together
        masks: List[np.ndarray] = list()
        controlnet_ids: List[str] = list()
        weights: List[float] = list()

        for mask, processing, weight in zip(
                [mask_0, mask_1, mask_2],
                [processing_0, processing_1, processing_2],
                [weight_0, weight_1, weight_2]
        ):
            if isinstance(mask, np.ndarray):
                masks.append(mask)
                controlnet_ids.append(processing)
                weights.append(weight)

        # Instantiates the ControlNet pipeline if needed
        if isinstance(self.diffusion, type) or controlnet_ids != self.diffusion.controlnet_ids:
            self.diffusion = self.diffusion(controlnet_ids=controlnet_ids)

        # Creates the dictionary of arguments
        self.args = {
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

        self.latents, generated_images = self.diffusion(**self.args)
        return generated_images, \
            gr.update(open=True, visible=True), \
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


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
                        inputs=[self.images[idx], self.processing[idx]],
                        outputs=[self.masks[idx]]
                    )

    def on_click(self, image, processing):
        return self.image_processing_manager(image, processing)

    def retrieve_info(self) -> List[Any]:
        info: List[Any] = list()
        for idx in range(len(self.images)):
            info.append(self.masks[idx])
            info.append(self.processing[idx])
            info.append(self.weights[idx])

        return info
