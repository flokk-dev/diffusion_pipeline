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
import utils

from src.backend.image_processing import ImageProcessingManager
from src.backend.image_generation import ControlNetDiffuser

from src.frontend.component import \
    Component, Prompts, Hyperparameters, ImageGeneration, RankingFeedback


class ControlNetPage:
    """ Allows to generate images using control masks. """

    def __init__(self):
        """ Allows to generate images using control masks. """
        # ----- Components ----- #
        # Creates the component allowing to select the ControlNets to use
        self.controlnet = ControlNet(parent=self)

        # Creates the component allowing to specify the prompt/negative prompt
        self.prompts: Prompts = Prompts(parent=self)

        # Creates the component allowing to adjust the hyperparameters
        self.hyperparameters: Hyperparameters = CustomHyperparameters(parent=self)

        # Creates the component allowing to generate and display images
        self.image_generation: ImageGeneration = ImageGeneration(
            parent=self, diffuser_type=ControlNetDiffuser
        )

        # Creates the component allowing the user to give its feedback
        self.ranking_feedback: RankingFeedback = RankingFeedback(parent=self)

        # Defines the image generation inputs and outputs
        self.image_generation.button.click(
            fn=self.on_click,
            inputs=[
                *self.image_generation.retrieve_info(),
                *self.controlnet.retrieve_info(),
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
            mask_0: np.ndarray, processing_0: str, weight_0: float,
            mask_1: np.ndarray, processing_1: str, weight_1: float,
            mask_2: np.ndarray, processing_2: str, weight_2: float,
            prompt: str,
            negative_prompt: str,
            num_images: int,
            seed: int,
            width: int,
            height: int,
            guidance_scale: float,
            num_steps: int
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
                masks.append(utils.resize_image(mask, resolution=512))
                controlnet_ids.append(processing)
                weights.append(weight)

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
            "seed": int(seed) if seed >= 0 else None,
        }

        # Verifies if an instantiation of the diffuser is needed
        if self.image_generation.diffuser is None or self.image_generation.diffuser.is_different(
            pipeline_path=pipeline_id, controlnets=controlnet_ids
        ):
            self.image_generation.diffuser = ControlNetDiffuser(pipeline_id, controlnet_ids)

        self.latents, generated_images = self.image_generation.diffuser(**self.args)
        return generated_images, \
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


class ControlNet(Component):
    """ Allows to select the ControlNets to use and generate the masks. """

    def __init__(self, parent: Any):
        """
        Allows to select the ControlNets to use and generate the masks.

        Parameters
        ----------
            parent: Any
                parent of the component
        """
        super(ControlNet, self).__init__(parent)

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
                        # Creates the select box allowing to select the image processing
                        self.processing.append(
                            gr.Dropdown(
                                label="Image processing",
                                choices=list(self.image_processing_manager.keys())
                            )
                        )

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
        mask = self.image_processing_manager(image, processing)
        print(mask.shape)
        return mask

    def retrieve_info(self) -> List[Any]:
        info: List[Any] = list()
        for idx in range(len(self.images)):
            info.append(self.masks[idx])
            info.append(self.processing[idx])
            info.append(self.weights[idx])

        return info


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
        # Creates the slider allowing to specify the width of the image to generate
        self.components["width"]: gr.Slider = gr.Slider(
            label="Width",
            minimum=0, maximum=1024,
            value=512,
            step=8,
            interactive=True
        )

        # Creates the slider allowing to specify the height of the image to generate
        self.components["height"]: gr.Slider = gr.Slider(
            label="Height",
            minimum=0, maximum=1024,
            value=512,
            step=8,
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
