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
from src.frontend.component import Component


class Hyperparameters(Component):
    """ Represents the component allowing to adjust the hyperparameters. """
    def __init__(self, parent: Any):
        """
        Initializes the component allowing to adjust the hyperparameters.

        Parameters
        ----------
            parent: Any
                parent of the component
        """
        super(Hyperparameters, self).__init__(parent=parent)

        # ----- Attributes ----- #
        self.num_images: gr.Number = None
        self.width: gr.Slider = None
        self.height: gr.Slider = None
        self.guidance_scale: gr.Slider = None
        self.denoising_steps: gr.Slider = None
        self.seed: gr.Number = None

        # ----- Components ----- #
        with gr.Accordion(label="Hyperparameters", open=False):
            with gr.Row():
                # Creates the area allowing to specify the prompt
                self.num_images = gr.Number(
                    label="Number of images",
                    value=1,
                    interactive=True
                )

                # Creates the area allowing to specify the prompt
                self.seed = gr.Number(
                    label="Seed of the randomness",
                    value=-1,
                    interactive=True
                )

            with gr.Row():
                # Creates the slider allowing to specify the width of the image to generate
                self.width = gr.Slider(
                    label="Width",
                    minimum=0, maximum=1024,
                    value=512,
                    step=8,
                    interactive=True
                )

                # Creates the slider allowing to specify the height of the image to generate
                self.height = gr.Slider(
                    label="Height",
                    minimum=0, maximum=1024,
                    value=512,
                    step=8,
                    interactive=True
                )

                # Creates the slider allowing to specify the guidance scale
                self.guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0, maximum=21.0,
                    value=7.5,
                    step=0.1,
                    interactive=True
                )

                # Creates the slider allowing to specify the number of denoising steps
                self.denoising_steps = gr.Slider(
                    label="Denoising steps",
                    minimum=0, maximum=100,
                    value=30,
                    step=1,
                    interactive=True
                )

    def retrieve_info(self) -> List[Any]:
        """
        Retrieves the component information.

        Returns
        ----------
            List[Any]
                info within the component
        """
        return [
            self.num_images,
            self.width,
            self.height,
            self.denoising_steps,
            self.guidance_scale,
            self.seed
        ]
