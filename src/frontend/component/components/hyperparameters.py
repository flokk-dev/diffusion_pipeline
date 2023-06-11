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
    def __init__(self, parent: Any, components: List[str]):
        """
        Initializes the component allowing to adjust the hyperparameters.

        Parameters
        ----------
            parent: Any
                parent of the component
            components: Dict[str: Any]
                list of the hyperparameters to create
        """
        super(Hyperparameters, self).__init__(parent=parent)

        # ----- Attributes ----- #
        self.components: Dict[str: Any] = dict()

        # ----- Components ----- #
        with gr.Accordion(label="Hyperparameters", open=True):
            with gr.Row():
                # Creates the area allowing to specify the prompt
                if "num_images" in components:
                    self.components["num_images"]: gr.Number = gr.Number(
                        label="Number of images",
                        value=1,
                        interactive=True
                    )

                # Creates the area allowing to specify the prompt
                if "seed" in components:
                    self.components["seed"]: gr.Number = gr.Number(
                        label="Seed of the randomness",
                        value=-1,
                        interactive=True
                    )

            with gr.Row():
                # Creates the slider allowing to specify the width of the image to generate
                if "width" in components:
                    self.components["width"]: gr.Slider = gr.Slider(
                        label="Width",
                        minimum=0, maximum=1024,
                        value=512,
                        step=8,
                        interactive=True
                    )

                # Creates the slider allowing to specify the height of the image to generate
                if "height" in components:
                    self.components["height"]: gr.Slider = gr.Slider(
                        label="Height",
                        minimum=0, maximum=1024,
                        value=512,
                        step=8,
                        interactive=True
                    )

            with gr.Row():
                # Creates the slider allowing to specify the strength of an input image
                if "strength" in components:
                    self.components["strength"]: gr.Slider = gr.Slider(
                        label="Strength of the input image",
                        minimum=0.0, maximum=1.0,
                        value=0.8,
                        step=0.01,
                        interactive=True
                    )

                # Creates the slider allowing to specify the guidance scale
                if "guidance_scale" in components:
                    self.components["guidance_scale"]: gr.Slider = gr.Slider(
                        label="Guidance scale",
                        minimum=0.0, maximum=21.0,
                        value=7.5,
                        step=0.1,
                        interactive=True
                    )

                # Creates the slider allowing to specify the number of denoising steps
                if "num_steps" in components:
                    self.components["num_steps"]: gr.Slider = gr.Slider(
                        label="Number of denoising steps",
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
        return list(self.components.values())
