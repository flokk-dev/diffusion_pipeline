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
    """ Allows to adjust the hyperparameters. """
    def __init__(self, parent: Any):
        """
        Allows to adjust the hyperparameters.

        Parameters
        ----------
            parent: Any
                parent of the component
        """
        super(Hyperparameters, self).__init__(parent)

        # ----- Attributes ----- #
        self.components: Dict[str: Any] = dict()

        # ----- Components ----- #
        with gr.Accordion(label="Hyperparameters", open=True):
            with gr.Row():
                self._create_row_1()

            with gr.Row():
                self._create_row_2()

    def _create_row_1(self):
        """ Creates the components of the first row. """
        # Creates the area allowing to specify the number of images to generate
        self.components["num_images"]: gr.Number = gr.Number(
            label="Number of images",
            value=1,
            interactive=True
        )

        # Creates the area allowing to specify the seed of the randomness
        self.components["seed"]: gr.Number = gr.Number(
            label="Seed of the randomness",
            value=-1,
            interactive=True
        )

    def _create_row_2(self):
        """ Creates the components of the second row. """
        # Creates the slider allowing to specify the guidance scale
        self.components["guidance_scale"]: gr.Slider = gr.Slider(
            label="Guidance scale",
            minimum=0.0, maximum=21.0,
            value=7.5,
            step=0.1,
            interactive=True
        )

        # Creates the slider allowing to specify the number of denoising steps
        self.components["num_steps"]: gr.Slider = gr.Slider(
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
        return list(self.components.values())
