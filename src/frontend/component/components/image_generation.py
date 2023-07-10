"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import gradio as gr

import torch

# IMPORT: project
from src.frontend.component import Component


class ImageGeneration(Component):
    """ Allows to generate images. """

    def __init__(self, parent: Any, diffuser_type: Any):
        """
        Allows to generate images.

        Parameters
        ----------
            parent: Any
                parent of the component
            diffuser_type: type
                type of the diffusion pipeline
        """
        super(ImageGeneration, self).__init__(parent)

        # ----- Attributes ----- #
        self.diffuser: Any = None

        self.latents: torch.Tensor = None
        self.args: Dict[str, Any] = dict()

        # ----- Components ----- #
        # Creates the component allowing to generate and display images
        with gr.Accordion(label="Generation", open=True):
            # Creates the carousel containing the generated images
            self.generated_images: gr.Gallery = gr.Gallery(label="Images", columns=3)

            with gr.Row():
                # Creates the list of the available diffusion models
                self.pipeline_id: gr.Dropdown = gr.Dropdown(
                    label="Diffusion model",
                    choices=list(diffuser_type.PIPELINES.keys())
                )

                # Creates the list of the available diffusion models
                try:
                    self.lora_id: gr.Dropdown = gr.Dropdown(
                        label="LoRA layers",
                        choices=[""] + list(diffuser_type.LORA)
                    )
                except AttributeError:
                    pass

                # Creates the button allowing to generate images
                self.button: gr.Button = gr.Button("Generate new images", scale=1)

    def retrieve_info(self) -> List[Any]:
        """
        Retrieves the component information.

        Returns
        ----------
            List[Any]
                info within the component
        """
        try:
            return [
                self.pipeline_id,
                self.lora_id
            ]
        except AttributeError:
            return [
                self.pipeline_id
            ]
