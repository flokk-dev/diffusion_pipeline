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


class Prompts(Component):
    """ Represents the component allowing to specify the prompt/negative prompt. """
    def __init__(self, parent: Any):
        """
        Initializes the component allowing to specify the prompt/negative prompt.

        Parameters
        ----------
            parent: Any
                parent of the component
        """
        super(Prompts, self).__init__(parent=parent)

        # ----- Attributes ----- #
        # Prompts
        self.prompt: gr.TextArea = None
        self.negative_prompt: gr.TextArea = None

        # ----- Components ----- #
        with gr.Accordion(label="Prompts", open=True):
            with gr.Row():
                # Creates the text area allowing to specify the prompt
                self.prompt = gr.TextArea(
                    label="Prompt",
                    placeholder="Please specify the prompt",
                    lines=3
                )

                # Creates the text area allowing to specify the prompt
                self.negative_prompt = gr.TextArea(
                    label="Negative prompt",
                    value="monochrome, lowres, bad anatomy, worst quality, low quality",
                    placeholder="Please specify the negative prompt",
                    lines=3
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
            self.prompt,
            self.negative_prompt
        ]
