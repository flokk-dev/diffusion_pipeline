"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import gradio as gr

# IMPORT: utils
from src.backend.deep_learning import Promptist


class Text2PromptPage:
    """ Represents the page allowing to convert a text into a prompt or improve a prompt. """
    def __init__(self):
        """ Initializes the page allowing to convert a text into a prompt or improve a prompt. """
        # ----- Components ----- #
        with gr.Row():
            # Creates the component allowing to upload an image
            text = gr.TextArea(label="Text", lines=3)

            # Creates the component allowing to display the prompt
            prompt = gr.TextArea(label="Prompt", lines=3)

        # Creates the component allowing to describe the image
        self.converter = Promptist

        self.button = gr.Button("Convert the text")
        self.button.click(
            fn=self.on_click,
            inputs=[text],
            outputs=[prompt]
        )

    def on_click(self, text):
        if isinstance(self.converter, type):
            self.converter = self.converter()

        return self.converter(text)
