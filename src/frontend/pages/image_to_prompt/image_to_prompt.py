"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import gradio as gr

# IMPORT: utils
from src.backend.text_generation import ClipInterrogator


class Image2PromptPage:
    """ Allows to write a prompt describing an image. """

    def __init__(self):
        """ Allows to write a prompt describing an image. """
        # ----- Components ----- #
        with gr.Row():
            # Creates the component allowing to upload an image
            image = gr.Image(label="Image").style(height=350)

            # Creates the component allowing to display the prompt
            prompt = gr.TextArea(label="Prompt", lines=5)

        # Creates the component allowing to describe the image
        self.describer = ClipInterrogator

        self.button = gr.Button("Describe the image")
        self.button.click(
            fn=self.on_click,
            inputs=[image],
            outputs=[prompt]
        )

    def on_click(self, image):
        if isinstance(self.describer, type):
            self.describer = self.describer()

        return self.describer(image)
