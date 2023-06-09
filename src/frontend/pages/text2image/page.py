"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import gradio as gr

# IMPORT: project
from .sub_pages import StableDiffusionSubPage, ControlNetSubPage


class Text2ImagePage:
    """ Represents the page allowing to process images. """
    def __init__(self):
        """ Initializes the page allowing to process images. """
        # ----- Components ----- #
        with gr.Tab("Diffusion"):
            StableDiffusionSubPage()

        with gr.Tab("Diffusion + ControlNet"):
            ControlNetSubPage()
