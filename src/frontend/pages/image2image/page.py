"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import gradio as gr

# IMPORT: project
from .sub_pages import Pix2PixSubPage, Image2ImageSubPage, ImageVariationSubPage


class Image2ImagePage:
    """ Represents the page allowing to process images. """
    def __init__(self):
        """ Initializes the page allowing to process images. """
        # ----- Components ----- #
        with gr.Tab("Pix2Pix"):
            Pix2PixSubPage()

        with gr.Tab("Image2Image"):
            Image2ImageSubPage()

        with gr.Tab("ImageVariation"):
            ImageVariationSubPage()
