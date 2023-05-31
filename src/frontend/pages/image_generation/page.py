"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import streamlit as st

# IMPORT: project
from src.frontend.pages import Page
from .sub_pages import ControlNetSelector, Prompts, HyperParameters, ImageGeneration


class ImageGenerationPage(Page):
    """ Represents the page allowing to generate images. """

    def __init__(self, parent: st._DeltaGenerator):
        """ Initializes the page allowing to generate images. """
        super(ImageGenerationPage, self).__init__(parent, page_id="image_generation")

        # ----- Components ----- #
        tabs = self.parent.tabs(["ControlNet", "Prompt", "Hyper-parameters", "Generation"])

        # Instantiates the sub-page allowing to upload ControlNet inputs
        ControlNetSelector(page=self, parent=tabs[0])

        # Instantiates the sub-page allowing to specify the prompt/negative prompt
        Prompts(page=self, parent=tabs[1])

        # Instantiates the sub-page allowing to adjust the hyperparameters
        HyperParameters(page=self, parent=tabs[2])

        # Instantiates the sub-page allowing to generate images
        ImageGeneration(page=self, parent=tabs[3])
