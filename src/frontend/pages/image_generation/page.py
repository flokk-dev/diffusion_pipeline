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
from .sub_pages import ControlNet, ImageGeneration, FeedbackPage


class ImageGenerationPage(Page):
    """ Represents the page allowing to generate images. """

    def __init__(self, parent: st._DeltaGenerator):
        """ Initializes the page allowing to generate images. """
        super(ImageGenerationPage, self).__init__(parent, page_id="image_generation")

        # ----- Components ----- #
        tabs = self.parent.tabs(["ControlNet", "Generation", "Images"])

        # Instantiates the sub-page allowing to upload ControlNet inputs
        ControlNet(page=self, parent=tabs[0])

        # Instantiates the sub-page allowing to generate images
        ImageGeneration(page=self, parent=tabs[1])

        # Instantiates the sub-page allowing to give its feedback
        FeedbackPage(page=self, parent=tabs[2])
