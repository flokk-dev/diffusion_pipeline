"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: project
from src.frontend.pages.page import Page
from .components import ControlNetSelector, Prompts, HyperParameters, ImageGeneration


class GenerationPage(Page):
    """ Represents an GenerationPage. """

    def __init__(
            self,
            parent
    ):
        """ Initializes an GenerationPage. """
        super(GenerationPage, self).__init__(id_="image_generation", parent=parent)

        # ----- Components ----- #
        tabs = self.parent.tabs([f"Step n°{i+1}" for i in range(3)] + ["Generation"])

        ControlNetSelector(page=self, parent=tabs[0])
        Prompts(page=self, parent=tabs[1])
        HyperParameters(page=self, parent=tabs[2])
        ImageGeneration(page=self, parent=tabs[3])
