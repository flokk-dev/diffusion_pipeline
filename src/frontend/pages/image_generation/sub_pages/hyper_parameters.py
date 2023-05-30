"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: UI
import streamlit as st

# IMPORT: project
from src.frontend.pages.page import Page
from src.frontend.components.component import Component


class HyperParameters(Component):
    """ Represents a HyperParameters. """

    def __init__(
            self,
            page: Page,
            parent: st._DeltaGenerator
    ):
        """
        Initializes a HyperParameters.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(HyperParameters, self).__init__(page=page, parent=parent)
        self.parent.info(
            "Here, you can adjust the hyper-parameters of StableDiffusion in order to "
            "modify the generation."
        )

        # ----- Components ----- #
        # Num images
        self.parent.text_input(
            label="number of images to generate",
            value=1,
            key=f"{self.page.ID}_num_images"
        )

        # Seed
        self.parent.text_input(
            label="seed of the randomness",
            value=-1,
            key=f"{self.page.ID}_seed"
        )

        cols = self.parent.columns([0.5, 0.5])

        # Col n°1
        cols[0].text_input(
            label="width of the image to generate",
            value=512,
            key=f"{self.page.ID}_width"
        )

        # Col n°2
        cols[1].text_input(
            label="height of the image to generate",
            value=512,
            key=f"{self.page.ID}_height"
        )

        # Guidance scale
        cols[0].slider(
            label="guidance scale of the generation",
            min_value=0.0, max_value=21.0, value=7.5, step=0.1,
            key=f"{self.page.ID}_guidance_scale"
        )

        # Sampling steps
        cols[0].slider(
            label="number of denoising steps",
            min_value=0, max_value=100, value=20,
            key=f"{self.page.ID}_num_steps"
        )