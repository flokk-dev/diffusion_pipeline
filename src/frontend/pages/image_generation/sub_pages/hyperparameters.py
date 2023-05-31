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
from src.frontend.components import Component


class HyperParameters(Component):
    """ Represents the sub-page allowing to adjust the hyperparameters. """

    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes the sub-page allowing to adjust the hyperparameters.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(HyperParameters, self).__init__(page, parent, component_id="hyperparameters")
        self.parent.info("Here, you can adjust the hyper-parameters of StableDiffusion in order to modify the generation.")

        # ----- Components ----- #
        # Row n°1
        self.parent.text_input(
            label="number of images to generate",
            value=1,
            key=f"{self.page.ID}_num_images"
        )

        # Row n°2
        self.parent.text_input(
            label="seed of the randomness",
            value=-1,
            key=f"{self.page.ID}_seed"
        )

        # Row n°3
        cols = self.parent.columns([0.5, 0.5])

        cols[0].text_input(
            label="width of the image to generate",
            value=512,
            key=f"{self.page.ID}_width"
        )

        cols[1].text_input(
            key=f"{self.page.ID}_height",
            label="height of the image to generate",
            value=512
        )

        # Row n°4
        cols[0].slider(
            label="guidance scale of the generation",
            min_value=0.0, max_value=21.0, value=7.5, step=0.1,
            key=f"{self.page.ID}_guidance_scale"
        )

        # Row n°5
        cols[0].slider(
            label="number of denoising steps",
            min_value=0, max_value=100, value=20,
            key=f"{self.page.ID}_num_steps"
        )
