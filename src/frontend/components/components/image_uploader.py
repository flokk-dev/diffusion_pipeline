"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import streamlit as st

# IMPORT: project
from src.frontend.components.component import Component
from src.frontend.pages.page import Page


class ImageUploader(Component):
    """ Represents an image uploader. """
    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes an image uploader.

        Parameters
        ----------
            page: Page
                page containing the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(ImageUploader, self).__init__(page, parent, component_id="image_uploader")

        # ----- Components ----- #
        self.parent.file_uploader(
            key=f"{self.page.ID}_{self.ID}",
            label="image uploader",
            label_visibility="collapsed",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            on_change=self.on_change
        )

    def on_change(self):
        # Retrieves the uploaded files
        uploaded_files: List[Any] = st.session_state[f"{self.page.ID}_{self.ID}"]

        # For each in memory image
        for idx, image in enumerate(self.session_state["images"]):
            # If the index of the image is less than the number of uploaded files
            if idx <= len(uploaded_files) - 1:
                # Loads the corresponding uploaded file
                image.load(uploaded_files[idx])
            else:
                # Resets the image
                image.reset()

        # Updates the idx of the current image
        self.session_state["image_idx"] = len(self.session_state["images"]) - 1
