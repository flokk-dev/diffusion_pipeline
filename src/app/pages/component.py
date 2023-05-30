"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: UI
import streamlit as st

# IMPORT: project
from .page import Page


class Component(st._DeltaGenerator):
    """ Represents a Component. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator
    ):
        """
        Initializes a Component.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(Component, self).__init__()

        # ----- Attributes ----- #
        self.page = page
        self.session_state = st.session_state[self.page.id]

        # Parent
        self.parent = parent


class ImageUploader(Component):
    """ Represents an ImageUploader. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator
    ):
        """
        Initializes an ImageUploader.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(ImageUploader, self).__init__(page=page, parent=parent)

        # ----- Components ----- #
        self.parent.file_uploader(
            label="file uploader", label_visibility="collapsed",
            key=f"{self.page.id}_file_uploader",
            type=["jpg", "jpeg", "png"],
            on_change=self.on_change,
            accept_multiple_files=True
        )

    def on_change(self):
        # Retrieves the uploaded files
        uploaded_files = st.session_state[f"{self.page.id}_file_uploader"]

        # For each image in memory
        for idx, image in enumerate(self.session_state["images"]):
            # If the index of the image is less than the number of uploaded files
            if idx <= len(uploaded_files) - 1:
                # Load the corresponding uploaded file
                image.load(uploaded_files[idx])
            else:
                # Reset the image
                image.reset()

        # Updates the idx of the current image
        self.session_state["image_idx"] = len(self.session_state["images"]) - 1
