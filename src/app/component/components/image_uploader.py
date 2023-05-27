"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
from copy import deepcopy

# IMPORT: UI
import streamlit as st

# IMPORT: project
from src.app.component import Page, Component


class ImageUploader(Component):
    """ Represents a ImageUploader. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator,
        image_type: type
    ):
        """
        Initializes a ImageUploader.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        # ----- Mother class ----- #
        super(ImageUploader, self).__init__(page=page, parent=parent)

        # ----- Attributes ----- #
        self._image_type = image_type

        # ----- Components ----- #
        # with self.parent.expander(label="", expanded=True):
        self.parent.file_uploader(
            label="file uploader", label_visibility="collapsed",
            key=f"{self.page.id}_file_uploader",
            type=["jpg", "jpeg", "png"],
            on_change=self.on_change,
            accept_multiple_files=True
        )

    def on_change(self):
        # Retrieves the uploaded images in the file uploader
        uploaded_images = st.session_state[f"{self.page.id}_file_uploader"][:3]

        # If there is no more image uploaded
        if not uploaded_images:
            # Reset the memory
            self.session_state["images"] = list()
            return

        # Updates the removed images
        self.update_removed(uploaded_images)

        # Updates the added images
        self.update_added(uploaded_images)

        # Verification
        assert(len(self.session_state["images"]) == len(uploaded_images))

    def update_removed(
        self,
        uploaded_images: List[Any]
    ):
        """
        Updates the removed images.

        Parameters
        ----------
            uploaded_images: List[Any]
                list of the uploaded images
        """
        # Retrieves the id of the image in the file uploader
        uploaded_ids = [file.id for file in uploaded_images]

        # For each image in memory
        for idx, image in enumerate(self.session_state["images"]):
            # If the id isn't anymore in the file_uploader
            if image.id not in uploaded_ids:
                # Remove the image from memory
                del self.session_state["images"][idx]

                # Updates index of the new current image
                if self.session_state["image_idx"] > 0:
                    self.session_state["image_idx"] -= 1

    def update_added(
        self,
        uploaded_images: List[Any]
    ):
        """
        Updates the added images.

        Parameters
        ----------
            uploaded_images: List[Any]
                list of the uploaded images
        """
        # Retrieves the id of the image in memory
        in_memory_ids = [image.id for image in self.session_state["images"]]

        # For each image in the file uploader
        for uploaded_image in uploaded_images:
            # If the id is not already in memory
            if uploaded_image.id not in in_memory_ids:
                # Adds the image in memory
                self.session_state["images"].append(
                    self._image_type(uploaded_image)
                )

                # Updates index of the new current image
                self.session_state["image_idx"] = len(self.session_state["images"]) - 1
