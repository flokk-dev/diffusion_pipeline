"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: UI
import streamlit as st

# IMPORT: data processing
import numpy as np

# IMPORT: project
import utils

from src.app.component import Page, Component
from src.image_utils.image import ImageToProcess
from src.app.component import ImageUploader


class ImageProcessing(Page):
    """ Represents an ImageProcessing. """
    def __init__(
        self,
        parent
    ):
        """ Initializes an ImageProcessing. """
        # ----- Mother class ----- #
        super(ImageProcessing, self).__init__(id_="image_processing", parent=parent)

        # ----- Session state ----- #
        if "images" not in self.session_state:
            self.session_state["images"] = list()

        if "image_idx" not in self.session_state:
            self.session_state["image_idx"] = 0

        # ----- Components ----- #
        # Image carousel
        ImageCarousel(page=self, parent=self.parent)

        cols = self.parent.columns((0.5, 0.5))
        # Col n°1
        ImageUploader(page=self, parent=cols[0], image_type=ImageToProcess)

        # Col n°2
        ProcessingSelector(page=self, parent=cols[1])


class ImageCarousel(Component):
    """ Represents an ImageCarousel. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator
    ):
        """
        Initializes an ImageCarousel.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        # ----- Mother class ----- #
        super(ImageCarousel, self).__init__(page=page, parent=parent)

        # ----- Components ----- #
        # Verifies that there is uploaded images
        if len(self.session_state["images"]) > 0:
            image = self.session_state["images"][self.session_state["image_idx"]].image
            image_name = self.session_state["images"][self.session_state["image_idx"]].name

            mask = utils.resize_to_shape(
                image=self.session_state["images"][self.session_state["image_idx"]].mask,
                shape=image.shape
            )
            if self.session_state["images"][self.session_state["image_idx"]].process_id == "":
                mask_name = "empty_mask"
            else:
                mask_name = "mask"

        else:
            image = np.zeros((480, 640, 3))
            image_name = "empty image"

            mask = np.zeros_like(image)
            mask_name = "empty_mask"

        # Images
        with self.parent.expander("", expanded=True):
            cols = st.columns((0.5, 0.5))

            # Image to process
            cols[0].image(image=image, caption=image_name, use_column_width=True)

            # Processed image
            cols[1].image(image=mask, caption=mask_name, use_column_width=True)

            # Slider
            if len(self.session_state["images"]) > 1:
                st.slider(
                    label="slider", label_visibility="collapsed",
                    key=f"{self.page.id}_slider",
                    min_value=0, max_value=len(self.session_state["images"]) - 1,
                    value=self.session_state["image_idx"],
                    on_change=self.on_change
                )

    def on_change(self):
        self.session_state["image_idx"] = st.session_state[f"{self.page.id}_slider"]


class ProcessingSelector(Component):
    """ Represents an ProcessingSelector. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator
    ):
        """
        Initializes an ProcessingSelector.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        # ----- Mother class ----- #
        super(ProcessingSelector, self).__init__(page=page, parent=parent)

        # ----- Components ----- #
        options = list(st.session_state.backend.image_processing_manager.keys())

        # Verifies that there is uploaded images
        if len(self.session_state["images"]) > 0:
            process_id = self.session_state["images"][self.session_state["image_idx"]].process_id
            index = 0 if process_id == "" else options.index(process_id)
        else:
            index = 0

        with self.parent.form(key=f"{self.page.id}_form"):
            # Selects the processing
            st.selectbox(
                label="selectbox", label_visibility="collapsed",
                key=f"{self.page.id}_selectbox",
                options=options,
                index=index
            )

            # Applies the processing
            st.form_submit_button(
                label="Apply",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_click(self):
        if not len(self.session_state["images"]) > 0:
            return

        # Retrieves the processing id
        process_id = st.session_state[f"{self.page.id}_selectbox"]

        # Applies the processing to the current image
        self.session_state["images"][self.session_state["image_idx"]].mask = \
            st.session_state.backend.image_processing_manager(process_id)(
                # The image to process
                image=self.session_state["images"][self.session_state["image_idx"]].image
            )

        # Sets the process id of the current image
        self.session_state["images"][self.session_state["image_idx"]].process_id = process_id
