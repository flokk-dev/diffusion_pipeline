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
from src.frontend.components import Component, ImageUploader

from src.frontend.images import ImageToProcess


class ImageProcessingPage(Page):
    """ Represents the page allowing to process images. """
    def __init__(self, parent: st._DeltaGenerator):
        """ Initializes the page allowing to process images. """
        super(ImageProcessingPage, self).__init__(parent, page_id="image_processing")

        # ----- Session state ----- #
        # Creates the list of images to process
        if "images" not in self.session_state:
            self.session_state["images"]: list = list()

        # Creates the idx indicating the current image
        if "image_idx" not in self.session_state:
            self.session_state["image_idx"]: int = 0

        # ----- Components ----- #
        # Writes the purpose of the page
        self.parent.info(
            "This tool allows you to process an image in order to create a ControlNet input"
        )

        # If at least 1 image have been loaded
        if len(self.session_state["images"]) > 0:
            ImageCarousel(page=self, parent=self.parent)  # instantiates the image carousel

            # Row n°1
            cols = self.parent.columns((0.5, 0.5))

            ImageUploader(ImageToProcess, page=self, parent=cols[0])  # displays the images
            ProcessingApplier(page=self, parent=cols[1])  # allows to select and apply a processing

        else:
            ImageUploader(ImageToProcess, page=self, parent=self.parent)  # displays the images


class ImageCarousel(Component):
    """ Represents the component that displays images. """
    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Represents the component that displays images.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(ImageCarousel, self).__init__(page, parent, component_id="image_carousel")

        # ----- Components ----- #
        # Retrieves the current image
        image = self.session_state["images"][self.session_state["image_idx"]]

        with self.parent.expander("", expanded=True):
            cols = st.columns((0.5, 0.5))

            # Displays the current image
            cols[0].image(image=image.image, caption=image.name, use_column_width=True)

            # Displays the mask of the current image
            cols[1].image(image=image.mask, caption=image.processing, use_column_width=True)

            # Creates the slider allowing to navigate between the uploaded images
            if len(self.session_state["images"]) > 1:
                st.slider(
                    key=f"{self.page.ID}_{self.ID}_slider",
                    label="slider", label_visibility="collapsed",
                    min_value=0,
                    max_value=len(self.session_state["images"]) - 1,
                    value=self.session_state["image_idx"],
                    on_change=self.on_change
                )

    def on_change(self):
        # Updates the index of the current image according to the slider value
        self.session_state["image_idx"] = st.session_state[f"{self.page.ID}_{self.ID}_slider"]


class ProcessingApplier(Component):
    """ Represents the component allowing to select and apply a processing.  """
    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes the component allowing to select and apply a processing.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(ProcessingApplier, self).__init__(page, parent, component_id="processing_applier")

        # ----- Components ----- #
        # Retrieves the processing options
        options = [""] + list(st.session_state.backend.image_processing_manager.keys())

        with self.parent.form(key=f"{self.page.ID}_{self.ID}_form"):
            # Creates the select-box allowing to select the processing to use
            st.selectbox(
                key=f"{self.page.ID}_{self.ID}_select_box",
                label="select box", label_visibility="collapsed",
                options=options,
                index=options.index(
                    self.session_state["images"][self.session_state["image_idx"]].processing
                )
            )

            # Creates the button allowing to run the processing
            st.form_submit_button(
                label="Run the processing",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_click(self):
        # Retrieves the selected processing
        processing = st.session_state[f"{self.page.ID}_{self.ID}_select_box"]

        # If no process has been selected return
        if processing == "":
            st.sidebar.warning(
                "WARNING: you need to select a processing before trying to apply it."
            )
            return

        # Runs the selected processing on the current image
        self.session_state["images"][self.session_state["image_idx"]].mask = \
            st.session_state.backend.image_processing_manager(
                processing_id=processing,
                image=self.session_state["images"][self.session_state["image_idx"]].image
            )

        # Updates the processing of the current image
        self.session_state["images"][self.session_state["image_idx"]].processing = processing
