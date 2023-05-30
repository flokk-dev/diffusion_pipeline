"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import streamlit as st

# IMPORT: project
from src.frontend.pages.page import Page
from src.frontend.components import Component, ImageUploader

from src.backend.image import Images, ImageToProcess


class ImageProcessingPage(Page):
    """ Represents the page allowing to process images. """
    def __init__(self, parent: st._DeltaGenerator):
        """ Initializes the page allowing to process images. """
        super(ImageProcessingPage, self).__init__(parent, page_id="image_processing")

        # ----- Session state ----- #
        # Creates the list of images to process
        if "images" not in self.session_state:
            self.session_state["images"]: Images = Images(image_type=ImageToProcess)

        # Creates the idx indicating the current image
        if "image_idx" not in self.session_state:
            self.session_state["image_idx"]: int = 0

        # ----- Components ----- #
        # Writes the purpose of the page
        self.parent.info("This tool allows you to process an image in order to create an input for a ControlNet.")

        # Instantiates the image carousel
        ImageCarousel(page=self, parent=self.parent)

        # Instantiates the image uploader and the processing selector
        cols = self.parent.columns((0.5, 0.5))

        ImageUploader(page=self, parent=cols[0])
        ProcessingSelector(page=self, parent=cols[1])


class ImageCarousel(Component):
    """ Represents the image carousel. """
    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Represents the image carousel.

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
                    key=f"{self.page.ID}_slider",
                    label="slider",
                    label_visibility="collapsed",
                    min_value=0,
                    max_value=len(self.session_state["images"]) - 1,
                    value=self.session_state["image_idx"],
                    on_change=self.on_change
                )

    def on_change(self):
        # Updates the index of the current image according to the slider value
        self.session_state["image_idx"] = st.session_state[f"{self.page.ID}_slider"]


class ProcessingSelector(Component):
    """ Represents a processing selector. """
    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes a processing selector.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(ProcessingSelector, self).__init__(page, parent, component_id="processing_selector")

        # ----- Components ----- #
        # Retrieves the processing options
        options = [""] + list(st.session_state.backend.image_processing_manager.keys())

        with self.parent.form(key=f"{self.page.ID}_form"):
            # Creates the select-box allowing to select the processing to use
            st.selectbox(
                label="selectbox", label_visibility="collapsed",
                key=f"{self.page.ID}_selectbox",
                options=options,
                index=options.index(
                    self.session_state["images"][self.session_state["image_idx"]].processing
                )
            )

            # Creates the button allowing to run the processing
            st.form_submit_button(
                label="Apply the processing",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_click(self):
        # If no image has been loaded return
        if len(self.session_state["images"]) == 0:
            return

        # Retrieves the selected processing
        processing = st.session_state[f"{self.page.ID}_selectbox"]

        # If no process has been selected return
        if processing == "":
            return

        # Runs the selected processing on the current image
        self.session_state["images"][self.session_state["image_idx"]].mask = \
            st.session_state.backend.image_processing_manager(processing)(
                image=self.session_state["images"][self.session_state["image_idx"]].image
            )

        # Updates the processing of the current image
        self.session_state["images"][self.session_state["image_idx"]].processing = processing
