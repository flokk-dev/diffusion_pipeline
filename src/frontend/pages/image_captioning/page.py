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
from src.frontend.components import Component, ImageUploader

from src.backend.image import Images, ImageToDescribe


class ImageCaptioningPage(Page):
    """ Represents the page allowing to describe (prompt) an image. """
    def __init__(self, parent: st._DeltaGenerator):
        """ Initializes the page allowing to describe (prompt) an image. """
        super(ImageCaptioningPage, self).__init__(parent, page_id="image_captioning")

        # ----- Session state ----- #
        # Creates the list of images to process
        if "images" not in self.session_state:
            self.session_state["images"]: Images = Images(image_type=ImageToDescribe)

        # Creates the idx indicating the current image
        if "image_idx" not in self.session_state:
            self.session_state["image_idx"]: int = 0

        # ----- Components ----- #
        # Writes the purpose of the page
        self.parent.info("This page allows you to generate a prompt that describes an image.")

        cols = self.parent.columns((0.5, 0.5))
        # Instantiates the image carousel and the image describer
        ImageCarousel(page=self, parent=cols[0])
        ImageCaptioner(page=self, parent=cols[0])

        # Instantiates the image uploader
        ImageUploader(page=self, parent=cols[1])


class ImageCarousel(Component):
    """ Represents the image carousel. """
    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes the image carousel.

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
            # Displays the current image
            st.image(image=image.image, caption=image.name, use_column_width=True)

            # Creates the slider allowing to navigate between the uploaded images
            if len(self.session_state["images"]) > 1:
                st.slider(
                    label="slider", label_visibility="collapsed",
                    key=f"{self.page.ID}_slider",
                    min_value=0, max_value=len(self.session_state["images"]) - 1,
                    value=self.session_state["image_idx"],
                    on_change=self.on_change
                )

    def on_change(self):
        # Updates the index of the current image according to the slider value
        self.session_state["image_idx"] = st.session_state[f"{self.page.ID}_slider"]


class ImageCaptioner(Component):
    """ Represents a component allowing to generate a prompt for an image. """
    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes a component allowing to generate a prompt for an image.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(ImageCaptioner, self).__init__(page, parent, component_id="image_captioner")

        # ----- Components ----- #
        with self.parent.form(key=f"{self.page.ID}_form"):
            # Creates the text_area in which to display the prompt
            st.text_area(
                label="text_area", label_visibility="collapsed",
                key=f"{self.page.ID}_text_area",
                value=self.session_state["images"][self.session_state["image_idx"]].caption,
                height=125
            )

            # Creates the button allowing to generate the prompt
            st.form_submit_button(
                label="Describe the image",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_click(self):
        # If no image has been loaded return
        if len(self.session_state["images"]) == 0:
            return

        # Generates the prompt of the current image
        st.session_state.backend.check_clip_interrogator()
        prompt = st.session_state.backend.clip_interrogator(
            image=self.session_state["images"][self.session_state["image_idx"]].image
        )

        # Updates the content of the text area
        st.session_state[f"{self.page_id}_text_area"] = prompt

        # Updates the prompt of the current image
        self.session_state["images"][self.session_state["image_idx"]].prompt = prompt
