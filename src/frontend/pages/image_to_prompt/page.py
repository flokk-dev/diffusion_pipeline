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

from src.backend.image import Images, ImageToDescribe


class ImageToPromptPage(Page):
    """ Represents the page allowing transform images into prompts. """
    def __init__(self, parent: st._DeltaGenerator):
        """ Initializes the page allowing transform images into prompts. """
        super(ImageToPromptPage, self).__init__(parent, page_id="image_captioning")

        # ----- Session state ----- #
        # Creates the list of images to process
        if "images" not in self.session_state:
            self.session_state["images"]: Images = Images(image_type=ImageToDescribe)

        # Creates the idx indicating the current image
        if "image_idx" not in self.session_state:
            self.session_state["image_idx"]: int = 0

        # ----- Components ----- #
        # Writes the purpose of the page
        self.parent.info("This page allows you to transform images into prompts.")

        # Row n°1
        cols = self.parent.columns((0.5, 0.5))

        ImageCarousel(page=self, parent=cols[0])  # displays the uploaded images
        ImageToPrompt(page=self, parent=cols[0])  # allowing to transform images into prompts

        # Row n°2
        ImageUploader(page=self, parent=cols[1])  # allowing to upload images


class ImageCarousel(Component):
    """ Represents the component that displays the uploaded images. """
    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes the component that displays the uploaded images.

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


class ImageToPrompt(Component):
    """ Represents a component allowing to transform the image into a prompt. """
    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes a component allowing to transform the image into a prompt.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(ImageToPrompt, self).__init__(page, parent, component_id="image_to_prompt")

        # ----- Components ----- #
        with self.parent.form(key=f"{self.page.ID}_form"):
            # Creates the text_area in which to display the prompt
            st.text_area(
                key=f"{self.page.ID}_{self.ID}_prompt",
                label="text_area", label_visibility="collapsed",
                value=self.session_state["images"][self.session_state["image_idx"]].prompt,
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
        st.session_state[f"{self.page.ID}_{self.ID}_prompt"] = prompt

        # Updates the prompt of the current image
        self.session_state["images"][self.session_state["image_idx"]].prompt = prompt
