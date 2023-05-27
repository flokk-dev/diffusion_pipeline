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
from src.app.component import Page, Component
from src.image_utils.image import ImageToDescribe
from src.app.component import ImageUploader


class ImageCaptioning(Page):
    """ Represents an ImageCaptioning. """
    def __init__(
        self,
        parent
    ):
        """ Initializes an ImageCaptioning. """
        # ----- Mother class ----- #
        super(ImageCaptioning, self).__init__(id_="image_captioning", parent=parent)

        # ----- Session state ----- #
        if "images" not in self.session_state:
            self.session_state["images"] = list()

        if "image_idx" not in self.session_state:
            self.session_state["image_idx"] = 0

        # ----- Components ----- #
        cols = self.parent.columns((0.5, 0.5))

        # Col n°1
        ImageCarousel(page=self, parent=cols[0])
        ImageUploader(page=self, parent=cols[0], image_type=ImageToDescribe)

        # Col n°2
        CaptionGenerator(page=self, parent=cols[1])
        CaptionImprovement(page=self, parent=cols[1])


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
        else:
            image = np.zeros((480, 640, 3))

        # Images
        with self.parent.expander("", expanded=True):
            # Image to process
            st.image(image=image, caption="", use_column_width=True)

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


class CaptionGenerator(Component):
    """ Represents an CaptionGenerator. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator
    ):
        """
        Initializes an CaptionGenerator.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        # ----- Mother class ----- #
        super(CaptionGenerator, self).__init__(page=page, parent=parent)

        # ----- Components ----- #
        # Verifies that there is uploaded images
        if len(self.session_state["images"]) > 0:
            caption = self.session_state["images"][self.session_state["image_idx"]].caption
        else:
            caption = ""

        with self.parent.form(key=f"{self.page.id}_form_0"):
            # Selects the processing
            st.text_area(
                label="text_area", label_visibility="collapsed",
                key=f"{self.page.id}_text_area_0",
                value=caption,
                height=125
            )

            # Applies the processing
            st.form_submit_button(
                label="Describe",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_click(self):
        if not len(self.session_state["images"]) > 0:
            return

        # Generates a caption for the current image
        caption = st.session_state.backend.image_captioning_manager("clip_interrogator")(
            # The image to process
            image=self.session_state["images"][self.session_state["image_idx"]].image
        )

        # Updates the text area
        st.session_state[f"{self.page_id}_text_area_0"] = caption

        # Sets the caption of the current image
        self.session_state["images"][self.session_state["image_idx"]].caption = caption


class CaptionImprovement(Component):
    """ Represents an CaptionImprovement. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator
    ):
        """
        Initializes an CaptionImprovement.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        # ----- Mother class ----- #
        super(CaptionImprovement, self).__init__(page=page, parent=parent)

        # ----- Components ----- #
        # Verifies that there is uploaded images
        if len(self.session_state["images"]) > 0:
            caption = self.session_state["images"][self.session_state["image_idx"]].improved_caption
        else:
            caption = ""

        with self.parent.form(key=f"{self.page.id}_form_1"):
            # Selects the processing
            st.text_area(
                label="text_area", label_visibility="collapsed",
                key=f"{self.page.id}_text_area_1",
                value=caption,
                height=125
            )

            # Applies the processing
            st.form_submit_button(
                label="Improve",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_click(self):
        if st.session_state[f"{self.page.id}_text_area_1"] == "" or \
                not len(self.session_state["images"]) > 0:
            return

        # Improves the current prompt
        prompt = st.session_state.backend.image_captioning_manager("promptist")(
            # The prompt to improve
            prompt=st.session_state[f"{self.page.id}_text_area_1"]
        )

        # Updates the text area
        st.session_state[f"{self.page_id}_text_area_1"] = prompt

        # Sets the caption of the current image
        self.session_state["images"][self.session_state["image_idx"]].improved_caption = prompt
