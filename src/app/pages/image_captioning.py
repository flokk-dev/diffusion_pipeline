"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: UI
import streamlit as st

# IMPORT: project
from src.app.component import Page, Component
from src.image_utils.image import Images, ImageToDescribe
from src.app.component import ImageUploader


class ImageCaptioning(Page):
    """ Represents an ImageCaptioning. """
    def __init__(
        self,
        parent
    ):
        """ Initializes an ImageCaptioning. """
        super(ImageCaptioning, self).__init__(id_="image_captioning", parent=parent)

        # ----- Session state ----- #
        if "images" not in self.session_state:
            self.session_state["images"] = Images(image_type=ImageToDescribe)

        if "image_idx" not in self.session_state:
            self.session_state["image_idx"] = 0

        # ----- Components ----- #
        cols = self.parent.columns((0.5, 0.5))

        # Col n°1
        ImageCarousel(page=self, parent=cols[0])
        ImageUploader(page=self, parent=cols[1])

        # Col n°2
        CaptionGenerator(page=self, parent=cols[0])


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
        super(ImageCarousel, self).__init__(page=page, parent=parent)

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
                    key=f"{self.page.id}_slider",
                    min_value=0, max_value=len(self.session_state["images"]) - 1,
                    value=self.session_state["image_idx"],
                    on_change=self.on_change
                )

    def on_change(self):
        # Change the index of the current image according to the slider value
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
        super(CaptionGenerator, self).__init__(page=page, parent=parent)

        # ----- Components ----- #
        with self.parent.form(key=f"{self.page.id}_form"):
            # Creates the text_area in which to display the caption of the current image
            st.text_area(
                label="text_area", label_visibility="collapsed",
                key=f"{self.page.id}_text_area",
                value=self.session_state["images"][self.session_state["image_idx"]].caption,
                height=125
            )

            # Creates the button allowing to generate the caption
            st.form_submit_button(
                label="Describe the image",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_click(self):
        # If no image has been loaded
        if len(self.session_state["images"]) == 0:
            return

        # Generates a caption for the current image
        caption = st.session_state.backend.image_captioning_manager("clip_interrogator")(
            image=self.session_state["images"][self.session_state["image_idx"]].image
        )

        # Updates the content of the text area
        st.session_state[f"{self.page_id}_text_area"] = caption

        # Updates the caption of the current image
        self.session_state["images"][self.session_state["image_idx"]].caption = caption
