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
from src.image_utils.image import Images, ImageToProcess
from src.app.component import ImageUploader


class ImageProcessing(Page):
    """ Represents an ImageProcessing. """
    def __init__(
        self,
        parent
    ):
        """ Initializes an ImageProcessing. """
        super(ImageProcessing, self).__init__(id_="image_processing", parent=parent)

        # ----- Session state ----- #
        if "images" not in self.session_state:
            self.session_state["images"] = Images(image_type=ImageToProcess)

        if "image_idx" not in self.session_state:
            self.session_state["image_idx"] = 0

        # ----- Components ----- #
        # Row n°1
        ImageCarousel(page=self, parent=self.parent)

        # Row n°2
        cols = self.parent.columns((0.5, 0.5))

        ImageUploader(page=self, parent=cols[0])
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
        super(ImageCarousel, self).__init__(page=page, parent=parent)

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
                    label="slider", label_visibility="collapsed",
                    key=f"{self.page.id}_slider",
                    min_value=0, max_value=len(self.session_state["images"]) - 1,
                    value=self.session_state["image_idx"],
                    on_change=self.on_change
                )

    def on_change(self):
        # Change the index of the current image according to the slider value
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
        super(ProcessingSelector, self).__init__(page=page, parent=parent)

        # ----- Components ----- #
        # Retrieves the processing options
        options = [""] + list(st.session_state.backend.image_processing_manager.keys())

        with self.parent.form(key=f"{self.page.id}_form"):
            # Creates the selectbox allowing to select a processing
            st.selectbox(
                label="selectbox", label_visibility="collapsed",
                key=f"{self.page.id}_selectbox",
                options=options,
                index=options.index(
                    self.session_state["images"][self.session_state["image_idx"]].processing
                )
            )

            # Creates the button allowing to apply the processing
            st.form_submit_button(
                label="Apply the processing",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_click(self):
        # If no image has been loaded
        if len(self.session_state["images"]) == 0:
            return

        # Retrieves the selected processing
        processing = st.session_state[f"{self.page.id}_selectbox"]

        # If no process has been selected
        if processing == "":
            return

        # Applies the processing to the current image
        self.session_state["images"][self.session_state["image_idx"]].mask = \
            st.session_state.backend.image_processing_manager(processing)(
                image=self.session_state["images"][self.session_state["image_idx"]].image
            )

        # Updates the processing of the current image
        self.session_state["images"][self.session_state["image_idx"]].processing = processing
