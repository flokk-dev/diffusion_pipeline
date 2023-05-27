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
from src.image_utils.image import Mask
from src.app.component import ImageUploader


class ImageGeneration(Page):
    """ Represents an ImageGeneration. """
    def __init__(
        self,
        parent
    ):
        """ Initializes an ImageGeneration. """
        # ----- Mother class ----- #
        super(ImageGeneration, self).__init__(id_="image_generation", parent=parent)

        # ----- Session state ----- #
        if "images" not in self.session_state:
            self.session_state["images"] = list()

        if "image_idx" not in self.session_state:
            self.session_state["image_idx"] = 0

        # ----- Components ----- #
        with self.parent.expander(label="", expanded=True):
            # Image carousel
            ImageCarousel(page=self, parent=st)

            cols = st.columns((0.5, 0.5))
            # Col n°1
            ImageUploader(page=self, parent=cols[0], image_type=Mask)

            # Col n°2
            # ProcessingSelector(page=self, parent=cols[1])


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
        if not len(self.session_state["images"]) > 0:
            return

        # Masks
        with self.parent.form(key=f"{self.page.id}_form"):
            n = len(self.session_state["images"])
            for mask, col in zip(self.session_state["images"], st.columns([1/n for i in range(n)])):
                # Mask
                col.image(image=mask.image, caption=mask.name, use_column_width=True)

            cols = st.columns((0.5, 0.5))
            # Ranks the images
            cols[0].text_input(
                label="text_input", label_visibility="collapsed",
                key=f"{self.page.id}_text_input"
            )

            # Ranks the images
            cols[1].form_submit_button(
                label="Rank",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_click(self):
        # Retrieves the masks ranking
        ranks = [int(e) for e in st.session_state[f"{self.page.id}_text_input"].split("-")]

        # Applies the ranking on the in memory images
        self.session_state["images"] = [
            self.session_state["images"][rank]
            for rank
            in ranks
        ]

        # Sets the current index according to the modifications
        self.session_state["image_idx"] = ranks.index(self.session_state["image_idx"])
