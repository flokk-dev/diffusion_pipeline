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
from src.frontend.components.component import Component, ImageUploader

from src.backend.image import Images, Mask


class ControlNetSelector(Component):
    """ Represents a ControlNetSelector. """

    def __init__(
            self,
            page: Page,
            parent: st._DeltaGenerator
    ):
        """
        Initializes a ControlNetSelector.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(ControlNetSelector, self).__init__(page=page, parent=parent)
        self.parent.info(
            "Here, you can load some masks that will then guide the generation using ControlNet."
        )

        # ----- Session state ----- #
        if "images" not in self.session_state:
            self.session_state["images"] = Images(image_type=Mask)

        # ----- Components ----- #
        # Row n°1
        MaskDisplayer(page=self.page, parent=self.parent)

        # Row n°2
        cols = self.parent.columns((0.5, 0.5))

        ImageUploader(page=self.page, parent=cols[0])
        MaskRanker(page=self.page, parent=cols[1])


class MaskDisplayer(Component):
    """ Represents a MaskDisplayer. """

    def __init__(
            self,
            page: Page,
            parent: st._DeltaGenerator
    ):
        """
        Initializes a MaskDisplayer.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(MaskDisplayer, self).__init__(page=page, parent=parent)

        # ----- Components ----- #
        # Retrieves the processing options
        options = [""] + list(st.session_state.backend.control_net.CONTROL_NETS_IDS.keys())

        with self.parent.expander(label="", expanded=True):
            # For each in memory image creates a column
            for idx, col in enumerate(st.columns([1 for _ in self.session_state["images"]])):
                # Displays the mask
                col.image(
                    image=self.session_state["images"][idx].image,
                    caption=self.session_state["images"][idx].name,
                    use_column_width=True
                )

                # Creates a selectbox allowing to indicate the processing that gives the mask
                col.selectbox(
                    label="selectbox", label_visibility="collapsed",
                    key=f"{self.page.ID}_processing_{idx}",
                    options=options,
                    index=options.index(self.session_state["images"][idx].processing),
                    on_change=self.on_change_controlnet, args=(idx, )
                )

                # Creates a text_input allowing to indicate the weight of the mask
                col.text_input(
                    label="weight", label_visibility="collapsed",
                    key=f"{self.page.ID}_weight_{idx}",
                    placeholder="Here, you can specify the weight",
                    value=self.session_state["images"][idx].weight,
                    on_change=self.on_change_weight, args=(idx, )
                )

    def on_change_controlnet(self, idx):
        # Updates the processing of the mask at index idx
        self.session_state["images"][idx].processing = \
            st.session_state[f"{self.page.ID}_processing_{idx}"]

    def on_change_weight(self, idx):
        # Updates the weight of the mask at index idx
        self.session_state["images"][idx].weight = \
            float(st.session_state[f"{self.page.ID}_weight_{idx}"])


class MaskRanker(Component):
    """ Represents a MaskRanker. """

    def __init__(
            self,
            page: Page,
            parent: st._DeltaGenerator
    ):
        """
        Initializes a MaskRanker.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(MaskRanker, self).__init__(page=page, parent=parent)

        # ----- Components ----- #
        with self.parent.form(key=f"{self.page.ID}_form_0"):
            # Creates the text_input allowing to specify the ranking of the images
            st.text_input(
                label="text_input", label_visibility="collapsed",
                key=f"{self.page.ID}_text_input",
                placeholder="Here, you can rank the masks by importance (separated by a dash)"
            )

            # Creates the button allowing to rank the images
            st.form_submit_button(
                label="Rank the masks",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_click(self):
        # If no image has been loaded
        if len(self.session_state["images"]) == 0:
            return

        # If no rank has been entered
        if len(st.session_state[f"{self.page.ID}_text_input"]) == 0:
            return

        # If the provided ranking isn't conform
        ranking = [int(idx)-1 for idx in st.session_state[f"{self.page.ID}_text_input"].split("-")]
        if sum(ranking) != sum(range(len(self.session_state["images"]))):
            return

        # Applies the ranking on the in memory images
        new_images = [self.session_state["images"][idx] for idx in ranking]
        for idx in range(len(new_images)):
            self.session_state["images"][idx] = new_images[idx]
