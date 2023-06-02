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

from src.backend.image import Images, Mask


class ControlNet(Component):
    """ Represents the sub-page allowing to upload the ControlNet masks. """

    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes the sub-page allowing to upload the ControlNet masks.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(ControlNet, self).__init__(page, parent, component_id="controlnet_selector")
        self.parent.info(
            "Here, you can upload ControlNet inputs in order to guide the generation"
        )

        # ----- Session state ----- #
        # Creates the list of ControlNet masks
        if "images" not in self.session_state:
            self.session_state["images"] = Images(image_type=Mask)

        # ----- Components ----- #
        # Row n°1
        MaskDisplayer(page=self.page, parent=self.parent)  # displays the uploaded ControlNet masks

        # Row n°2
        cols = self.parent.columns((0.5, 0.5))

        ImageUploader(page=self.page, parent=cols[0])  # allows to upload images
        MaskRanker(page=self.page, parent=cols[1])  # allows to rank ControlNet masks


class MaskDisplayer(Component):
    """ Represents the component that displays the ControlNet masks. """

    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes the component that displays the ControlNet masks.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(MaskDisplayer, self).__init__(page, parent, component_id="mask_displayer")

        # ----- Components ----- #
        # Retrieves the processing options
        options = [""] + list(st.session_state.backend.controlnet.CONTROLNET_IDS.keys())

        with self.parent.expander(label="", expanded=True):
            # For each in memory image creates a column
            for idx, col in enumerate(st.columns([1 for _ in self.session_state["images"]])):
                # Retrieves the current image
                image = self.session_state["images"][idx]

                # Displays the mask
                col.image(image=image.image, caption=image.name, use_column_width=True)

                # Creates a select box allowing to indicate the processing that gives the mask
                col.selectbox(
                    key=f"{self.page.ID}_{self.ID}_select_box_{idx}",
                    label="select box", label_visibility="collapsed",
                    options=options,
                    index=options.index(image.processing),
                    on_change=self.on_change_controlnet, args=(idx, )
                )

                # Creates a text_input allowing to indicate the weight of the mask
                col.text_input(
                    key=f"{self.page.ID}_{self.ID}_text_input{idx}",
                    label="weight", label_visibility="collapsed",
                    value=image.weight,
                    placeholder="Here, you can specify the weight",
                    on_change=self.on_change_weight, args=(idx, )
                )

    def on_change_controlnet(self, idx):
        # Updates the processing of the mask at index idx
        self.session_state["images"][idx].processing = \
            st.session_state[f"{self.page.ID}_{self.ID}_select_box_{idx}"]

    def on_change_weight(self, idx):
        # Updates the weight of the mask at index idx
        self.session_state["images"][idx].weight = \
            float(st.session_state[f"{self.page.ID}_{self.ID}_text_input{idx}"])


class MaskRanker(Component):
    """ Represents the component allowing to rank the ControlNet masks. """

    def __init__(
            self,
            page: Page,
            parent: st._DeltaGenerator
    ):
        """
        Initializes the component allowing to rank the ControlNet masks.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(MaskRanker, self).__init__(page, parent, component_id="mask_ranker")

        # ----- Components ----- #
        with self.parent.form(key=f"{self.page.ID}_{self.ID}_form"):
            # Creates the text_input allowing to specify the ranking of the images
            st.text_input(
                key=f"{self.page.ID}_{self.ID}_text_input",
                label="text_input", label_visibility="collapsed",
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
        if len(self.session_state["images"]) <= 1:
            st.sidebar.warning(
                "WARNING: you need to provide at least 2 masks before ranking them."
            )
            return

        # If no rank has been entered
        ranking = st.session_state[f"{self.page.ID}_{self.ID}_text_input"]
        if len(ranking) == 0:
            st.sidebar.warning(
                "WARNING: you need to provide a ranking before trying to rank the masks."
            )
            return

        # If the provided ranking isn't conform
        ranking = [int(idx)-1 for idx in ranking]
        if sum(ranking) != sum(range(len(self.session_state["images"]))):
            st.sidebar.warning(
                "WARNING: there is something wrong with the ranking you provided."
            )
            return

        # Applies the ranking on the in memory images
        new_images = [self.session_state["images"][idx] for idx in ranking]
        for idx in range(len(new_images)):
            self.session_state["images"][idx] = new_images[idx]
