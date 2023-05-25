"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: UI
import streamlit as st

# IMPORT: project
from src.app.component import Component, SubComponent


class MaskFusion:
    """ Represents a MaskFusion. """
    def __init__(
        self
    ):
        """ Initializes a MaskFusion. """
        # ----- Mother class ----- #
        super(MaskFusion, self).__init__()

        # ----- Components ----- #
        if len(st.session_state.images) > 1:
            st.markdown("---")
            st.subheader(st.session_state.textual_content["mask_fusion"]["title"])
            st.markdown(st.session_state.textual_content["mask_fusion"]["description"])

            MaskDisplayer()
            MaskFusionOptions()


class MaskDisplayer(Component):
    """ Represents an MaskDisplayer. """
    def __init__(
        self
    ):
        """ Initializes an MaskDisplayer. """
        # ----- Mother class ----- #
        super(MaskDisplayer, self).__init__()

        # ----- Components ----- #
        n = len(st.session_state.images)
        for content, col in zip(st.session_state.images, self.columns([1/n for _ in range(n)])):
            col.image(
                image=content["mask"],
                caption=content["name"],
                use_column_width=True
            )


class MaskFusionOptions(Component):
    """ Represents an MaskFusionOptions. """
    def __init__(
        self
    ):
        """ Initializes an MaskFusionOptions. """
        # ----- Mother class ----- #
        super(MaskFusionOptions, self).__init__()

        # ----- Components ----- #
        cols = self.columns((0.5, 0.5))

        MaskRanking(parent=cols[0])
        ApplyButton(parent=cols[1])

        # Apply a custom style to the buttons
        self.markdown(
            body="""<style>                
                div.stButton > button {
                    height: 40.5px;
                    border: None;
                    background-color: #262730;
                }
            </style>""",
            unsafe_allow_html=True
        )


class MaskRanking(SubComponent):
    """ Represents a MaskRanking. """
    def __init__(
            self,
            parent: st._DeltaGenerator
    ):
        """
        Initializes a MaskRanking.

        Parameters
        ----------
            parent: st._DeltaGenerator
                container of the MaskRanking
        """
        # ----- Mother class ----- #
        super(MaskRanking, self).__init__(parent, component_type="text_input")

        # ----- Components ----- #
        self.parent.text_input(
            label="", label_visibility="collapsed",
            value="",
            key="mask_ranking"
        )


class ApplyButton(SubComponent):
    """ Represents an ApplyButton. """
    def __init__(
            self,
            parent: st._DeltaGenerator
    ):
        """
        Initializes an ApplyButton.

        Parameters
        ----------
            parent: st._DeltaGenerator
                container of the ApplyButton
        """
        # ----- Mother class ----- #
        super(ApplyButton, self).__init__(parent, component_type="apply_button")

        # ----- Components ----- #
        self.parent.button(
            label="apply",
            on_click=self.on_click,
            use_container_width=True,
            key=f"{self.type}_{self.parent.id}_{self.id}"
        )

    @staticmethod
    def on_click():
        # Retrieves the masks ranking
        ranks = [int(e) for e in st.session_state.mask_ranking.split("-")]

        # Applies the ranking on the in memory images
        st.session_state.images = [
            st.session_state.images[rank]
            for rank
            in ranks
        ]

        # Sets the current index according to the modifications
        st.session_state.image_idx = ranks.index(st.session_state.image_idx)
