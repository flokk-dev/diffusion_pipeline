"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: UI
import streamlit as st


class Page:
    """ Represents a Page. """
    def __init__(
        self,
        id_: str,
        parent: st._DeltaGenerator
    ):
        """
        Initializes a Page.

        Parameters
        ----------
            id_: str
                id of the page
            parent: st._DeltaGenerator
                parent of the page
        """
        # ----- Attributes ----- #
        self.id = id_

        # Parent
        self.parent = parent

        # ----- Session state ----- #
        if self.id not in st.session_state:
            st.session_state[self.id] = dict()
        self.session_state = st.session_state[self.id]
