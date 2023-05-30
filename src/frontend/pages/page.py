"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import streamlit as st


class Page:
    """ Represents a page of the application. """
    def __init__(self, parent: st._DeltaGenerator, page_id: str):
        """
        Initializes a page of the application.

        Parameters
        ----------
            parent: st._DeltaGenerator
                parent of the page
            page_id: str
                unique id of the page
        """
        # ----- Attributes ----- #
        self.ID: str = page_id

        # Parent of the page (container)
        self.parent: st._DeltaGenerator = parent

        # ----- Session state ----- #
        # Stores the session_state of the page directly in it
        if self.ID not in st.session_state:
            st.session_state[self.ID] = dict()
        self.session_state: dict = st.session_state[self.ID]
