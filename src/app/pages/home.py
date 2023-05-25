"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: UI
import streamlit as st

# IMPORT: project
from src.app.component import Page


class HomePage(Page):
    """ Represents a HomePage. """
    def __init__(
            self
    ):
        """ Initializes a HomePage. """
        # ----- Mother class ----- #
        super(HomePage, self).__init__(page_id="home_page")

        # ----- COMPONENTS ----- #
        # Header
        st.markdown(
            f"<h1 style='text-align: center;'> {st.session_state.text[self.id]['title']} </h1>",
            unsafe_allow_html=True
        )
        st.markdown(st.session_state.text[self.id]["description"])
