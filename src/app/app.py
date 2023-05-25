"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import PIL
import json

# IMPORT: UI
import streamlit as st

# IMPORT: project
import paths

from src.image_utils import Backend
from src.app.pages import HomePage, ImageProcessingPage, MaskFusion


class App:
    """ Represents an App. """
    _PAGES = {"home": HomePage, "image_processing": ImageProcessingPage}

    def __init__(
            self
    ):
        """ Initializes an App. """
        # ----- Session states ----- #
        st.set_page_config(page_title="Aimpower", page_icon=PIL.Image.open(paths.FAVICON))

        # Backend
        if "backend" not in st.session_state:
            st.session_state.backend = Backend()

        # Languages
        if "language" not in st.session_state:
            st.session_state.language = "english"

        with open(paths.LANGUAGES) as file:
            st.session_state.text = json.load(file)[st.session_state.language]

        # Current page
        if "current_page" not in st.session_state:
            st.session_state.current_page = "home"

        # ----- COMPONENTS ----- #
        # Pages
        st.sidebar.button(
            label="Home",
            on_click=self.on_click, args=("home", )
        )

        with st.container():
            st.sidebar.markdown("---")

            st.sidebar.button(
                label="Image processing",
                on_click=self.on_click, args=("image_processing", )
            )
            st.sidebar.button(
                label="Image captioning",
                on_click=self.on_click, args=("home", )
            )

        # Apply a custom style to the buttons
        st.markdown(
            body="""<style>                
                div.stButton > button {
                    border: None;
                    background-color: #262730;
                }
            </style>""",
            unsafe_allow_html=True
        )

        # Current page
        self._PAGES[st.session_state.current_page]()

    def on_click(
        self,
        page_id: str
    ):
        """
        Parameters
        ----------
            page_id: str
                id of the page to go to
        """
        st.session_state.current_page = page_id
