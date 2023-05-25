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
from src.app.pages import HomePage, ImageProcessingPage, ImageCaptioningPage, ImageGenerationPage


class App:
    """ Represents an App. """
    _PAGES = {
        "home": HomePage,
        "image_processing": ImageProcessingPage,
        "image_captioning": ImageCaptioningPage,
        "image_generation": ImageGenerationPage
    }

    def __init__(
            self
    ):
        """ Initializes an App. """
        st.set_page_config(page_title="Aimpower", page_icon=PIL.Image.open(paths.FAVICON))

        # ----- Session states ----- #
        # Languages
        if "language" not in st.session_state:
            st.session_state.language = "english"

        with open(paths.LANGUAGES) as file:
            st.session_state.text = json.load(file)[st.session_state.language]

        # Backend
        if "backend" not in st.session_state:
            st.session_state.backend = Backend()

        # Current page
        if "current_page" not in st.session_state:
            st.session_state.current_page = "home"

        # ----- COMPONENTS ----- #
        # Current page
        self._PAGES[st.session_state.current_page]()

        # Sidebar
        Sidebar()


class Sidebar:
    """ Represents a Sidebar. """
    def __init__(
            self
    ):
        """ Initializes a Sidebar. """
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
                on_click=self.on_click, args=("image_captioning", )
            )

            st.sidebar.button(
                label="Image generation",
                on_click=self.on_click, args=("image_generation", )
            )

        # Apply a custom style to the buttons
        st.markdown(
            body="""<style>                
                div.stButton > button {
                    border: None;
                    background-color: #262730;
                },
                div.stButton > button: hover {
                    border: None;
                    background-color: #ffffff;
                }
            </style>""",
            unsafe_allow_html=True
        )

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
