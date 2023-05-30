"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import pyautogui
import json

import PIL

# IMPORT: UI
import streamlit as st

# IMPORT: project
import paths

from src.backend import Backend
from src.frontend.pages import ProcessingPage, CaptioningPage, GenerationPage


class App:
    """ Represents an App. """
    _PAGES = {
        "IMAGE PROCESSING": ProcessingPage,
        "IMAGE CAPTIONING": CaptioningPage,
        "IMAGE GENERATION": GenerationPage
    }

    def __init__(
            self
    ):
        """ Initializes an App. """
        st.set_page_config(
            page_title="Aimpower",
            page_icon=PIL.Image.open(paths.FAVICON),
            layout="wide"
        )

        # ----- Session states ----- #
        # Languages
        if "language" not in st.session_state:
            st.session_state.language = "english"

        with open(paths.LANGUAGES) as file:
            st.session_state.text = json.load(file)[st.session_state.language]

        # Backend
        if "backend" not in st.session_state:
            st.session_state.backend = Backend()

        # ----- COMPONENTS ----- #
        w, _ = pyautogui.size()
        middle, side = 1750 / w, (1 - (1750 / w)) / 2

        # Application
        _, app_container, _ = st.columns((0.1, 0.8, 0.1))
        for page, tab in zip(self._PAGES.values(), app_container.tabs(self._PAGES.keys())):
            page(parent=tab)

        # Style
        st.markdown(
            body="""<style>
                div.stButton > button {
                    border-color: #FF4B4B;
                    color: #FF4B4B;
                },
                
                div.stButton > button:hover {
                    border-color: #4BFF4B;
                    color: #4BFF4B;
                }
            </style>""",
            unsafe_allow_html=True
        )
