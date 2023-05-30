"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import streamlit as st
from PIL import Image

# IMPORT: project
from paths import FAVICON

from src.backend import Backend
from src.frontend.pages import ImageProcessingPage, ImageCaptioningPage, GenerationPage


class App:
    """ Represents the streamlit application. """
    PAGES = {
        "IMAGE PROCESSING": ImageProcessingPage,
        "IMAGE CAPTIONING": ImageCaptioningPage,
        "IMAGE GENERATION": GenerationPage
    }

    def __init__(self):
        """ Initializes the streamlit application. """
        st.set_page_config(page_title="AIMPower", page_icon=Image.open(FAVICON), layout="wide")

        # ----- Session state ----- #
        # Instantiates the backend of the application
        if "backend" not in st.session_state:
            st.session_state.backend = Backend()

        # ----- Components ----- #
        # Instantiates the pages of the application
        _, app_container, _ = st.columns((0.1, 0.8, 0.1))
        for page, tab in zip(self.PAGES.values(), app_container.tabs(self.PAGES.keys())):
            page(parent=tab)

        # Modifies the style of all the buttons
        st.markdown(
            body="""<style>
                div.stButton > button {
                    border-color: #FF4B4B;
                    color: #FF4B4B;
                }
            </style>""",
            unsafe_allow_html=True
        )
