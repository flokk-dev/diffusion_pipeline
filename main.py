"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import gradio as gr

# IMPORT: project
from paths import FAVICON
from src.frontend import Text2ImagePage, Image2ImagePage, Image2PromptPage, Text2PromptPage


PAGES = {
    "Text2Image": Text2ImagePage,
    "Image2Image": Image2ImagePage,
    "Image2Prompt": Image2PromptPage,
    "Text2Prompt": Text2PromptPage
}

with gr.Blocks() as demo:
    # ----- Components ----- #
    # Instantiates the pages of the application
    for page_id, page in PAGES.items():
        with gr.Tab(page_id):
            page()

if __name__ == "__main__":
    demo.launch(favicon_path=FAVICON, server_port=8080)
