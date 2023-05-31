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
from src.frontend.components import Component


class Prompts(Component):
    """ Represents the sub-page where to specify the prompt and the negative prompt. """

    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes the component where to specify the prompt and the negative prompt.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(Prompts, self).__init__(page, parent, component_id="prompts")
        self.parent.info("Here, you can specify a prompt and a negative prompt (to avoid some key words) that will then guide the generation.")

        # ----- Components ----- #
        # Row n°1
        cols = self.parent.columns([0.5, 0.5])

        Prompt(page=self.page, parent=cols[0])  # where to specify the prompt
        NegativePrompt(page=self.page, parent=cols[1])  # where to specify the negative prompt


class Prompt(Component):
    """ Represents the component where to specify the prompt. """

    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes the component where to specify the prompt.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(Prompt, self).__init__(page, parent, component_id="prompt")

        # ----- Session state ----- #
        # Creates the prompt to use during generation
        if "prompt" not in self.session_state:
            self.session_state["prompt"] = ""

        # ----- Components ----- #
        with self.parent.expander(label="", expanded=True):
            # Creates the text_area allowing to specify the prompt
            st.text_area(
                key=f"{self.page.ID}_{self.ID}_text_area",
                label="text_area", label_visibility="collapsed",
                value=self.session_state["prompt"],
                placeholder="Here, you have to describe the content of the generation",
                height=125,
                on_change=self.on_change
            )

            # Creates the button allowing to improve the prompt
            st.button(
                label="Improve the prompt",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_change(self):
        # Assigns the value of the text_area to the prompt
        self.session_state["prompt"] = st.session_state[f"{self.page.ID}_{self.ID}_text_area"]

    def on_click(self):
        # If the text_area containing the prompt to improve is empty
        if st.session_state[f"{self.page.ID}_prompt"] == "":
            return

        # Improves the prompt
        st.session_state.backend.check_promptist()
        prompt = st.session_state.backend.promptist(
            prompt=st.session_state[f"{self.page.ID}_prompt"]
        )

        # Updates the content of the text area
        self.session_state["prompt"] = prompt


class NegativePrompt(Component):
    """ Represents the component where to specify the negative prompt. """

    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes the component where to specify the negative prompt.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(NegativePrompt, self).__init__(page, parent, component_id="negative_prompt")

        # ----- Session state ----- #
        # Creates the negative prompt to use during generation
        if "negative_prompt" not in self.session_state:
            self.session_state["negative_prompt"] = ""

        # ----- Components ----- #
        with self.parent.expander(label="", expanded=True):
            # Creates the text_area allowing to specify the negative prompt
            st.text_area(
                key=f"{self.page.ID}_{self.ID}_text_area",
                label="text_area", label_visibility="collapsed",
                value=self.session_state["negative_prompt"],
                placeholder="Here, you have to specify what you don't want in your generation (key words)",
                height=125,
                on_change=self.on_change
            )

            # Creates the button allowing to load the default negative prompt
            st.button(
                label="Load default",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_change(self):
        # Assigns the value of the text_area to the negative prompt
        self.session_state["negative_prompt"] = \
            st.session_state[f"{self.page.ID}_{self.ID}_text_area"]

    def on_click(self):
        # Loads the default negative prompt
        self.session_state["negative_prompt"] = \
            "monochrome, lowres, bad anatomy, worst quality, low quality"
