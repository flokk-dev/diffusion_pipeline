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


class Component:
    """ Represents a page's component. """
    def __init__(self, page: Page, parent: st._DeltaGenerator, component_id: str):
        """
        Initializes a page's component.

        Parameters
        ----------
            page: Page
                page containing the component
            parent: st._DeltaGenerator
                parent of the component
            component_id: str
                unique id of the component
        """
        # ----- Attributes ----- #
        self.ID: str = component_id

        # Page containing the component
        self.page: Page = page

        # Parent of the component (container)
        self.parent: st._DeltaGenerator = parent

    @property
    def session_state(self) -> dict:
        return self.page.session_state
