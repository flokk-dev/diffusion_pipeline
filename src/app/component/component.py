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


class Component(st._DeltaGenerator):
    """ Represents a Component. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator
    ):
        """
        Initializes a Component.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        # ----- Mother class ----- #
        super(Component, self).__init__()

        # ----- Attributes ----- #
        self.page = page
        self.session_state = st.session_state[self.page.id]

        # Parent
        self.parent = parent
