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
        page_id: str
    ):
        """
        Initializes a Component.

        Parameters
        ----------
            page_id: str
                id of the page
        """
        # ----- Attributes ----- #
        self.id = page_id

        # Copies the st.session_state
        if self.id not in st.session_state:
            st.session_state[self.id] = dict()
        self.session_state = st.session_state[self.id]


class Component(st._DeltaGenerator):
    """ Represents a Component. """
    id = 0

    def __init__(
        self,
        page_id: str
    ):
        """
        Initializes a Component.

        Parameters
        ----------
            page_id: str
                id of the page containing the Component
        """
        # ----- Mother class ----- #
        super(Component, self).__init__()

        # ----- Attributes ----- #
        self.id += 1
        self.page_id = page_id

        self.session_state = st.session_state[page_id]


class SubComponent(st._DeltaGenerator):
    """ Represents a SubComponent. """
    id = 0

    def __init__(
        self,
        parent: st._DeltaGenerator,
        page_id: str
    ):
        """
        Initializes a SubComponent.

        Parameters
        ----------
            parent: st._DeltaGenerator
                container of the SubComponent
            page_id: str
                id of the page containing the Component
        """
        # ----- Mother class ----- #
        super(SubComponent, self).__init__(parent=parent)

        # ----- Attributes ----- #
        self.id += 1
        self.page_id = page_id

        self.parent = parent
        self.session_state = st.session_state[page_id]
