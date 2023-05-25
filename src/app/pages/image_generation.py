"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: UI
import streamlit as st

# IMPORT: project
import utils

from src.app.component import Page, Component, SubComponent
from src.image_utils.image import ImageToProcess
from src.app.component import ImageUploader


class ImageGenerationPage(Page):
    """ Represents an ImageGenerationPage. """
    def __init__(
        self
    ):
        """ Initializes an ImageGenerationPage. """
        # ----- Mother class ----- #
        super(ImageGenerationPage, self).__init__(page_id="image_generation")

        # ----- Session state ----- #
        if "images" not in self.session_state:
            self.session_state["images"] = list()

        if "image_idx" not in self.session_state:
            self.session_state["image_idx"] = 0

        # ----- Components ----- #
        # Header
        st.markdown(
            f"<h1 style='text-align: center;'> {st.session_state.text[self.id]['title']} </h1>",
            unsafe_allow_html=True
        )
        st.markdown(st.session_state.text[self.id]["description"])

        # Image uploader
        st.markdown("---")
        ImageUploader(page_id=self.id, image_type=ImageToProcess)

        if len(self.session_state["images"]) > 0:
            st.markdown("---")

            # Image Carousel
            ImageDisplayer(page_id=self.id)
            ImageProcessorOptions(page_id=self.id)


class ImageDisplayer(Component):
    """ Represents an ImageDisplayer. """
    def __init__(
        self,
        page_id: str
    ):
        """
        Initializes an ImageDisplayer.

        Parameters
        ----------
            page_id: str
                id of the page containing the Component
        """
        # ----- Mother class ----- #
        super(ImageDisplayer, self).__init__(page_id)

        # ----- Components ----- #
        n = len(self.session_state["images"])
        for image, col in zip(self.session_state["images"], self.colums([1/n for _ in range(n)])):
            col.image(
                image=image,
                caption="",
                use_column_width=True
            )


class ImageProcessorOptions(Component):
    """ Represents an ImageProcessorOptions. """
    def __init__(
        self,
        page_id: str
    ):
        """
        Initializes an ImageProcessorOptions.

        Parameters
        ----------
            page_id: str
                id of the page containing the Component
        """
        # ----- Mother class ----- #
        super(ImageProcessorOptions, self).__init__(page_id)

        # ----- Components ----- #
        # If only one image has been loaded
        if len(self.session_state["images"]) == 1:
            cols = self.columns((0.5, 0.25, 0.25))

            ProcessingSelector(parent=cols[0], page_id=page_id)
            ProcessButton(parent=cols[1], page_id=page_id)
            ResetButton(parent=cols[2], page_id=page_id)

        # If more than one image has been loaded
        elif len(self.session_state["images"]) > 1:
            cols = self.columns((0.1, 0.4, 0.2, 0.2, 0.1))

            PrevButton(parent=cols[0], page_id=page_id)
            ProcessingSelector(parent=cols[1], page_id=page_id)
            ProcessButton(parent=cols[2], page_id=page_id)
            ResetButton(parent=cols[3], page_id=page_id)
            NextButton(parent=cols[4], page_id=page_id)

        # Apply a custom style to the buttons
        self.markdown(
            body="""<style>                
                div.stButton > button {
                    height: 40.5px;
                    border: None;
                    background-color: #262730;
                }
            </style>""",
            unsafe_allow_html=True
        )


class ProcessingSelector(SubComponent):
    """ Represents a ProcessingSelector. """
    def __init__(
        self,
        parent: st._DeltaGenerator,
        page_id: str
    ):
        """
        Initializes a ProcessingSelector.

        Parameters
        ----------
            parent: st._DeltaGenerator
                container of the SubComponent
            page_id: str
                id of the page containing the Component
        """
        # ----- Mother class ----- #
        super(ProcessingSelector, self).__init__(parent, page_id)

        # ----- Components ----- #
        possible_processes = [""] + list(st.session_state.backend.image_processing_manager.keys())

        # Retrieve the current image's option index
        current_idx = possible_processes.index(
            self.session_state["images"][self.session_state["image_idx"]].process_id
        )

        self.parent.selectbox(
            label="processing selector", label_visibility="collapsed",
            options=possible_processes,
            index=current_idx,
            key=f"{self.page_id}_selectbox"
        )


class ProcessButton(SubComponent):
    """ Represents an ProcessButton. """
    def __init__(
        self,
        parent: st._DeltaGenerator,
        page_id: str
    ):
        """
        Initializes an ProcessButton.

        Parameters
        ----------
            parent: st._DeltaGenerator
                container of the SubComponent
            page_id: str
                id of the page containing the Component
        """
        # ----- Mother class ----- #
        super(ProcessButton, self).__init__(parent, page_id)

        # ----- Components ----- #
        self.parent.button(
            label="process",
            on_click=self.on_click,
            use_container_width=True,
            key=f"{self.parent.id}_{self.id}"
        )

    def on_click(self):
        # Retrieves the processing id
        process_id = st.session_state[f"{self.page_id}_selectbox"]
        if process_id == "":
            return

        # Applies the processing to the current image
        self.session_state["images"][self.session_state["image_idx"]].modified_image = \
            st.session_state.backend.image_processing_manager(process_id)(
                # The image to process
                image=self.session_state["images"][self.session_state["image_idx"]].image
            )

        # Sets the process id of the current image
        self.session_state["images"][self.session_state["image_idx"]].process_id = process_id


class ResetButton(SubComponent):
    """ Represents an ResetButton. """
    def __init__(
        self,
        parent: st._DeltaGenerator,
        page_id: str
    ):
        """
        Initializes a ResetButton.

        Parameters
        ----------
            parent: st._DeltaGenerator
                container of the SubComponent
            page_id: str
                id of the page containing the Component
        """
        # ----- Mother class ----- #
        super(ResetButton, self).__init__(parent, page_id)

        # ----- Components ----- #
        self.parent.button(
            label="reset",
            on_click=self.on_click,
            use_container_width=True,
            key=f"{self.parent.id}_{self.id}"
        )

    def on_click(self):
        # Reset the current image
        self.session_state["images"][self.session_state["image_idx"]].reset()


class PrevButton(SubComponent):
    """ Represents an PrevButton. """
    def __init__(
        self,
        parent: st._DeltaGenerator,
        page_id: str
    ):
        """
        Initializes a PrevButton.

        Parameters
        ----------
            parent: st._DeltaGenerator
                container of the SubComponent
            page_id: str
                id of the page containing the Component
        """
        # ----- Mother class ----- #
        super(PrevButton, self).__init__(parent, page_id)

        # ----- Components ----- #
        self.parent.button(
            label="<",
            on_click=self.on_click,
            use_container_width=True,
            key=f"{self.parent.id}_{self.id}"
        )

    def on_click(self):
        # If the current image is the first one
        if self.session_state["image_idx"] == 0:
            # Sets the current index to the one of the last image
            self.session_state["image_idx"] = len(self.session_state["images"]) - 1
            return

        # Decreases the current index by one
        self.session_state["image_idx"] -= 1


class NextButton(SubComponent):
    """ Represents an NextButton. """
    def __init__(
        self,
        parent: st._DeltaGenerator,
        page_id: str
    ):
        """
        Initializes a NextButton.

        Parameters
        ----------
            parent: st._DeltaGenerator
                container of the SubComponent
            page_id: str
                id of the page containing the Component
        """
        # ----- Mother class ----- #
        super(NextButton, self).__init__(parent, page_id)

        # ----- Components ----- #
        self.parent.button(
            label="\>",
            on_click=self.on_click,
            use_container_width=True,
            key=f"{self.parent.id}_{self.id}"
        )

    def on_click(self):
        # If the current image is the last one
        if self.session_state["image_idx"] == len(self.session_state["images"]) - 1:
            # Sets the current index to the one of the first image
            self.session_state["image_idx"] = 0
            return

        # Increases the current index by one
        self.session_state["image_idx"] += 1
