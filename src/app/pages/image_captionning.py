"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: UI
import streamlit as st

# IMPORT: project
from src.app.component import Page, Component, SubComponent
from src.image_utils.image import ImageToDescribe
from src.app.component import ImageUploader


class ImageCaptioningPage(Page):
    """ Represents an ImageCaptioningPage. """
    def __init__(
        self
    ):
        """ Initializes an ImageCaptioningPage. """
        # ----- Mother class ----- #
        super(ImageCaptioningPage, self).__init__(page_id="image_captioning")

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
        ImageUploader(page_id=self.id, image_type=ImageToDescribe)

        if len(self.session_state["images"]) > 0:
            st.markdown("---")

            # Image Carousel
            ImageDisplayer(page_id=self.id)
            ImageCaptionerOptions(page_id=self.id)


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
        # Image
        self.image(
            image=self.session_state["images"][self.session_state["image_idx"]].image,
            caption="",
            use_column_width=True
        )


class ImageCaptionerOptions(Component):
    """ Represents an ImageCaptionerOptions. """
    def __init__(
        self,
        page_id: str
    ):
        """
        Initializes an ImageCaptionerOptions.

        Parameters
        ----------
            page_id: str
                id of the page containing the Component
        """
        # ----- Mother class ----- #
        super(ImageCaptionerOptions, self).__init__(page_id)

        # ----- Components ----- #
        # If only one image has been loaded
        if len(self.session_state["images"]) == 1:
            cols = self.columns((0.8, 0.2))

            ImageCaption(parent=cols[0], page_id=page_id)
            DescribeButton(parent=cols[1], page_id=page_id)

        # If more than one image has been loaded
        elif len(self.session_state["images"]) > 1:
            cols = self.columns((0.1, 0.6, 0.2, 0.1))

            PrevButton(parent=cols[0], page_id=page_id)
            ImageCaption(parent=cols[1], page_id=page_id)
            DescribeButton(parent=cols[2], page_id=page_id)
            NextButton(parent=cols[3], page_id=page_id)

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


class ImageCaption(SubComponent):
    """ Represents a ImageCaption. """
    def __init__(
        self,
        parent: st._DeltaGenerator,
        page_id: str
    ):
        """
        Initializes a ImageCaption.

        Parameters
        ----------
            parent: st._DeltaGenerator
                container of the SubComponent
            page_id: str
                id of the page containing the Component
        """
        # ----- Mother class ----- #
        super(ImageCaption, self).__init__(parent, page_id)

        # ----- Components ----- #
        self.parent.text_area(
            label="caption", label_visibility="collapsed",
            height=50,
            value=self.session_state["images"][self.session_state["image_idx"]].caption,
            key=f"{self.page_id}_text_area"
        )


class DescribeButton(SubComponent):
    """ Represents an DescribeButton. """
    def __init__(
        self,
        parent: st._DeltaGenerator,
        page_id: str
    ):
        """
        Initializes a DescribeButton.

        Parameters
        ----------
            parent: st._DeltaGenerator
                container of the SubComponent
            page_id: str
                id of the page containing the Component
        """
        # ----- Mother class ----- #
        super(DescribeButton, self).__init__(parent, page_id)

        # ----- Components ----- #
        self.parent.button(
            label="describe",
            on_click=self.on_click,
            use_container_width=True,
            key=f"{self.parent.id}_{self.id}"
        )

    def on_click(self):
        # Applies the processing to the current image
        caption = st.session_state.backend.image_captioning_manager("clip_interrogator")(
            # The image to process
            image=self.session_state["images"][self.session_state["image_idx"]].image
        )

        # Updates the text area
        st.session_state[f"{self.page_id}_text_area"] = caption

        # Sets the caption of the current image
        self.session_state["images"][self.session_state["image_idx"]].caption = caption


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
