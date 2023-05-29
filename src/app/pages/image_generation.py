"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: UI
import streamlit as st

# IMPORT: data processing
import cv2
import numpy as np
import PIL

# IMPORT: project
from src.app.component import Page, Component
from src.app.component import ImageUploader
from src.image_utils.image import Mask


class ImageGeneration(Page):
    """ Represents an ImageGeneration. """
    def __init__(
        self,
        parent
    ):
        """ Initializes an ImageGeneration. """
        super(ImageGeneration, self).__init__(id_="image_generation", parent=parent)

        # ----- Session state ----- #
        if "images" not in self.session_state:
            self.session_state["images"] = [Mask() for _ in range(3)]

        if "image_idx" not in self.session_state:
            self.session_state["image_idx"] = 0

        if "prompt" not in self.session_state:
            self.session_state["prompt"] = ""

        if "generated_image" not in self.session_state:
            self.session_state["generated_image"] = np.zeros((480, 640, 3))

        # ----- Components ----- #
        # Row n°1
        ImageDisplayer(page=self, parent=self.parent)

        # Row n°2
        cols = self.parent.columns((0.5, 0.5))

        ImageUploader(page=self, parent=cols[0])
        MaskRanker(page=self, parent=cols[1])

        self.parent.markdown("---")
        # Row n°3
        PromptImprovement(page=self, parent=self.parent)

        self.parent.markdown("---")
        # Row n°4
        ImageGenerator(page=self, parent=self.parent)


class ImageDisplayer(Component):
    """ Represents an ImageDisplayer. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator
    ):
        """
        Initializes an ImageDisplayer.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(ImageDisplayer, self).__init__(page=page, parent=parent)

        # ----- Components ----- #
        # Retrieves the processing options
        options = [""] + list(st.session_state.backend.control_net.CONTROL_NETS_IDS.keys())
        print("HEEEEEEEERE")
        with self.parent.expander(label="", expanded=True):
            for idx, col in enumerate(st.columns([1 for _ in self.session_state["images"]])):
                print(type(self.session_state["images"][idx].image))
                print(self.session_state["images"][idx].image.shape)
                # Creates the selectbox allowing to select a processing
                col.selectbox(
                    label="selectbox", label_visibility="collapsed",
                    key=f"{self.page.id}_selectbox_{idx}",
                    options=options,
                    on_change=self.on_change, args=(idx, ),
                    index=options.index(self.session_state["images"][idx].processing)
                )

                # Displays the current image
                col.image(
                    image=self.session_state["images"][idx].image,
                    caption=self.session_state["images"][idx].name,
                    use_column_width=True
                )

    def on_change(self, idx):
        # Updates the processing of the image at index idx
        self.session_state["images"][idx].processing = \
            st.session_state[f"{self.page.id}_selectbox_{idx}"]

        # Resets the ControlNet
        st.session_state.backend.reset_control_net()


class MaskRanker(Component):
    """ Represents a MaskRanker. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator
    ):
        """
        Initializes a MaskRanker.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(MaskRanker, self).__init__(page=page, parent=parent)

        # ----- Components ----- #
        with self.parent.form(key=f"{self.page.id}_form_0"):
            # Creates the text_input allowing to specify the ranking of the images
            st.text_input(
                label="text_input", label_visibility="collapsed",
                key=f"{self.page.id}_text_input",
                placeholder="Please rank the masks by importance (separated by a dash)"
            )

            # Creates the button allowing to rank the images
            st.form_submit_button(
                label="Rank the images",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_click(self):
        # If no image has been loaded
        if len(self.session_state["images"]) == 0:
            return

        # If no rank has been entered
        if len(st.session_state[f"{self.page.id}_text_input"]) == 0:
            return

        # Retrieves the masks ranking
        ranks = [int(e) for e in st.session_state[f"{self.page.id}_text_input"].split("-")]

        # Applies the ranking on the in memory images
        self.session_state["images"] = [
            self.session_state["images"][rank]
            for rank
            in ranks
        ]

        # Sets the current index according to the modifications
        self.session_state["image_idx"] = ranks.index(self.session_state["image_idx"])


class PromptImprovement(Component):
    """ Represents an PromptImprovement. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator
    ):
        """
        Initializes an PromptImprovement.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(PromptImprovement, self).__init__(page=page, parent=parent)

        # ----- Components ----- #
        with self.parent.form(key=f"{self.page.id}_form_1"):
            # Creates the text_area in which to display the prompt needed to generate the image
            st.text_area(
                label="text_area", label_visibility="collapsed",
                key=f"{self.page.id}_text_area",
                value=self.session_state["prompt"],
                height=125
            )

            # Creates the button allowing to improve the caption
            st.form_submit_button(
                label="Improve the prompt",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_change(self):
        # Updates the content of the text area
        self.session_state["prompt"] = st.session_state[f"{self.page.id}_text_area"]

    def on_click(self):
        # If the text_area containing the prompt to improve is empty
        if st.session_state[f"{self.page.id}_text_area"] == "":
            return

        # Improves the prompt
        prompt = st.session_state.backend.image_captioning_manager("promptist")(
            prompt=st.session_state[f"{self.page.id}_text_area"]
        )

        # Updates the content of the text area
        self.session_state["prompt"] = prompt


class ImageGenerator(Component):
    """ Represents an ImageGenerator. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator
    ):
        """
        Initializes an ProcessingSelector.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(ImageGenerator, self).__init__(page=page, parent=parent)

        # ----- Components ----- #
        # Displays the generated image
        self.parent.image(
            image=self.session_state["generated_image"],
            caption="generated image",
            use_column_width=True
        )

        # Creates the button allowing to generate an image
        self.parent.button(
            label="Generate image",
            on_click=self.on_click,
            use_container_width=True
        )

    def on_click(self):
        # If the text_area containing the prompt to use is empty
        if st.session_state[f"{self.page.id}_text_area"] == "":
            return
        prompt = st.session_state[f"{self.page.id}_text_area"]
        print(prompt)

        # Retrieves the processing used
        processing_ids = list()
        for image in self.session_state["images"]:
            if image.processing != "":
                processing_ids.append(image.processing)
        print(processing_ids)
        print(len(processing_ids))

        # If the ControlNet has not been already instantiated
        if isinstance(st.session_state.backend.control_net, type):
            st.session_state.backend.control_net = st.session_state.backend.control_net(
                processing_ids=processing_ids
            )

        # Retrieves the input masks
        input_masks = [
            PIL.Image.fromarray(
                cv2.resize(image.image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            )
            for image
            in self.session_state["images"]
            if image.id is not None
        ]
        print(len(input_masks))

        # Generates an image based on all the inputs
        generated_image = st.session_state.backend.control_net(
            prompt=prompt,
            negative_prompt="",
            images=input_masks,
            weights=[1. for _ in input_masks],
            seed=0
        )
        generated_image = cv2.resize(
            np.array(generated_image), (1000, 600), interpolation=cv2.INTER_LANCZOS4
        )
        print(generated_image.shape)

        # Updates the processing of the current image
        self.session_state["generated_image"] = generated_image
