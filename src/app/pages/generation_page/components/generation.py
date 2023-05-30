"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import random

# IMPORT: UI
import streamlit as st

# IMPORT: data processing
import numpy as np

# IMPORT: project
from src.frontend.pages.page import Page
from src.frontend.pages.component import Component


class ImageGeneration(Component):
    """ Represents a ImageGeneration. """

    def __init__(
            self,
            page: Page,
            parent: st._DeltaGenerator
    ):
        """
        Initializes a ImageGeneration.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(ImageGeneration, self).__init__(page=page, parent=parent)

        # ----- Components ----- #
        # Row n°1
        ImageDisplayer(page=self.page, parent=self.parent)

        # Row n°2
        cols = self.parent.columns([0.5, 0.5])

        ImageGenerator(page=self.page, parent=cols[0])
        RankingFeedback(page=self.page, parent=cols[1])


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

        # ----- Session state ----- #
        if "generated_images" not in self.session_state:
            self.session_state["generated_images"] = [np.zeros((480, 640, 3))]

        # ----- Components ----- #
        # Retrieves the number of images to display
        num_images = len(self.session_state["generated_images"])

        # If there is 1 image or less
        if num_images <= 1:
            _, col, _ = self.parent.columns([0.25, 0.5, 0.25])
            with col.expander(label="", expanded=True):
                # Display the generated image
                st.image(
                    image=self.session_state["generated_images"][0],
                    use_column_width=True
                )
            return

        # If there is several images
        with self.parent.expander(label="", expanded=True):
            cols = st.columns([0.33, 0.33, 0.33])

            # For each generated image
            for idx, image in enumerate(self.session_state["generated_images"]):
                # Display the generated image in the wright column
                cols[idx % 3].image(
                    image=image,
                    use_column_width=True
                )


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
        with self.parent.form(key=f"{self.page.id}_generation_form"):
            # Creates a text_area allowing to specify a LoRA
            st.text_input(
                label="LoRA", label_visibility="collapsed",
                key=f"{self.page.id}_LoRA_text_input",
                placeholder="Here, you can specify a LoRA ID"
            )

            # Creates the button allowing to generate an image
            st.form_submit_button(
                label="Generate image",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_click(self):
        # If the prompt is empty
        if self.session_state["prompt"] == "":
            return

        # Retrieves the parameters needed to generate an image
        args = {
            "prompt": self.session_state["prompt"],
            "negative_prompt": self.session_state["negative_prompt"],
            "width": int(st.session_state[f"{self.page.id}_width"]),
            "height": int(st.session_state[f"{self.page.id}_height"]),
            "num_steps": st.session_state[f"{self.page.id}_num_steps"],
            "guidance_scale": st.session_state[f"{self.page.id}_guidance_scale"],
            "num_images": int(st.session_state[f"{self.page.id}_num_images"]),
            "seed": random.randint(0, 1000)
            if st.session_state[f"{self.page.id}_num_images"] == "-1"
            else int(st.session_state[f"{self.page.id}_num_images"])
        }

        # If no mask has been uploaded
        if len(self.session_state["images"]) == 0:
            # Generates images using basic StableDiffusion
            st.session_state.backend.check_stable_diffusion()
            generated_images = st.session_state.backend.stable_diffusion(**args)

        else:
            # Retrieves the uploaded masks and their corresponding processing
            input_masks, processing_ids = list(), list()
            for idx in range(len(self.session_state["images"])):
                # If an image has been uploaded without providing the processing used
                if self.session_state["images"][idx].processing == "":
                    return

                input_masks.append(self.session_state["images"][idx].image)
                processing_ids.append(self.session_state["images"][idx].processing)

            # Generates images using ControlNet
            st.session_state.backend.reset_control_net()
            st.session_state.backend.check_control_net(processing_ids=processing_ids)
            generated_images = st.session_state.backend.control_net(images=input_masks, **args)

        # Updates the in memory generated images
        self.session_state["generated_images"] = [np.array(image) for image in generated_images]


class RankingFeedback(Component):
    """ Represents a RankingFeedback. """

    def __init__(
            self,
            page: Page,
            parent: st._DeltaGenerator
    ):
        """
        Initializes a RankingFeedback.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(RankingFeedback, self).__init__(page=page, parent=parent)

        # ----- Components ----- #
        with self.parent.form(key=f"{self.page.id}_feedback_form"):
            # Creates the button allowing to generate an image
            st.text_input(
                label="LoRA", label_visibility="collapsed",
                key=f"{self.page.id}_feedback_text_input",
                placeholder="Here, you can give your feedback by ranking the images"
            )

            # Creates the button allowing to generate an image
            st.form_submit_button(
                label="Give feedback",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_click(self):
        pass
