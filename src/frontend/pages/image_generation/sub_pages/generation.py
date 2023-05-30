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
from src.frontend.components.component import Component


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

        # ----- Session state ----- #
        if "generated_images" not in self.session_state:
            self.session_state["generated_images"] = list()

        # ----- Components ----- #
        # Row n°1
        if len(self.session_state["generated_images"]) > 0:
            ImageDisplayer(page=self.page, parent=self.parent)

        # Row n°2
        if len(self.session_state["generated_images"]) > 1:
            cols = self.parent.columns([0.5, 0.5])

            ImageGenerator(page=self.page, parent=cols[0])
            RankingFeedback(page=self.page, parent=cols[1])
        else:
            ImageGenerator(page=self.page, parent=self.parent)


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
        modulo = len(self.session_state["generated_images"])
        if modulo > 3:
            modulo = 3

        with self.parent.expander(label="", expanded=True):
            if modulo == 1:
                cols = st.columns([1/4, 1/2, 1/4])[1:-1]
            elif modulo == 2:
                cols = st.columns([1/6, 1/3, 1/3, 1/6])[1:-1]
            else:
                cols = st.columns([1/modulo, 1/modulo, 1/modulo])

            # For each generated image
            for idx, image in enumerate(self.session_state["generated_images"]):
                # Display the generated image in the wright column
                cols[idx % modulo].image(
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
        with self.parent.form(key=f"{self.page.ID}_generation_form"):
            # Creates a text_area allowing to specify a LoRA
            st.text_input(
                label="LoRA", label_visibility="collapsed",
                key=f"{self.page.ID}_LoRA_text_input",
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
            "width": int(st.session_state[f"{self.page.ID}_width"]),
            "height": int(st.session_state[f"{self.page.ID}_height"]),
            "num_steps": st.session_state[f"{self.page.ID}_num_steps"],
            "guidance_scale": st.session_state[f"{self.page.ID}_guidance_scale"],
            "num_images": int(st.session_state[f"{self.page.ID}_num_images"]),
            "seed": random.randint(0, 1000)
            if st.session_state[f"{self.page.ID}_num_images"] == "-1"
            else int(st.session_state[f"{self.page.ID}_num_images"])
        }

        # If no mask has been uploaded
        if len(self.session_state["images"]) == 0:
            # Generates images using basic StableDiffusion
            st.session_state.backend.check_stable_diffusion()
            generated_images = st.session_state.backend.stable_diffusion(**args)

        else:
            # Retrieves the uploaded masks and their corresponding processing
            input_masks, processing_ids, weights = list(), list(), list()
            for idx in range(len(self.session_state["images"])):
                # If an image has been uploaded without providing the processing used
                if self.session_state["images"][idx].processing == "":
                    return

                input_masks.append(self.session_state["images"][idx].image)
                processing_ids.append(self.session_state["images"][idx].processing)
                weights.append(self.session_state["images"][idx].weight)

            # Generates images using ControlNet
            st.session_state.backend.reset_control_net()
            st.session_state.backend.check_control_net(processing_ids=processing_ids)
            generated_images = st.session_state.backend.control_net(
                images=input_masks,
                weights=weights,
                **args
            )

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
        with self.parent.form(key=f"{self.page.ID}_feedback_form"):
            # Creates the button allowing to generate an image
            st.text_input(
                label="LoRA", label_visibility="collapsed",
                key=f"{self.page.ID}_feedback_text_input",
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
