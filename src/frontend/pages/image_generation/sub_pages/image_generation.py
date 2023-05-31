"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import streamlit as st
import random

# IMPORT: data processing
import numpy as np

# IMPORT: project
from src.frontend.pages import Page
from src.frontend.components.component import Component


class ImageGeneration(Component):
    """ Represents the sub-page allowing to generate images. """

    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes the sub-page allowing to generate images.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(ImageGeneration, self).__init__(page, parent, component_id="image_generation")
        self.parent.info("Here, you can generate images depending on the choices you maid before.")

        # ----- Session state ----- #
        # Creates the list of generated images
        if "generated_images" not in self.session_state:
            self.session_state["generated_images"] = list()

        # ----- Components ----- #
        # Row n°1
        if len(self.session_state["generated_images"]) > 0:
            ImageDisplayer(page=self.page, parent=self.parent)  # displays generated images

        # Row n°2
        if len(self.session_state["generated_images"]) > 1:
            cols = self.parent.columns([0.5, 0.5])

            ImageGenerator(page=self.page, parent=cols[0])  # launches the generation
            RankingFeedback(page=self.page, parent=cols[1])  # allows to improve results
        else:
            ImageGenerator(page=self.page, parent=self.parent)  # launches the generation


class ImageDisplayer(Component):
    """ Represents the components that displays the generated images. """

    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes the components that displays the generated images.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(ImageDisplayer, self).__init__(page, parent, component_id="image_displayer")

        # ----- Components ----- #
        # Checks how much images to display on each row
        modulo = len(self.session_state["generated_images"])
        if modulo > 3:
            modulo = 3

        with self.parent.expander(label="", expanded=True):
            # If there should be 1 image per row, then creates 1 centered container
            if modulo == 1:
                cols = st.columns([1/4, 1/2, 1/4])[1:-1]
            # If there should be 2 images per row, then creates 2 centered containers
            elif modulo == 2:
                cols = st.columns([1/6, 1/3, 1/3, 1/6])[1:-1]
            # If there should be 3 images per row, then creates 3 containers
            else:
                cols = st.columns([1/modulo, 1/modulo, 1/modulo])

            # For each generated image
            for idx, image in enumerate(self.session_state["generated_images"]):
                # Display the generated image in the wright column
                cols[idx % modulo].image(image=image, use_column_width=True)


class ImageGenerator(Component):
    """ Represents the component allowing to launch the generation. """

    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes the component allowing to launch the generation.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(ImageGenerator, self).__init__(page, parent, component_id="image_generator")

        # ----- Components ----- #
        with self.parent.form(key=f"{self.page.ID}_{self.ID}_form"):
            # Creates a text_area allowing to specify a LoRA
            st.text_input(
                key=f"{self.page.ID}_{self.ID}_text_input",
                label="LoRA", label_visibility="collapsed",
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
            # Generates images using only StableDiffusion
            st.session_state.backend.check_stable_diffusion()
            generated_images = st.session_state.backend.stable_diffusion(**args)

        else:
            # Retrieves the uploaded masks and their corresponding (processing, weight)
            input_masks, controlnet_ids, weights = list(), list(), list()
            for idx in range(len(self.session_state["images"])):
                # If an image has been uploaded without providing the processing used
                if self.session_state["images"][idx].processing == "":
                    return

                input_masks.append(self.session_state["images"][idx].image)
                controlnet_ids.append(self.session_state["images"][idx].processing)
                weights.append(self.session_state["images"][idx].weight)

            # Generates images using ControlNet + StableDiffusion
            st.session_state.backend.check_controlnet(controlnet_ids=controlnet_ids)
            generated_images = st.session_state.backend.controlnet(
                images=input_masks,
                weights=weights,
                **args
            )

        # Updates the in memory generated images
        self.session_state["generated_images"] = [np.array(image) for image in generated_images]


class RankingFeedback(Component):
    """ Represents the component allowing to improve the generation using ranking feedback. """

    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes the component allowing to improve the generation using ranking feedback.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(RankingFeedback, self).__init__(page, parent, component_id="ranking_feedback")

        # ----- Components ----- #
        with self.parent.form(key=f"{self.page.ID}_{self.ID}_form"):
            # Creates the button allowing to generate an image
            st.text_input(
                key=f"{self.page.ID}_{self.ID}_text_input",
                label="LoRA", label_visibility="collapsed",
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
