"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import streamlit as st

import torch
from PIL import Image

# IMPORT: project
from src.frontend.pages import Page
from src.frontend.components.component import Component


class ImageGeneration(Component):
    """ Represents the sub-page allowing to adjust the diffusion parameters and generate images. """

    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes the sub-page allowing to adjust the diffusion parameters and generate images.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(ImageGeneration, self).__init__(page, parent, component_id="image_generation")
        self.parent.info("Here, you can adjust the diffusion parameters and generate images")

        # ----- Session state ----- #
        # Creates the list of generated images
        if "generated_images" not in self.session_state:
            self.session_state["generated_images"] = list()

        # Creates the list of latents from which the image generation has started
        if "latents" not in self.session_state:
            self.session_state["latents"] = None

        # ----- Components ----- #
        # Row n°1
        Prompt(page=self.page, parent=self.parent)  # allows to specify the prompt/negative prompt

        # Row n°2
        ImageGenerator(page=self.page, parent=self.parent)  # allows to set up/launch the generation


class Prompt(Component):
    """ Represents the component allowing to specify the prompt and the negative prompt. """

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
        # Creates the prompt for the generation
        if "prompt" not in self.session_state:
            self.session_state["prompt"] = ""

        # ----- Components ----- #
        with self.parent.expander(label="", expanded=True):
            # Row n°1
            col1, col2 = st.columns([0.5, 0.5])

            # Creates the text_area allowing to specify the prompt
            col1.text_area(
                key=f"{self.page.ID}_{self.ID}_prompt",
                label="text_area", label_visibility="collapsed",
                value=self.session_state["prompt"],
                placeholder="Here, you have to write the prompt",
                height=125,
                on_change=self.on_change
            )

            # Creates the text_area allowing to specify the negative prompt
            col2.text_area(
                key=f"{self.page.ID}_{self.ID}_negative_prompt",
                label="text_area", label_visibility="collapsed",
                value="monochrome, lowres, bad anatomy, worst quality, low quality",
                placeholder="Here, you have to write the negative prompt",
                height=125
            )

            # Row n°2
            # Creates the button allowing to improve the prompt
            st.button(
                label="Improve the prompt",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_change(self):
        # Assigns the value of the text_area to the prompt
        self.session_state["prompt"] = st.session_state[f"{self.page.ID}_{self.ID}_prompt"]

    def on_click(self):
        # If the text_area containing the prompt to improve is empty
        if st.session_state[f"{self.page.ID}_{self.ID}_prompt"] == "":
            st.sidebar.warning(
                "WARNING: you need to provide a prompt before trying to improve it."
            )
            return

        # Improves the prompt
        st.session_state.backend.check_promptist()
        prompt = st.session_state.backend.promptist(
            prompt=st.session_state[f"{self.page.ID}_{self.ID}_prompt"]
        )

        # Updates the content of the text area
        self.session_state["prompt"] = prompt


class ImageGenerator(Component):
    """ Represents the component allowing to generate images. """

    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes the component allowing to generate images.

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
            # Creates the hyperparameters allowing to adjust the generation
            HyperParameters(page=self.page, parent=st)

            # Creates the button allowing to generate an image
            st.form_submit_button(
                label="Generate images",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_click(self):
        # If the prompt is empty
        if self.session_state["prompt"] == "":
            st.sidebar.warning(
                "WARNING: you need to provide a prompt before trying to generate an image."
            )
            return

        # Retrieves the parameters needed to generate an image
        args = {
            "prompt": self.session_state["prompt"],
            "negative_prompt": st.session_state[f"{self.page.ID}_prompt_negative_prompt"],
            "num_images": st.session_state[f"{self.page.ID}_hyperparameters_num_images"],
            "width": st.session_state[f"{self.page.ID}_hyperparameters_width"],
            "height": st.session_state[f"{self.page.ID}_hyperparameters_height"],
            "num_steps": st.session_state[f"{self.page.ID}_hyperparameters_num_steps"],
            "guidance_scale": st.session_state[f"{self.page.ID}_hyperparameters_guidance_scale"],
            "seed": None
            if st.session_state[f"{self.page.ID}_hyperparameters_seed"] == -1
            else st.session_state[f"{self.page.ID}_hyperparameters_seed"]
        }

        # If no mask has been uploaded
        if len(self.session_state["images"]) == 0:
            # Generates images using only StableDiffusion
            st.session_state.backend.check_stable_diffusion()
            latents, generated_images = st.session_state.backend.stable_diffusion(**args)

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
            latents, generated_images = st.session_state.backend.controlnet(
                images=input_masks,
                weights=weights,
                **args
            )

        # Resets the ranking feedback
        st.session_state.backend.reset_ranking_feedback()

        # Updates the in memory generated images
        self.session_state["generated_images"] = generated_images

        # Updates the in memory latents
        self.session_state["latents"] = latents


class HyperParameters(Component):
    """ Represents the sub-page allowing to adjust the hyperparameters. """

    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes the sub-page allowing to adjust the hyperparameters.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(HyperParameters, self).__init__(page, parent, component_id="hyperparameters")

        # ----- Components ----- #
        with self.parent.expander(label="Hyperparameters", expanded=False):
            # Row n°1
            st.markdown("---")
            col1, col2 = st.columns([0.5, 0.5])

            col1.slider(
                key=f"{self.page.ID}_{self.ID}_num_images",
                label="Number of images to generate",
                min_value=0, max_value=10, value=1, step=1,
            )

            col2.slider(
                key=f"{self.page.ID}_{self.ID}_seed",
                label="Seed of the randomness (random if -1)",
                min_value=-1, max_value=None, value=-1, step=1,
            )

            # Row n°2
            col1, col2 = st.columns([0.5, 0.5])

            col1.slider(
                key=f"{self.page.ID}_{self.ID}_width",
                label="Width of the image to generate",
                min_value=0, max_value=1024, value=512, step=8,
            )

            col2.slider(
                key=f"{self.page.ID}_{self.ID}_height",
                label="Height of the image to generate",
                min_value=0, max_value=1024, value=512, step=8,
            )

            # Row n°3
            col1, col2 = st.columns([0.5, 0.5])

            col1.slider(
                key=f"{self.page.ID}_{self.ID}_guidance_scale",
                label="Guidance scale of the generation",
                min_value=0.0, max_value=21.0, value=7.5, step=0.1,
            )

            col2.slider(
                key=f"{self.page.ID}_{self.ID}_num_steps",
                label="Number of denoising steps",
                min_value=0, max_value=100, value=20, step=1,
            )
