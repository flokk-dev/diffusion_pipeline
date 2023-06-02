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


class FeedbackPage(Component):
    """ Represents the sub-page allowing to improve the generation using ranking feedback. """

    def __init__(self, page: Page, parent: st._DeltaGenerator):
        """
        Initializes the sub-page allowing to improve the generation using ranking feedback.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(FeedbackPage, self).__init__(page, parent, component_id="image_generation")
        self.parent.info("Here, you can view the generated images and improve them via feedback")

        # ----- Session state ----- #
        # Creates the list of generated images
        if "generated_images" not in self.session_state:
            self.session_state["generated_images"] = list()

        # Creates the list of latents from which the image generation has started
        if "latents" not in self.session_state:
            self.session_state["latents"] = None

        # ----- Components ----- #
        # Row n°1
        if len(self.session_state["generated_images"]) > 0:
            ImageDisplayer(page=self.page, parent=self.parent)  # displays generated images

        # Row n°2
        if len(self.session_state["generated_images"]) > 1:
            RankingFeedback(page=self.page, parent=self.parent)  # allows to improve results


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
                cols[idx % modulo].image(image=image, caption=str(idx + 1), use_column_width=True)


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
            if isinstance(st.session_state.backend.ranking_feedback, type):
                col1, col2 = st.columns([0.5, 0.5])
                col1.text_input(
                    key=f"{self.page.ID}_{self.ID}_feedback",
                    label="feedback", label_visibility="collapsed",
                    placeholder="Please indicate the index of the best image"
                )

                col2.text_input(
                    key=f"{self.page.ID}_{self.ID}_num_images",
                    label="number of images", label_visibility="collapsed",
                    placeholder="Please specify the number of images to generate at each step"
                )

            else:
                placeholder = \
                    "Please rank the images from best to worst (ex: 6-1-3-4)" \
                    if st.session_state.backend.ranking_feedback.step == "gradient_estimation" \
                    else "Please specify the index of the best image" \

                st.text_input(
                    key=f"{self.page.ID}_{self.ID}_feedback",
                    label="feedback", label_visibility="collapsed",
                    placeholder=placeholder
                )

            # Creates the button allowing to run the ranking feedback
            st.form_submit_button(
                label="Give feedback",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_click(self):
        # Retrieves the ranking of the images
        ranking = st.session_state[f"{self.page.ID}_{self.ID}_feedback"]

        # Instantiate the ranking feedback if needed
        if isinstance(st.session_state.backend.ranking_feedback, type):
            best_image_idx = int(ranking.split("-")[0]) - 1

            st.session_state.backend.ranking_feedback = st.session_state.backend.ranking_feedback(
                latent=self.session_state["latents"][best_image_idx],
                image=self.session_state["generated_images"][best_image_idx],
                num_latents=int(st.session_state[f"{self.page.ID}_{self.ID}_num_images"]),
                smoothing_factor=0.1,
                lr=2.0
            )
        else:
            st.session_state.backend.ranking_feedback.ranking_feedback(
                ranking=ranking,
                generated_images=self.session_state["generated_images"]
            )

        # Launches an iteration
        latents = st.session_state.backend.ranking_feedback.gen_new_latents()
        _, generated_images = self.generate_image(latents)

        # Updates the in memory generated images
        self.session_state["generated_images"] = \
            st.session_state.backend.ranking_feedback.adjust_generated_images(generated_images)

    def generate_image(
        self,
        latents: torch.Tensor = None
    ) -> Tuple[torch.Tensor, List[Image.Image]]:
        """
        Generates images using the in memory parameters.

        Parameters
        ----------
            latents: torch.Tensor
                latents from which to start the generation
        """
        # If no mask has been uploaded
        if len(self.session_state["images"]) == 0:
            # Generates images using only StableDiffusion
            st.session_state.backend.check_stable_diffusion()
            return st.session_state.backend.stable_diffusion(
                **self.session_state["generation_args"],
                num_images=latents.shape[0],
                latents=latents
            )

        else:
            # Generates images using ControlNet + StableDiffusion
            return st.session_state.backend.controlnet(
                **self.session_state["generation_args"],
                num_images=latents.shape[0],
                latents=latents
            )
