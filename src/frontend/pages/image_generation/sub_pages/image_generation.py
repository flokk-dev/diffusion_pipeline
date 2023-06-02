"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
import streamlit as st

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

        # Creates the list of parameters used during the image generation
        if "generation_args" not in self.session_state:
            self.session_state["generation_args"]: dict = None

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

        # ----- Components ----- #
        with self.parent.expander(label="", expanded=True):
            # Row n°1
            col1, col2 = st.columns([0.5, 0.5])

            # Creates the text_area allowing to specify the prompt
            col1.text_area(
                key=f"{self.page.ID}_{self.ID}",
                label="text_area", label_visibility="collapsed",
                value="",
                placeholder="Here, you have to write the prompt",
                height=125
            )

            # Creates the text_area allowing to specify the negative prompt
            col2.text_area(
                key=f"{self.page.ID}_{self.ID}_negative",
                label="text_area", label_visibility="collapsed",
                value="monochrome, lowres, bad anatomy, worst quality, low quality",
                placeholder="Here, you have to write the negative prompt",
                height=125
            )

            # Row n°2
            # Creates the button allowing to improve the prompt
            st.button(
                key=f"{self.page.ID}_{self.ID}_button",
                label="Improve the prompt",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_click(self):
        # If the text_area containing the prompt to improve is empty
        if st.session_state[f"{self.page.ID}_{self.ID}"] == "":
            st.sidebar.warning(
                "WARNING: you need to provide a prompt before trying to improve it."
            )
            return

        # Improves the prompt
        st.session_state.backend.check_promptist()
        prompt = st.session_state.backend.promptist(
            prompt=st.session_state[f"{self.page.ID}_{self.ID}"]
        )

        # Updates the content of the text area
        st.session_state[f"{self.page.ID}_{self.ID}"] = prompt


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
        if st.session_state[f"{self.page.ID}_prompt"] == "":
            st.sidebar.warning(
                "WARNING: you need to provide a prompt before trying to generate an image."
            )
            return

        # Retrieves the parameters needed to generate images
        args = {
            "prompt": st.session_state[f"{self.page.ID}_prompt"],
            "negative_prompt": st.session_state[f"{self.page.ID}_prompt_negative"],
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
            for image in self.session_state["images"]:
                # If an image has been uploaded without providing the processing used
                if image.processing == "":
                    st.sidebar.warning(
                        "WARNING: you need to specify the processing used for each "
                        "uploaded ControlNet mask"
                    )
                    return

                input_masks.append(image.image)
                controlnet_ids.append(image.processing)
                weights.append(image.weight)

            # Adds the new parameters to the previous ones
            args = {**args, "images": input_masks, "weights": weights}

            # Generates images using ControlNet + StableDiffusion
            st.session_state.backend.check_controlnet(controlnet_ids=controlnet_ids)
            latents, generated_images = st.session_state.backend.controlnet(**args)

        # Stores the parameters used to generate images
        del args["num_images"]
        self.session_state["generation_args"] = args

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
                min_value=0, max_value=25, value=1, step=1,
            )  # Creates the object to select the number of images to generate

            col2.slider(
                key=f"{self.page.ID}_{self.ID}_seed",
                label="Seed of the randomness (random if -1)",
                min_value=-1, max_value=None, value=-1, step=1,
            )  # Creates the object to select the seed of the randomness

            # Row n°2
            col1, col2 = st.columns([0.5, 0.5])

            col1.slider(
                key=f"{self.page.ID}_{self.ID}_width",
                label="Width of the image to generate",
                min_value=0, max_value=1024, value=512, step=8,
            )  # Creates the object to select the width of the image to generate

            col2.slider(
                key=f"{self.page.ID}_{self.ID}_height",
                label="Height of the image to generate",
                min_value=0, max_value=1024, value=512, step=8,
            )  # Creates the object to select the height of the image to generate

            # Row n°3
            col1, col2 = st.columns([0.5, 0.5])

            col1.slider(
                key=f"{self.page.ID}_{self.ID}_guidance_scale",
                label="Guidance scale of the generation",
                min_value=0.0, max_value=21.0, value=7.5, step=0.1,
            )  # Creates the object to select the guidance scale of the generation

            col2.slider(
                key=f"{self.page.ID}_{self.ID}_num_steps",
                label="Number of denoising steps",
                min_value=0, max_value=100, value=30, step=1,
            )  # Creates the object to select the number of denoising step of the generation
