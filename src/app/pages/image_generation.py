"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: UI
import streamlit as st

# IMPORT: data processing
import numpy as np

# IMPORT: project
from src.app.component import Page, Component
from src.app.component import ImageUploader
from src.image_utils.image import Images, Mask, Image


class ImageGenerationPage(Page):
    """ Represents an ImageGenerationPage. """
    def __init__(
        self,
        parent
    ):
        """ Initializes an ImageGenerationPage. """
        super(ImageGenerationPage, self).__init__(id_="image_generation", parent=parent)

        # ----- Session state ----- #
        if "images" not in self.session_state:
            self.session_state["images"] = Images(image_type=Mask)

        # ----- Components ----- #
        # Options
        tabs = self.parent.tabs(["ControlNet", "LoRA", "Prompt", "Hyper-parameters"])

        ControlNetSelector(page=self, parent=tabs[0])
        Prompts(page=self, parent=tabs[2])
        HyperParameters(page=self, parent=tabs[3])

        # Generation
        ImageGeneration(page=self, parent=self.parent)


# ---------- CONTROLNET ---------- #

class ControlNetSelector(Component):
    """ Represents a ControlNetSelector. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator
    ):
        """
        Initializes a ControlNetSelector.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(ControlNetSelector, self).__init__(page=page, parent=parent)
        self.parent.info(
            "Here, you can load some masks that will then guide the generation using ControlNet."
        )

        # ----- Components ----- #
        # Row n°1
        MaskDisplayer(page=self.page, parent=self.parent)

        # Row n°2
        cols = self.parent.columns((0.5, 0.5))

        ImageUploader(page=self.page, parent=cols[0])
        MaskRanker(page=self.page, parent=cols[1])


class MaskDisplayer(Component):
    """ Represents a MaskDisplayer. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator
    ):
        """
        Initializes a MaskDisplayer.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(MaskDisplayer, self).__init__(page=page, parent=parent)

        # ----- Components ----- #
        # Retrieves the processing options
        options = [""] + list(st.session_state.backend.control_net.CONTROL_NETS_IDS.keys())

        with self.parent.expander(label="", expanded=True):
            # For each in memory image creates a column
            for idx, col in enumerate(st.columns([1 for _ in self.session_state["images"]])):
                # Displays the mask
                col.image(
                    image=self.session_state["images"][idx].image,
                    use_column_width=True
                )

                # Creates the selectbox allowing to indicate the processing that gives the mask
                col.selectbox(
                    label="selectbox", label_visibility="collapsed",
                    key=f"{self.page.id}_selectbox_{idx}",
                    options=options,
                    on_change=self.on_change, args=(idx, ),
                    index=options.index(self.session_state["images"][idx].processing)
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
                label="Rank the masks",
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

        # If the number of ranked element is less than the number of masks
        ranking = [int(idx) for idx in st.session_state[f"{self.page.id}_text_input"].split("-")]
        if len(ranking) < len(self.session_state["images"]):
            return

        # Separates the valid and invalid indexes
        valid_idx, invalid_idx = list(), list()
        for idx in ranking:
            if idx <= len(self.session_state["images"]) - 1:
                valid_idx.append(idx)
            else:
                invalid_idx.append(idx)

        # Applies the ranking on the in memory images
        new_images = [self.session_state["images"][idx] for idx in valid_idx + invalid_idx]
        for idx, image in enumerate(self.session_state["images"]):
            image = 
        self.session_state["images"] = [
            self.session_state["images"][idx]
            for idx
            in valid_idx + invalid_idx
        ]


# ---------- PROMPTS ---------- #

class Prompts(Component):
    """ Represents a Prompts. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator
    ):
        """
        Initializes a Prompts.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(Prompts, self).__init__(page=page, parent=parent)
        self.parent.info(
            "Here, you can specify a prompt and a negative prompt (to avoid some key words) "
            "that will then guide the generation."
        )

        # ----- Components ----- #
        cols = self.parent.columns([0.5, 0.5])

        # Col n°1
        Prompt(page=self.page, parent=cols[0])

        # Col n°2
        NegativePrompt(page=self.page, parent=cols[1])


class Prompt(Component):
    """ Represents a Prompt. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator
    ):
        """
        Initializes a Prompt.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(Prompt, self).__init__(page=page, parent=parent)

        # ----- Session state ----- #
        if "prompt" not in self.session_state:
            self.session_state["prompt"] = ""

        # ----- Components ----- #
        with self.parent.expander(label="", expanded=True):
            # Creates the text_area allowing to specify the prompt
            st.text_area(
                label="text_area", label_visibility="collapsed",
                key=f"{self.page.id}_prompt",
                height=125,
                placeholder="Here, you have to describe the content of the generation.",
                value=self.session_state["prompt"],
                on_change=self.on_change
            )

            # Creates the button allowing to improve the prompt
            st.button(
                label="Improve the prompt",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_change(self):
        # Assigns the value of the text_area to the prompt
        self.session_state["prompt"] = st.session_state[f"{self.page.id}_prompt"]

    def on_click(self):
        # If the text_area containing the prompt to improve is empty
        if st.session_state[f"{self.page.id}_prompt"] == "":
            return

        # Improves the prompt
        st.session_state.backend.check_promptist()
        prompt = st.session_state.backend.promptist(
            prompt=st.session_state[f"{self.page.id}_prompt"]
        )

        # Updates the content of the text area
        self.session_state["prompt"] = prompt


class NegativePrompt(Component):
    """ Represents a NegativePrompt. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator
    ):
        """
        Initializes a NegativePrompt.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(NegativePrompt, self).__init__(page=page, parent=parent)

        # ----- Session state ----- #
        if "negative_prompt" not in self.session_state:
            self.session_state["negative_prompt"] = ""

        # ----- Components ----- #
        with self.parent.expander(label="", expanded=True):
            # Creates the text_area allowing to specify the negative prompt
            st.text_area(
                label="text_area", label_visibility="collapsed",
                key=f"{self.page.id}_negative_prompt",
                height=125,
                placeholder="Here, you have to specify what you don't want in your generation "
                            "(key words).",
                value=self.session_state["negative_prompt"],
                on_change=self.on_change
            )

            # Creates the button allowing to load the default negative prompt
            st.button(
                label="Load default",
                on_click=self.on_click,
                use_container_width=True
            )

    def on_change(self):
        # Assigns the value of the text_area to the negative prompt
        self.session_state["negative_prompt"] = st.session_state[f"{self.page.id}_negative_prompt"]

    def on_click(self):
        # Loads the default negative prompt
        self.session_state["negative_prompt"] = \
            "monochrome, lowres, bad anatomy, worst quality, low quality"


# ---------- HYPER-PARAMETERS ---------- #

class HyperParameters(Component):
    """ Represents a HyperParameters. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator
    ):
        """
        Initializes a HyperParameters.

        Parameters
        ----------
            page: Page
                page of the component
            parent: st._DeltaGenerator
                parent of the component
        """
        super(HyperParameters, self).__init__(page=page, parent=parent)

        # ----- Components ----- #
        self.parent.info("bla bla")


# ---------- IMAGE GENERATION ---------- #

class ImageGeneration(Component):
    """ Represents an ImageGeneration. """
    def __init__(
        self,
        page: Page,
        parent: st._DeltaGenerator
    ):
        """
        Initializes an ImageGeneration.

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

        # ----- Session state ----- #
        if "generated_images" not in self.session_state:
            self.session_state["generated_images"] = [np.zeros((480, 640, 3)) for _ in range(6)]

        # ----- Components ----- #
        with self.parent.expander(label="", expanded=True):
            cols = st.columns([1, 1, 1])

            # For each generated image
            for idx, image in enumerate(self.session_state["generated_images"]):
                # Display the generated image
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
        with self.parent.form(key=f"{self.page.id}_form_3"):
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

        # If no mask has been uploaded
        if len(self.session_state["images"]) == 0:
            # Generates images using basic StableDiffusion
            st.session_state.backend.check_stable_diffusion()
            generated_images = st.session_state.backend.stable_diffusion(
                prompt=self.session_state["prompt"],
                negative_prompt=self.session_state["negative_prompt"],
                seed=1
            )

        else:
            # Retrieves the uploaded masks and their corresponding processing
            input_masks, processing_ids = list(), list()
            for idx in range(len(self.session_state["images"])):
                # If an image has been uploaded without providing the processing used
                if self.session_state["images"][idx].processing == "":
                    return

                input_masks.append(self.session_state["images"][idx])
                processing_ids.append(self.session_state["images"][idx].processing)

            # Generates images using ControlNet
            st.session_state.backend.check_control_net(processing_ids=processing_ids)
            generated_images = st.session_state.backend.control_net(
                prompt="a white dog in front of a house, best quality",
                negative_prompt="blur, monochrome, lowres, bad anatomy, worst quality, low quality",
                images=input_masks,
                seed=1
            )

        # Updates the in memory generated images
        for idx, image in enumerate(generated_images):
            self.session_state["generated_images"][idx] = np.array(image)
