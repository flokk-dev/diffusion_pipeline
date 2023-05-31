"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: project
from .image_processing import ImageProcessingManager
from .deep_learning.prompt import ClipInterrogator, Promptist
from .deep_learning.diffusion import StableDiffusion, ControlNetStableDiffusion


class Backend:
    """
    Represents the backend of the application.

    Attributes
    ----------
        image_processing_manager: ImageProcessingManager
            manager of the different image processing available
        clip_interrogator: ClipInterrogator
            object allowing to transform the image into a prompt
        promptist: Promptist
            object allowing to improve a prompt
        stable_diffusion: StableDiffusion
            object allowing to generate images using only StableDiffusion
        controlnet: ControlNetStableDiffusion
            object allowing to generate images using ControlNet + StableDiffusion
    """
    def __init__(self):
        """ Initializes the backend of the application. """
        # ----- Attributes ----- #
        # Manager of the different image processing available
        self.image_processing_manager = ImageProcessingManager()

        # Object allowing to transform the image into a prompt
        self.clip_interrogator: type | ClipInterrogator = ClipInterrogator

        # Object allowing to improve a prompt
        self.promptist: type | Promptist = Promptist

        # Object allowing to generate images using only StableDiffusion
        self.stable_diffusion: type | StableDiffusion = StableDiffusion

        # Object allowing to generate images using ControlNet + StableDiffusion
        self.controlnet: type | ControlNetStableDiffusion = ControlNetStableDiffusion

    def check_clip_interrogator(self):
        """ Instantiates the ClipInterrogator if he is not. """
        if isinstance(self.clip_interrogator, type):
            self.clip_interrogator = self.clip_interrogator()

    def check_promptist(self):
        """ Instantiates the Promptist if he is not. """
        if isinstance(self.promptist, type):
            self.promptist = self.promptist()

    def check_stable_diffusion(self):
        """ Instantiates the StableDiffusion if he is not. """
        if isinstance(self.stable_diffusion, type):
            self.stable_diffusion = self.stable_diffusion()

    def check_controlnet(self, controlnet_ids: List[str]):
        """
        Instantiates the ControlNetStableDiffusion if the ControlNet ids have been modified.

        Parameters
        ----------
            controlnet_ids: List[str]
                list of the ControlNet to use
        """
        if isinstance(self.controlnet, type) or controlnet_ids != self.controlnet.controlnet_ids:
            self.controlnet = ControlNetStableDiffusion(controlnet_ids=controlnet_ids)
