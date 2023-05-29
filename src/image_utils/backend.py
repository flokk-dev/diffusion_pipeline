"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: project
from .image_processing import ImageProcessorManager
from .image_captioning import ClipInterrogator, Promptist
from .image_generation import StableDiffusion, ControlNet


class Backend:
    """
    Represents a Backend.

    Attributes
    ----------
        image_processing_manager: ImageProcessorManager
            manager of the different image processing method
        clip_interrogator: ClipInterrogator
            ...
        promptist: Promptist
            ...
        stable_diffusion: StableDiffusion
            ...
        control_net: ControlNet
            ...
    """
    def __init__(
        self
    ):
        """ Initializes a Backend. """
        # ----- Attributes ----- #
        # Image processing
        self.image_processing_manager = ImageProcessorManager()

        # Image captioning
        self.clip_interrogator = ClipInterrogator
        self.promptist = Promptist

        # Image generation
        self.stable_diffusion = StableDiffusion
        self.control_net = ControlNet

    def check_clip_interrogator(self):
        if isinstance(self.clip_interrogator, type):
            self.clip_interrogator = self.clip_interrogator()

    def check_promptist(self):
        if isinstance(self.promptist, type):
            self.promptist = self.promptist()

    def check_stable_diffusion(self):
        if isinstance(self.stable_diffusion, type):
            self.stable_diffusion = self.stable_diffusion()

    def check_control_net(
        self,
        processing_ids: List[str]
    ):
        if isinstance(self.control_net, type):
            self.control_net = self.control_net(processing_ids=processing_ids)

    def reset_control_net(self):
        """ Resets the ControlNet. """
        self.control_net = ControlNet
