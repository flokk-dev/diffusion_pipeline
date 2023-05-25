"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: project
from .image_generator import ImageGenerator
from .image_generators import StableDiffusion, ControlNet


class ImageGeneratorManager(Dict):
    """ Represents a ImageGeneratorManager. """
    def __init__(
        self
    ):
        """ Initializes a ImageGeneratorManager. """
        # ----- Mother class ----- #
        super(ImageGeneratorManager, self).__init__()

        # Image generators
        self["stable_diffusion"]: StableDiffusion = StableDiffusion
        self["control_net"]: ControlNet = ControlNet

    def __call__(
        self,
        generator_id: str
    ) -> ImageGenerator:
        """
        Returns the ImageGenerator of the specified id.

        Parameters
        ----------
            generator_id: str
                id of the ImageGenerator to use

        Returns
        ----------
            ImageGenerator
                desired ImageGenerator
        """
        if isinstance(self[ImageGenerator], type):
            self[ImageGenerator] = self[ImageGenerator]()

        # Generates image
        return self[ImageGenerator]
