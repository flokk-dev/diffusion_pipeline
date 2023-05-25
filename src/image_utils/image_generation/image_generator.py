"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
from PIL import Image

# IMPORT: data processing
import torch

# IMPORT: deep learning
from diffusers import DiffusionPipeline


class ImageGenerator:
    """
    Represents an ImageGenerator.

    Attributes
    ----------
        _pipeline: DiffusionPipeline
            diffusion pipeline needed to generate images
    """

    def __init__(self):
        """ Initializes an ImageGenerator. """
        # ----- Attributes ----- #
        # Pipeline
        self._pipeline: DiffusionPipeline = None

    def __call__(
        self,
        prompt: str,
        negative_prompt: str,
        latents: torch.Tensor = None,
        num_images: int = 1,
        seed: int = None
    ) -> Image.Image | List[Image.Image]:
        """
        Parameters
        ----------
            prompt: str
                prompt from which to generate images
            negative_prompt: str
                prompt to avoid during the generation
            latents: torch.Tensor
                random noise from which to generate images
            num_images: int
                number of images to generate
            seed: int
                random seed used for the generation

        Returns
        ----------
            Image.Image | List[Image.Image]
                generated images

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()
