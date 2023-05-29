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
from diffusers import StableDiffusionPipeline

# IMPORT: project
from src.image_utils.image_generation.image_generator import ImageGenerator


class StableDiffusion(ImageGenerator):
    """
    Represents a StableDiffusion.

    Attributes
    ----------
        _pipeline: StableDiffusionPipeline
            diffusion pipeline needed to generate images
    """

    def __init__(
        self,
        pipeline_path: str = "runwayml/stable-diffusion-v1-5"
    ):
        """
        Initializes a StableDiffusionPipeline.

        Parameters
        ----------
            pipeline_path: str
                path to the pretrained pipeline
        """
        super(StableDiffusion, self).__init__()

        # ----- Attributes ----- #
        # Pipeline
        self._pipeline: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=pipeline_path,
            torch_dtype=torch.float16
        )

        # Options
        self._pipeline.enable_model_cpu_offload()

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
        """
        # Generates images
        images: List[Image.Image] = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            latents=latents,
            num_images_per_prompt=num_images,
            generator=torch.Generator(device="cpu").manual_seed(seed)
        ).images

        if num_images == 1:
            return images[0]
        return images
