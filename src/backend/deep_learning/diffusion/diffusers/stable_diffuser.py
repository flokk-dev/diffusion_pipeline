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
from src.backend.deep_learning.diffusion import Diffuser


class StableDiffuser(Diffuser):
    """
    Represents an object allowing to generate images using only StableDiffusion.

    Attributes
    ----------
        _pipeline: StableDiffusionPipeline
            diffusion pipeline needed to generate images
    """

    def __init__(self, pipeline_path: str = "runwayml/stable-diffusion-v1-5"):
        """
        Initializes an object allowing to generate images using only StableDiffusion.

        Parameters
        ----------
            pipeline_path: str
                path to the pretrained pipeline
        """
        super(StableDiffuser, self).__init__(pipeline_path=pipeline_path)

    def _load_pipeline(self, pipeline_path: str) -> StableDiffusionPipeline:
        """
        Loads the diffusion pipeline.

        Parameters
        ----------
            pipeline_path: str
                path to the pretrained pipeline

        Returns
        ----------
            DiffusionPipeline
                pretrained pipeline
        """
        return StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=pipeline_path,
            torch_dtype=torch.float16,
            safety_checker=None
        )

    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
        width: int = 512,
        height: int = 512,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        latents: torch.Tensor = None,
        seed: int = None
    ) -> Tuple[torch.Tensor, List[Image.Image]]:
        """
        Generates images.

        Parameters
        ----------
            prompt: str
                prompt describing the output image
            negative_prompt: str
                prompt describing what to avoid in the output image
            num_images: int
                number of images to generate
            width: int
                width of the output image
            height: int
                height of the output image
            num_steps: int
                number of denoising steps
            guidance_scale: int
                strength of the prompt during the generation
            latents: torch.Tensor
                random noise from which to start the generation
            seed: int
                random seed to use during the generation

        Returns
        ----------
            torch.FloatTensor
                latents from which the image generation has started
            List[Image.Image]
                generated images
        """
        # Creates the object that controls the randomness
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)

        # Creates the latents from which the image generation will start
        if latents is None:
            latents = self._latents_generator(num_images, width, height, generator)

        # Generates images
        generated_images = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            latents=latents.type(torch.float16),
            generator=generator
        ).images

        return latents, generated_images
