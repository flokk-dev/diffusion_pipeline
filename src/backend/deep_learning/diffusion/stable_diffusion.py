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


class StableDiffusion:
    """
    Represents an object allowing to generate images using only StableDiffusion.

    Attributes
    ----------
        _pipeline: StableDiffusionPipeline
            diffusion pipeline needed to generate images
    """

    def __init__(self, path: str = "runwayml/stable-diffusion-v1-5"):
        """
        Initializes an object allowing to generate images using only StableDiffusion.

        Parameters
        ----------
            path: str
                path to the pretrained pipeline
        """
        # ----- Attributes ----- #
        # Pipeline allowing to generate images
        self._pipeline: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=path,
            torch_dtype=torch.float16
        )

        # Optimizes the use of the GPU's VRAM
        self._pipeline.enable_model_cpu_offload()

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
            num_images: int
                number of images to generate

        Returns
        ----------
            torch.FloatTensor
                latents from which the image generation has started
            List[Image.Image]
                generated images
        """
        # Creates the random generator and the latents from which to start the image generation
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
        if latents is None:
            latents: torch.Tensor = torch.randn(
                size=(
                    num_images,
                    self._pipeline.unet.config.in_channels,
                    height // self._pipeline.vae_scale_factor,
                    width // self._pipeline.vae_scale_factor
                ),
                generator=None
            )

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
