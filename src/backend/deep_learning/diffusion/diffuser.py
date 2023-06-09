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
from diffusers import UniPCMultistepScheduler


class Diffuser:
    """
    Represents an object allowing to generate images using diffusion pipeline.

    Attributes
    ----------
        _pipeline: StableDiffusionPipeline
            diffusion pipeline needed to generate images
    """
    MODEL = [
        "runwayml/stable-diffusion-v1-5"
    ]

    def __init__(self, pipeline_path: str = "runwayml/stable-diffusion-v1-5"):
        """
        Initializes an object allowing to generate images using diffusion pipeline.

        Parameters
        ----------
            pipeline_path: str
                path to the pretrained pipeline
        """
        # ----- Attributes ----- #
        # Pipeline allowing to generate images
        self._pipeline: Any = self._load_pipeline(pipeline_path=pipeline_path)

        # Pipeline's noise scheduler allowing to modulate the denoising
        self._pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self._pipeline.scheduler.config
        )

        # Optimizes the use of the GPU's VRAM
        self._pipeline.enable_model_cpu_offload()

    def _load_pipeline(self, pipeline_path: str) -> Any:
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

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()

    def _latents_generator(
        self,
        num_images,
        width: int,
        height: int,
        generator: torch.Generator | None
    ) -> torch.Tensor:
        """
        Generates images.

        Parameters
        ----------
            num_images: int
                number of images to generate
            width: int
                width of the output image
            height: int
                height of the output image
            generator: torch.Generator | None
                object that controls the randomness

        Returns
        ----------
            torch.Tensor
                latents from which the image generation has started
        """
        # Creates the latents from which the image generation will start
        return torch.randn(
            size=(
                num_images,
                self._pipeline.unet.config.in_channels,
                height // self._pipeline.vae_scale_factor,
                width // self._pipeline.vae_scale_factor
            ),
            generator=generator
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

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()
