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

    def __init__(self, pipeline_path: str):
        """
        Initializes an object allowing to generate images using diffusion pipeline.

        Parameters
        ----------
            pipeline_path: str
                path to the pretrained pipeline
        """
        # ----- Attributes ----- #
        self._pipeline_path = pipeline_path

        # Pipeline allowing to generate images
        self._pipeline: Any = self._load_pipeline()

        # Pipeline's noise scheduler allowing to modulate the denoising
        self._pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self._pipeline.scheduler.config
        )

        # Optimizes the use of the GPU's VRAM
        self._pipeline.enable_model_cpu_offload()

    def is_different(self, pipeline_path: str) -> bool:
        """
        Verifies if the diffuser need to be re-instantiate.

        Parameters
        ----------
            pipeline_path: str
                path to the desired pipeline

        Returns
        ----------
            bool
                whether or not a re-instantiation is needed
        """
        if self._pipeline_path != pipeline_path:
            return True
        return False

    def _load_pipeline(self) -> Any:
        """
        Loads the diffusion pipeline.

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
        num_images: int,
        num_channels: int,
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
            num_channels: int
                number of channels required by the model
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
        latents = torch.randn(
            size=(
                num_images,
                num_channels,
                height // self._pipeline.vae_scale_factor,
                width // self._pipeline.vae_scale_factor
            ),
            generator=generator
        )

        # Scales the latents by the standard deviation required by the scheduler
        return latents * self._pipeline.scheduler.init_noise_sigma

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
            guidance_scale: float
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
