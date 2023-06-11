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
from torchvision.transforms import ToTensor

# IMPORT: deep learning
from diffusers import StableDiffusionInstructPix2PixPipeline

# IMPORT: project
from src.backend.deep_learning.diffusion import Diffuser


class Pix2PixDiffuser(Diffuser):
    """
    Represents an object allowing to generate images using pix to pix (StableDiffusion).

    Attributes
    ----------
        _pipeline: StableDiffusionPipeline
            diffusion pipeline needed to generate images
    """
    PIPELINES = {
        "Instruct_Pix2Pix": "timbrooks/instruct-pix2pix"
    }

    def __init__(self, pipeline_path: str):
        """
        Initializes an object allowing to generate images using pix to pix (StableDiffusion).

        Parameters
        ----------
            pipeline_path: str
                path to the pretrained pipeline
        """
        super(Pix2PixDiffuser, self).__init__(pipeline_path=pipeline_path)

    def _load_pipeline(self) -> StableDiffusionInstructPix2PixPipeline:
        """
        Loads the diffusion pipeline.

        Returns
        ----------
            DiffusionPipeline
                pretrained pipeline
        """
        return StableDiffusionInstructPix2PixPipeline.from_pretrained(
            pretrained_model_name_or_path=self.PIPELINES[self._pipeline_path],
            torch_dtype=torch.float16,
            safety_checker=None
        )

    def __call__(
        self,
        prompt: str,
        image: torch.Tensor,
        negative_prompt: str = "",
        num_images: int = 1,
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
            image: torch.Tensor
                image to modify
            negative_prompt: str
                prompt describing what to avoid in the output image
            num_images: int
                number of images to generate
            num_steps: int
                number of denoising steps
            guidance_scale: float
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
        # If the images are not tensors
        if not isinstance(image, torch.FloatTensor):
            image = ToTensor()(image).unsqueeze(0)

        # Creates the object that controls the randomness
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)

        # Creates the latents from which the image generation will start
        height, width = image.shape[-2:]
        if latents is None:
            latents = self._latents_generator(
                num_images=num_images,
                num_channels=self._pipeline.vae.config.latent_channels,
                width=width,
                height=height,
                generator=generator
            )

        # Generates images
        generated_images = self._pipeline(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            latents=latents.type(torch.float16),
            generator=generator
        ).images

        return latents, generated_images
