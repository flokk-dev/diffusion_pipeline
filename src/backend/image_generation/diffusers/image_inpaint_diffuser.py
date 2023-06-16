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
from diffusers import StableDiffusionInpaintPipeline

# IMPORT: project
from src.backend.image_generation.diffuser import Diffuser


class ImageInpaintDiffuser(Diffuser):
    """
    Allows to generate images using image InPainting.

    Attributes
    ----------
        _pipeline: StableDiffusionInpaintPipeline
            diffusion pipeline
    """
    PIPELINES = {
        "StableDiffusion_v1.2": "stabilityai/stable-diffusion-inpainting",
        "StableDiffusion_v2.0": "stabilityai/stable-diffusion-2-inpainting"
    }

    def __init__(self, pipeline_path: str):
        """
        Allows to generate images using image InPainting.

        Parameters
        ----------
            pipeline_path: str
                path to the pretrained pipeline
        """
        super(ImageInpaintDiffuser, self).__init__(pipeline_path=pipeline_path)

    def _init_pipeline(self) -> StableDiffusionInpaintPipeline:
        """
        Initializes the diffusion pipeline.

        Returns
        ----------
            StableDiffusionPipeline
                diffusion pipeline
        """
        return StableDiffusionInpaintPipeline.from_pretrained(
            pretrained_model_name_or_path=self.PIPELINES[self._pipeline_path],
            torch_dtype=torch.float16,
            safety_checker=None
        )

    def __call__(
            self,
            image: torch.Tensor,
            mask: torch.Tensor,
            prompt: str,
            negative_prompt: str = "",
            num_images: int = 1,
            num_steps: int = 50,
            guidance_scale: float = 7.5,
            latents: torch.Tensor = None,
            seed: int = None
    ) -> Tuple[torch.Tensor, List[Image.Image]]:
        """
        Parameters
        ----------
            image: List[torch.Tensor]
                original image
            mask: List[torch.Tensor]
                mask specifying where to generate
            prompt: str
                prompt describing the images to generate
            negative_prompt: str
                prompt describing prohibition in the images to generate
            num_images: int
                number of images to generate
            num_steps: int
                number of denoising steps to go through
            guidance_scale: float
                strength of the prompts during the generation
            latents: torch.Tensor
                starting random noise
            seed: int
                seed of the randomness

        Returns
        ----------
            torch.FloatTensor
                starting random noise
            List[Image.Image]
                generated images
        """
        # Verifies the input images
        processing: ToTensor = ToTensor()

        image = processing(image).unsqueeze(0)
        mask = processing(mask)[0].unsqueeze(0).unsqueeze(0)

        # Creates the randomness controller
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)

        # Creates the latents from which the image generation will start
        if latents is None:
            latents = self._randn(
                b=num_images,
                c=self._pipeline.vae.config.latent_channels,
                w=image.shape[-1],
                h=image.shape[-2],
                generator=generator
            )

        # Generates the images
        return latents, self._pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            width=image.shape[-1],
            height=image.shape[-2],
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            latents=latents.type(torch.float16),
            generator=generator
        ).images
