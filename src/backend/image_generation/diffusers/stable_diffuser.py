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
from src.backend.image_generation.diffuser import Diffuser


class StableDiffuser(Diffuser):
    """
    Allows to generate images using StableDiffusion.

    Attributes
    ----------
        _pipeline: StableDiffusionPipeline
            diffusion pipeline
    """
    PIPELINES = {
        "StableDiffusion_v1.5": "runwayml/stable-diffusion-v1-5",
        "StableDiffusion_v2.0": "stabilityai/stable-diffusion-2",
        "StableDiffusion_v2.1": "stabilityai/stable-diffusion-2-1",
        "StableDiffusion_XL": "RamAnanth1/stable-diffusion-xl",
        "DreamLike_v1.0": "dreamlike-art/dreamlike-photoreal-1.0",
        "DreamLike_v2.0": "dreamlike-art/dreamlike-photoreal-2.0",
        "OpenJourney_v4.0": "prompthero/openjourney-v4",
        "Deliberate_v1.0": "XpucT/Deliberate",
        "RealisticVision_v2.0": "SG161222/Realistic_Vision_V2.0",
        "Anything_v4.0": "andite/anything-v4.0"
    }

    def __init__(self, pipeline_path: str):
        """
        Allows to generate images using StableDiffusion.

        Parameters
        ----------
            pipeline_path: str
                path to the pretrained pipeline
        """
        super(StableDiffuser, self).__init__(pipeline_path=pipeline_path)

    def _init_pipeline(self) -> StableDiffusionPipeline:
        """
        Initializes the diffusion pipeline.

        Returns
        ----------
            StableDiffusionPipeline
                diffusion pipeline
        """
        return StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=self.PIPELINES[self._pipeline_path],
            torch_dtype=torch.float16,
            safety_checker=None
        )

    def __call__(
            self,
            lora_path: str,
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
        Parameters
        ----------
            prompt: str
                prompt describing the images to generate
            negative_prompt: str
                prompt describing prohibition in the images to generate
            num_images: int
                number of images to generate
            width: int
                width of the images to generate
            height: int
                height of the images to generate
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
        if self._lora_path is not None or lora_path != "":
            self._lora_path = lora_path
            self._pipeline.unet.load_attn_procs(lora_path)

        # Creates the randomness controller
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)

        # Creates the starting random noise
        if latents is None:
            latents = self._randn(
                b=num_images,
                c=self._pipeline.unet.config.in_channels,
                w=width,
                h=height,
                generator=generator
            )

        # Generates the images
        return latents, self._pipeline(
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
