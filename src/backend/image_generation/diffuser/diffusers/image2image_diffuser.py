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
from diffusers import StableDiffusionImg2ImgPipeline

# IMPORT: project
from src.backend.image_generation.diffuser import Diffuser


class Image2ImageDiffuser(Diffuser):
    """
    Allows to generate images using Image2Image.

    Attributes
    ----------
        _pipeline: StableDiffusionPipeline
            diffusion pipeline
    """
    PIPELINES = {
        "StableDiffusion_v1.5": "runwayml/stable-diffusion-v1-5",
        "StableDiffusion_v2.0": "stabilityai/stable-diffusion-2",
        "StableDiffusion_v2.1": "stabilityai/stable-diffusion-2-1",
        "DreamLike_v1.0": "dreamlike-art/dreamlike-photoreal-1.0",
        "DreamLike_v2.0": "dreamlike-art/dreamlike-photoreal-2.0",
        "OpenJourney_v4.0": "prompthero/openjourney-v4",
        "Deliberate_v1.0": "XpucT/Deliberate",
        "RealisticVision_v2.0": "SG161222/Realistic_Vision_V2.0",
        "Anything_v4.0": "andite/anything-v4.0"
    }

    def __init__(self, pipeline_path: str):
        """
        Allows to generate images using Image2Image.

        Parameters
        ----------
            pipeline_path: str
                path to the pretrained pipeline
        """
        super(Image2ImageDiffuser, self).__init__(pipeline_path=pipeline_path)

    def _init_pipeline(self) -> StableDiffusionImg2ImgPipeline:
        """
        Initializes the diffusion pipeline.

        Returns
        ----------
            StableDiffusionPipeline
                diffusion pipeline
        """
        return StableDiffusionImg2ImgPipeline.from_pretrained(
            pretrained_model_name_or_path=self.PIPELINES[self._pipeline_path],
            torch_dtype=torch.float16,
            safety_checker=None
        )

    def __call__(
        self,
        image: torch.Tensor,
        prompt: str,
        strength: float = 0.8,
        negative_prompt: str = "",
        num_images: int = 1,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = None
    ) -> List[Image.Image]:
        """
        Parameters
        ----------
            image: List[torch.Tensor]
                original image
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
        image = ToTensor()(image).unsqueeze(0)

        # Creates the randomness controller
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)

        # Generates the images
        return self._pipeline(
            prompt=prompt,
            image=image,
            strength=1.0 - strength,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images
