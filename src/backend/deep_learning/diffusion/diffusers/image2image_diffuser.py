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
from src.backend.deep_learning.diffusion import Diffuser


class Image2ImageDiffuser(Diffuser):
    """
    Represents an object allowing to generate images using image to image (StableDiffusion).

    Attributes
    ----------
        _pipeline: StableDiffusionPipeline
            diffusion pipeline needed to generate images
    """
    PIPELINES = {
        "StableDiffusion_v1.5": "runwayml/stable-diffusion-v1-5",
        "StableDiffusion_v2.0": "stabilityai/stable-diffusion-2",
        "DreamLike_v1.0": "dreamlike-art/dreamlike-photoreal-1.0",
        "DreamLike_v2.0": "dreamlike-art/dreamlike-photoreal-2.0",
        "OpenJourney_v4": "prompthero/openjourney-v4",
        "Deliberate_v1": "XpucT/Deliberate",
        "RealisticVision_v2.0": "SG161222/Realistic_Vision_V2.0",
        "Anything_v4": "andite/anything-v4.0"
    }

    def __init__(self, pipeline_path: str):
        """
        Initializes an object allowing to generate images using image to image (StableDiffusion).

        Parameters
        ----------
            pipeline_path: str
                path to the pretrained pipeline
        """
        super(Image2ImageDiffuser, self).__init__(pipeline_path=pipeline_path)

    def _load_pipeline(self) -> StableDiffusionImg2ImgPipeline:
        """
        Loads the diffusion pipeline.

        Returns
        ----------
            DiffusionPipeline
                pretrained pipeline
        """
        return StableDiffusionImg2ImgPipeline.from_pretrained(
            pretrained_model_name_or_path=self.PIPELINES[self._pipeline_path],
            torch_dtype=torch.float16,
            safety_checker=None
        )

    def __call__(
        self,
        prompt: str,
        image: torch.Tensor,
        strength: float = 0.8,
        negative_prompt: str = "",
        num_images: int = 1,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = None
    ) -> List[Image.Image]:
        """
        Generates images.

        Parameters
        ----------
            prompt: str
                prompt describing the output image
            image: torch.Tensor
                image on which the generation will be based
            strength: float
                strength of the starting image
            negative_prompt: str
                prompt describing what to avoid in the output image
            num_images: int
                number of images to generate
            num_steps: int
                number of denoising steps
            guidance_scale: float
                strength of the prompt during the generation
            seed: int
                random seed to use during the generation

        Returns
        ----------
            List[Image.Image]
                generated images
        """
        # If the images are not tensors
        if not isinstance(image, torch.FloatTensor):
            image = ToTensor()(image).unsqueeze(0)

        # Creates the object that controls the randomness
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)

        # Generates images
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
