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
from torchvision.transforms import Compose, ToTensor, Resize

# IMPORT: deep learning
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# IMPORT: project
from src.backend.image_generation.diffuser import Diffuser


class ControlNetDiffuser(Diffuser):
    """
    Allows to generate images using ControlNets.

    Attributes
    ----------
        _pipeline: StableDiffusionControlNetPipeline
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

    CONTROLNETS = {
        "Canny": "lllyasviel/sd-controlnet-canny",
        "Depth": "lllyasviel/sd-controlnet-depth",
        "Hed": "lllyasviel/sd-controlnet-hed",
        "Lineart": "lllyasviel/control_v11p_sd15_lineart",
        "Lineart anime": "lllyasviel/control_v11p_sd15s2_lineart_anime",
        "MLSD": "lllyasviel/sd-controlnet-mlsd",
        "Normal BAE": "lllyasviel/sd-controlnet-normalbae",
        "OpenPose": "lllyasviel/sd-controlnet-openpose",
        "Seg": "lllyasviel/control_v11p_sd15_seg",
        "Shuffle": "lllyasviel/control_v11e_sd15_shuffle"
    }

    def __init__(self, pipeline_path: str, controlnets: List[str]):
        """
        Allows to generate images using ControlNets.

        Parameters
        ----------
            pipeline_path: str
                path to the pretrained pipeline
            controlnets: List[str]
                list of the controlnets
        """
        # ----- Attributes ----- #
        self._controlnets: List[str] = controlnets

        # ----- Mother Class ----- #
        super(ControlNetDiffuser, self).__init__(pipeline_path=pipeline_path)

    def _init_pipeline(self) -> StableDiffusionControlNetPipeline:
        """
        Initializes the diffusion pipeline.

        Returns
        ----------
            StableDiffusionControlNetPipeline
                diffusion pipeline
        """
        # Loads the controlnets
        controlnets: List[ControlNetModel] = [
            ControlNetModel.from_pretrained(
                pretrained_model_name_or_path=self.CONTROLNETS[controlnet_id],
                torch_dtype=torch.float16
            )
            for controlnet_id
            in self._controlnets
        ]

        # Loads the diffusion pipeline
        return StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=self.PIPELINES[self._pipeline_path],
            controlnet=controlnets,
            torch_dtype=torch.float16,
            safety_checker=None
        )

    def is_different(self, pipeline_path: str, controlnets: List[str]) -> bool:
        return not self._pipeline_path == pipeline_path or not self._controlnets == controlnets

    def __call__(
            self,
            images: List[torch.Tensor],
            prompt: str,
            negative_prompt: str = "",
            num_images: int = 1,
            width: int = 512,
            height: int = 512,
            num_steps: int = 50,
            guidance_scale: float = 7.5,
            weights: List[float] = None,
            latents: torch.Tensor = None,
            seed: int = None
    ) -> Tuple[torch.Tensor, List[Image.Image]]:
        """
        Parameters
        ----------
            images: List[torch.Tensor]
                images allowing to control/guide the generation
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
        # Verifies the input images
        processing: Compose = Compose([ToTensor(), Resize((height, width), antialias=True)])
        images = [processing(image).unsqueeze(0) for image in images]

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
            image=images,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=weights,
            latents=latents.type(torch.float16),
            generator=generator
        ).images
