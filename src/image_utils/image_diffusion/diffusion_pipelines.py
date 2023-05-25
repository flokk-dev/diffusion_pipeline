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
from diffusers import \
    StableDiffusionPipeline as HFStableDiffusionPipeline, \
    StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# IMPORT: project
from .diffusion_pipeline import DiffusionPipeline


class StableDiffusionPipeline(DiffusionPipeline):
    """
    Represents a StableDiffusionPipeline.

    Attributes
    ----------
        _pipeline: HFDiffusionPipeline
            diffusion pipeline needed to generate images
    """

    def __init__(
        self,
        pipeline_path: str = "runwayml/stable-diffusion-v1-5"
    ):
        """
        Initializes a StableDiffusionPipeline.

        Parameters
        ----------
            pipeline_path: str
                path to the pretrained pipeline
        """
        # ----- Mother Class ----- #
        super(StableDiffusionPipeline, self).__init__(pipeline_path)

        # ----- Attributes ----- #
        # Pipeline
        self._pipeline: HFStableDiffusionPipeline = HFStableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=pipeline_path,
            torch_dtype=torch.float16
        )

        # Options
        self._pipeline.enable_model_cpu_offload()

    def __call__(
        self,
        prompt: str,
        negative_prompt: str,
        latents: torch.Tensor = None,
        num_images: int = 1,
        seed: int = None
    ) -> Image.Image | List[Image.Image]:
        """
        Parameters
        ----------
            prompt: str
                prompt from which to generate images
            negative_prompt: str
                prompt to avoid during the generation
            latents: torch.Tensor
                random noise from which to generate images
            num_images: int
                number of images to generate
            seed: int
                random seed used for the generation

        Returns
        ----------
            Image.Image | List[Image.Image]
                generated images
        """
        # Generates images
        images: List[Image.Image] = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            latents=latents,
            num_images_per_prompt=num_images,
            generator=torch.Generator(device="cpu").manual_seed(seed)
        ).images

        if num_images == 1:
            return images[0]
        return images


class ControlDiffusionPipeline(DiffusionPipeline):
    """
    Represents a ControlDiffusionPipeline.

    Attributes
    ----------
        _pipeline: StableDiffusionControlNetPipeline
            diffusion pipeline needed to generate images
    """

    def __init__(
        self,
        control_net: ControlNetModel | List[ControlNetModel],
        pipeline_path: str = "runwayml/stable-diffusion-v1-5"
    ):
        """
        Initializes a ControlDiffusionPipeline.

        Parameters
        ----------
            control_net: ControlNetModel | List[ControlNetModel]
                control net
            pipeline_path: str
                path to the pretrained pipeline
        """
        # ----- Mother Class ----- #
        super(ControlDiffusionPipeline, self).__init__(pipeline_path)

        # ----- Attributes ----- #
        # Pipeline
        self._pipeline: StableDiffusionControlNetPipeline = StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=pipeline_path,
            controlnet=control_net,
            torch_dtype=torch.float16
        )

        # Scheduler
        self._pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self._pipeline.scheduler.config
        )

        # Options
        self._pipeline.enable_model_cpu_offload()

    def __call__(
        self,
        prompt: str,
        negative_prompt: str,
        images: List[torch.Tensor],
        latents: torch.Tensor = None,
        weights: List[int] = None,
        num_images: int = 1,
        seed: int = None
    ) -> Image.Image | List[Image.Image]:
        """
        Parameters
        ----------
            prompt: str
                prompt from which to generate images
            negative_prompt: str
                prompt to avoid during the generation
            images: List[torch.Tensor]
                images from which to generate images
            latents: torch.Tensor
                random noise from which to generate images
            weights: List[int]
                weights of the control-nets
            num_images: int
                number of images to generate
            seed: int
                random seed used for the generation

        Returns
        ----------
            Image.Image | List[Image.Image]
                generated images
        """
        if weights is None:
            weights = [1.0] * len(images)

        # Generates images
        images: List[Image.Image] = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            images=images,
            latents=latents,
            num_images_per_prompt=num_images,
            controlnet_conditioning_scale=weights,
            generator=torch.Generator(device="cpu").manual_seed(seed)
        ).images

        if num_images == 1:
            return images[0]
        return images
