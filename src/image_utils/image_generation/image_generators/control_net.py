"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *
import PIL

# IMPORT: data processing
import torch
from torchvision import transforms

# IMPORT: deep learning
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# IMPORT: project
from src.image_utils.image_generation.image_generator import ImageGenerator


class ControlNet(ImageGenerator):
    """
    Represents a ControlNet.

    Attributes
    ----------
        _pipeline: StableDiffusionControlNetPipeline
            diffusion pipeline needed to generate images
    """
    CONTROL_NETS_IDS = {
        "canny": "lllyasviel/sd-controlnet-canny",
        "depth": "lllyasviel/sd-controlnet-depth",
        "hed": "lllyasviel/sd-controlnet-hed",
        "mlsd": "lllyasviel/sd-controlnet-mlsd",
        "normal": "lllyasviel/sd-controlnet-normal",
        "openpose": "lllyasviel/sd-controlnet-openpose",
        "scribble": "lllyasviel/sd-controlnet-scribble",
        "seg": "lllyasviel/sd-controlnet-seg",
    }

    def __init__(
        self,
        processing_ids: List[str],
        pipeline_path: str = "runwayml/stable-diffusion-v1-5"
    ):
        """
        Initializes a ControlNet.

        Parameters
        ----------
            processing_ids: List[str]
                ...
            pipeline_path: str
                path to the pretrained pipeline
        """
        super(ControlNet, self).__init__()

        # ----- Attributes ----- #
        # ControlNets
        control_nets = [
            ControlNetModel.from_pretrained(
                pretrained_model_name_or_path=self.CONTROL_NETS_IDS[processing_id],
                torch_dtype=torch.float16
            )
            for processing_id
            in processing_ids
        ]

        # Pipeline
        self._pipeline: StableDiffusionControlNetPipeline = StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=pipeline_path,
            controlnet=control_nets,
            torch_dtype=torch.float16,
            safety_checker=None
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
        images: List[torch.FloatTensor],
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        latents: torch.Tensor = None,
        weights: List[int] = None,
        num_images: int = 1,
        seed: int = 0
    ) -> PIL.Image.Image | List[PIL.Image.Image]:
        """
        Parameters
        ----------
            prompt: str
                prompt from which to generate images
            negative_prompt: str
                prompt to avoid during the generation
            images: List[torch.Tensor]
                images from which to generate images
            width: int
                ...
            height: int
                ...
            num_steps: int
                ...
            guidance_scale: int
                ...
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
            PIL.Image.Image | List[PIL.Image.Image]
                generated images
        """
        # If the images are not tensors
        if not isinstance(images[0], torch.FloatTensor):
            processing = transforms.Compose([
                transforms.ToTensor(), transforms.Resize((height, width))
            ])
            images = [processing(image).unsqueeze(0) for image in images]

        # If the weights have not been provided
        if weights is None:
            weights = [1.0] * len(images)

        # Generates images
        return self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=images,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            latents=latents,
            num_images_per_prompt=num_images,
            controlnet_conditioning_scale=weights,
            generator=torch.Generator(device="cpu").manual_seed(seed)
        ).images
