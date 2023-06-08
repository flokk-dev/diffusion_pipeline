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
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler


class ControlNetStableDiffusion:
    """
    Represents an object allowing to generate images using ControlNet + StableDiffusion.

    Attributes
    ----------
        _pipeline: StableDiffusionControlNetPipeline
            diffusion pipeline needed to generate images
    """
    CONTROLNET_IDS = {
        "canny": "lllyasviel/sd-controlnet-canny",
        "midas": "lllyasviel/sd-controlnet-depth",
        "hed": "lllyasviel/sd-controlnet-hed",
        "mlsd": "lllyasviel/sd-controlnet-mlsd",
        "normal": "lllyasviel/sd-controlnet-normal",
        "openpose": "lllyasviel/sd-controlnet-openpose",
        "scribble": "lllyasviel/sd-controlnet-scribble",
        "seg": "lllyasviel/sd-controlnet-seg",
    }

    def __init__(self, controlnet_ids: List[str], path: str = "runwayml/stable-diffusion-v1-5"):
        """
        Initializes an object allowing to generate images using ControlNet + StableDiffusion.

        Parameters
        ----------
            controlnet_ids: List[str]
                list of the ControlNet to use
            path: str
                path to the pretrained pipeline
        """
        # ----- Attributes ----- #
        # List of the ControlNet ids composing the pipeline
        self.controlnet_ids = controlnet_ids

        # List of the ControlNet composing the pipeline
        control_nets = [
            ControlNetModel.from_pretrained(
                pretrained_model_name_or_path=self.CONTROLNET_IDS[controlnet_id],
                torch_dtype=torch.float16
            )
            for controlnet_id
            in controlnet_ids
        ]

        # Pipeline allowing to generate images
        self._pipeline: StableDiffusionControlNetPipeline = StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=path,
            controlnet=control_nets,
            torch_dtype=torch.float16,
            safety_checker=None
        )

        # Pipeline's noise scheduler allowing to modulate the noising thing
        self._pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self._pipeline.scheduler.config
        )

        # Optimizes the use of the GPU's VRAM
        self._pipeline.enable_model_cpu_offload()

    def __call__(
        self,
        prompt: str,
        images: List[torch.FloatTensor],
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
        Generates images.

        Parameters
        ----------
            prompt: str
                prompt describing the output image
            negative_prompt: str
                prompt describing what to avoid in the output image
            images: List[torch.Tensor]
                ControlNet masks to guide the generation.
            width: int
                width of the output image
            height: int
                height of the output image
            num_steps: int
                number of denoising steps
            guidance_scale: int
                strength of the prompt during the generation
            weights: List[float]
                weight of each ControlNet
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
        """
        # If the images are not tensors
        if not isinstance(images[0], torch.FloatTensor):
            processing: Compose = Compose([ToTensor(), Resize((height, width))])
            images: List[torch.Tensor] = [processing(image).unsqueeze(0) for image in images]

        # If the weights have not been provided
        if weights is None:
            weights = [1.0] * len(images)

        # Creates the random generator and the latents from which to start the image generation
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)
        if latents is None:
            latents: torch.Tensor = torch.randn(
                size=(
                    num_images,
                    self._pipeline.unet.config.in_channels,
                    height // self._pipeline.vae_scale_factor,
                    width // self._pipeline.vae_scale_factor
                ),
                generator=generator
            )

        # Generates images
        generated_images = self._pipeline(
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

        return latents, generated_images
