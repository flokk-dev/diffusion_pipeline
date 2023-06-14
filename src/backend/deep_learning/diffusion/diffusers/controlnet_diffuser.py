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
from src.backend.deep_learning.diffusion import Diffuser


class ControlNetDiffuser(Diffuser):
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

    def __init__(self, pipeline_path: str, controlnet_ids: List[str]):
        """
        Initializes an object allowing to generate images using ControlNet + StableDiffusion.

        Parameters
        ----------
            pipeline_path: str
                path to the pretrained pipeline
            controlnet_ids: List[str]
                list of the ControlNet to use
        """
        # ----- Attributes ----- #
        # List of the ControlNet ids composing the pipeline
        self._controlnet_ids = controlnet_ids

        # ----- Mother Class ----- #
        super(ControlNetDiffuser, self).__init__(pipeline_path=pipeline_path)

    def is_different(self, pipeline_path: str, controlnet_ids: List[str]) -> bool:
        """
        Verifies if the diffuser need to be re-instantiate.

        Parameters
        ----------
            pipeline_path: str
                path to the desired pipeline
            controlnet_ids: List[str]
                list of the desired ControlNet ids

        Returns
        ----------
            bool
                whether or not a re-instantiation is needed
        """
        if pipeline_path != self._pipeline_path:
            return True

        if controlnet_ids != self._controlnet_ids:
            return True

        return False

    def _load_pipeline(self) -> StableDiffusionControlNetPipeline:
        """
        Loads the diffusion pipeline.

        Returns
        ----------
            DiffusionPipeline
                pretrained pipeline
        """
        # List of the ControlNet composing the pipeline
        control_nets = [
            ControlNetModel.from_pretrained(
                pretrained_model_name_or_path=self.CONTROLNET_IDS[controlnet_id],
                torch_dtype=torch.float16
            )
            for controlnet_id
            in self._controlnet_ids
        ]

        # Load the pipeline
        return StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=self.PIPELINES[self._pipeline_path],
            controlnet=control_nets,
            torch_dtype=torch.float16,
            safety_checker=None
        )

    def __call__(
        self,
        prompt: str,
        images: List[torch.Tensor],
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
            images: List[torch.Tensor]
                ControlNet masks to guide the generation.
            negative_prompt: str
                prompt describing what to avoid in the output image
            num_images: int
                number of images to generate
            width: int
                width of the output image
            height: int
                height of the output image
            num_steps: int
                number of denoising steps
            guidance_scale: float
                strength of the prompt during the generation
            weights: List[float]
                weight of each ControlNet
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
        processing: Compose = Compose([ToTensor(), Resize((height, width))])

        for idx, image in enumerate(images):
            if not isinstance(image, torch.FloatTensor):
                images[idx] = processing(image).unsqueeze(0)

        # If the weights have not been provided
        if weights is None:
            weights = [1.0] * len(images)

        # Creates the object that controls the randomness
        generator = None if seed is None else torch.Generator(device="cpu").manual_seed(seed)

        # Creates the latents from which the image generation will start
        if latents is None:
            latents = self._latents_generator(
                num_images=num_images,
                num_channels=self._pipeline.unet.config.in_channels,
                width=width,
                height=height,
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
