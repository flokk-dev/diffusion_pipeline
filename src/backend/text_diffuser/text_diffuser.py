"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *

import os
from tqdm import tqdm

from PIL import Image
import numpy as np

# IMPORT: data processing
import torch

# IMPORT: deep learning
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL
from diffusers.models.text_diffuser_unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.text_diffuser_scheduling_ddpm import DDPMScheduler

# IMPORT: project
import paths
import utils

from src.backend.text_diffuser.model import UNet


class TextDiffuser:
    """
    Allows to generate images with relevant text using diffusion.

    Attributes
    ----------
        ...: ...
            ...
    """
    PIPELINES = {
        "StableDiffusion_v1.5": "runwayml/stable-diffusion-v1-5",
    }

    def __init__(self, pipeline_path: str):
        """
        Allows to generate images with relevant text using diffusion.

        Parameters
        ----------
            pipeline_path: str
                path to the pretrained pipeline
        """
        # ----- Attributes ----- #
        # Loads the text tokenizer
        self._text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path=self.PIPELINES[pipeline_path],
            subfolder="text_encoder",
            revision="fp16"
        )
        self._text_encoder.requires_grad_(False)

        # Loads the tokenizer
        self._tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.PIPELINES[pipeline_path],
            subfolder="tokenizer",
            revision="fp16"
        )

        # Loads the image encoder
        self._vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=self.PIPELINES[pipeline_path],
            subfolder="vae",
            revision="fp16"
        ).cuda()
        self._vae.requires_grad_(False)

        # Loads the UNet
        self._unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path=os.path.join(paths.TEXT_DIFFUSER, "diffusion_backbone"),
            subfolder="unet",
            revision=None
        ).cuda()

        # Loads the noise scheduler
        self._scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path=self.PIPELINES[pipeline_path],
            subfolder="scheduler"
        )

        # Instantiates the segmentation model
        self._segmentation_model = UNet(3, 96, True).cuda()
        self._segmentation_model = torch.nn.DataParallel(self._segmentation_model)

        # Loads the pretrained weights
        self._segmentation_model.load_state_dict(
            state_dict=torch.load(
                os.path.join(paths.TEXT_DIFFUSER, "text_segmenter.pth")
            )
        )
        self._segmentation_model.eval()

    def _randn(
            self,
            b: int,
            c: int,
            w: int,
            h: int,
            generator: torch.Generator | None
    ) -> torch.Tensor:
        """
        Generates normalized random noise according to the pipeline components.

        Parameters
        ----------
            b: int
                number of random noise
            c: int
                number of channels
            w: int
                width of random noise
            h: int
                height of random noise
            generator: torch.Generator
                randomness controller

        Returns
        ----------
            torch.Tensor
                generated random noise
        """
        # Retrieves the VAE scale factor and the NoiseScheduler standard deviation
        scale_factor: int = 2 ** (len(self._vae.config.block_out_channels) - 1)
        sigma: float = self._scheduler.init_noise_sigma

        # Creates the normalized random noise
        latents: torch.Tensor = torch.randn(
            size=(b, c, h // scale_factor, w // scale_factor),
            generator=generator
        )
        return latents * sigma

    def _prepare_prompt(
            self,
            prompt: str,
            negative_prompt: str,
            num_images: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the prompts.

        Parameters
        ----------
            prompt: str
                prompt to prepare
            prompt: str
                negative prompt to prepare
            num_images: int
                number of images to generate

        Returns
        ----------
            torch.Tensor
                encoded version of the prompt
            torch.Tensor
                encoded version of the negative prompt
        """
        # Multiplies by the number of images to suit the requirement of pyTorch
        prompt: List[str] = [prompt] * num_images
        negative_prompt: List[str] = [negative_prompt] * num_images

        # Encoded the prompt
        inputs: torch.Tensor = self._tokenizer(
            prompt,
            max_length=self._tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        encoder_hidden_states: torch.Tensor = self._text_encoder(inputs)[0].cuda()

        # Encoded the negative prompt
        negative_inputs: torch.Tensor = self._tokenizer(
            negative_prompt, max_length=self._tokenizer.model_max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        ).input_ids
        encoder_negative_hidden_states: torch.Tensor = self._text_encoder(negative_inputs)[0].cuda()

        return encoder_hidden_states, encoder_negative_hidden_states

    def _segment_image(self, image: torch.Tensor, num_images: int):
        """
        Loads the character segmentation model.

        Parameters
        ----------
            image: torch.Tensor
                image containing the text to segment
            num_images: int
                number of images to generate

        Returns
        ----------
        character segmentation model
        """
        # Segments the image
        with torch.no_grad():
            segmentation_mask: torch.Tensor = self._segmentation_model(image)

        # Process the segmentation mask
        segmentation_mask: torch.Tensor = segmentation_mask.max(1)[1].squeeze(0)
        segmentation_mask: torch.Tensor = torch.nn.functional.interpolate(
            segmentation_mask.unsqueeze(0).unsqueeze(0).float(), size=(256, 256), mode='nearest'
        )

        return segmentation_mask.squeeze(1).repeat(num_images, 1, 1).long().to("cuda")

    def _prepare_text_diffuser_inputs(
            **kwargs: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare the text diffuser input mask.

        Parameters
        ----------
            prompt: str
                prompt to use
            num_images: int
                number of images to generate

        Returns
        ----------
            torch.Tensor
                ...
            torch.Tensor
                ...
            torch.Tensor
                ...

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()

    def _generate_images(
            self,
            latents: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            encoder_hidden_states_nocond: torch.Tensor,
            segmentation_mask: torch.Tensor,
            feature_mask: torch.Tensor,
            masked_feature: torch.Tensor,
            guidance_scale: float,
            generator: torch.Generator
    ) -> List[Image.Image]:
        """
        Prepare the text diffuser input mask.

        Parameters
        ----------
            latents: torch.Tensor
                ...
            encoder_hidden_states: torch.Tensor
                ...
            encoder_hidden_states_nocond: torch.Tensor
                ...
            segmentation_mask: torch.Tensor
                ...
            feature_mask: torch.Tensor
                ...
            masked_feature: torch.Tensor
                ...
            guidance_scale: float
                ...

        Returns
        ----------
            List[Image.Image]
                generated images
        """
        for t in tqdm(self._scheduler.timesteps):
            with torch.no_grad():
                latents = self._scheduler.scale_model_input(latents, t)

                # Generates the noise
                noise_pred: torch.Tensor = self._unet(
                    sample=latents,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states,
                    segmentation_mask=segmentation_mask,
                    feature_mask=feature_mask,
                    masked_feature=masked_feature
                ).sample

                # Generates the negative noise
                noise_pred_uncond: torch.Tensor = self._unet(
                    sample=latents,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states_nocond,
                    segmentation_mask=segmentation_mask,
                    feature_mask=feature_mask,
                    masked_feature=masked_feature
                ).sample

                # Subtract the noises in order to get the residual noise
                residual_noise: torch.Tensor = noise_pred_uncond + guidance_scale * \
                                               (noise_pred - noise_pred_uncond)

                latents: torch.Tensor = self._scheduler.step(
                    residual_noise, t, latents, generator=generator
                ).prev_sample

        # Decodes the latents in order to get the generated images
        latents: torch.Tensor = 1 / self._vae.config.scaling_factor * latents
        image_as_tensor: torch.Tensor = self._vae.decode(latents.float(), return_dict=False)[0]

        return utils.tensor_to_image(tensor=image_as_tensor)
