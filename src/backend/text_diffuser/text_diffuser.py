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

# IMPORT: data processing
import torch
from PIL import Image

# IMPORT: deep learning
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL
from diffusers.models.text_diffuser_unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.text_diffuser_scheduling_ddpm import DDPMScheduler

# IMPORT: project
import paths

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
        scale_factor = 2 ** (len(self._vae.config.block_out_channels) - 1)
        sigma = self._scheduler.init_noise_sigma

        # Creates the normalized random noise
        latents = torch.randn(size=(b, c, h // scale_factor, w // scale_factor),
                              generator=generator)
        return latents * sigma

    def _prepare_prompt(self, prompt: str, num_images: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the prompts.

        Parameters
        ----------
            prompt: str
                prompt to prepare
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
        negative_prompt: List[str] = [""] * num_images

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

    @staticmethod
    def _load_segmentation_model():
        """
        Loads the character segmentation model.

        Returns
        ----------
        character segmentation model
        """
        # Instantiates the model
        model = UNet(3, 96, True).cuda()
        model = torch.nn.DataParallel(model)

        # Loads the pretrained weights
        model.load_state_dict(
            state_dict=torch.load(
                os.path.join(paths.TEXT_DIFFUSER, "text_segmenter.pth")
            )
        )

        model.eval()
        return model

    def _prepare_text_diffuser_inputs(
            self,
            prompt: str,
            num_images: int
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
            guidance_scale: float
    ) -> torch.Tensor:
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
            torch.Tensor
                generated images
        """
        for t in tqdm(self._scheduler.timesteps):
            with torch.no_grad():
                # Generates the noise
                noise_pred = self._unet(
                    sample=latents,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states,
                    segmentation_mask=segmentation_mask,
                    feature_mask=feature_mask,
                    masked_feature=masked_feature
                ).sample

                # Generates the negative noise
                noise_pred_uncond = self._unet(
                    sample=latents,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states_nocond,
                    segmentation_mask=segmentation_mask,
                    feature_mask=feature_mask,
                    masked_feature=masked_feature
                ).sample

                # Subtract the noises in order to get the residual noise
                residual_noise = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
                latents = self._scheduler.step(residual_noise, t, latents).prev_sample

        # Decodes the latents in order to get the generated images
        latents = 1 / self._vae.config.scaling_factor * latents
        return self._vae.decode(latents.float(), return_dict=False)[0]
