"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Source: https://github.com/TZW1998/Taming-Stable-Diffusion-with-Human-Ranking-Feedback
Purpose:
"""

# IMPORT: utils
from typing import *

# IMPORT: data processing
import numpy as np
import torch


class RankingFeedback:
    """ Represents the object allowing to improve images using ranking feedback. """
    def __init__(self):
        """
        Initializes the object allowing to improve images using ranking feedback.

        Parameters
        ----------
            ...: ...
                ...
        """
        # ----- Attributes ----- #
        self.best_latent, self.latents = None, None
        self.best_image, self.images = None, None

        # Number of images to generate and shape of the images
        self._n, self._shape = None, None

        # Smoothing parameters to apply when creating new latents
        self._smoothing_factor = 0.1

        self._step = "grad"
        self._search_direction = None

    def gen_new_latents(self) -> torch.Tensor:
        # Gradient estimation
        if self._step == "grad":
            # Creates the variations needed to create the new latents
            latent_variations = torch.rand((self._n, *self._shape[1:])) * self._smoothing_factor

        # Search direction
        else:
            # Computes the search directions
            latent_variations = torch.cat(
                tensors=[self._search_direction * (0.5**i) for i in range(self._n - 2)],
                dim=0
            )

        # Combines the best latent with the variations to creates the new latents
        self.latents = latent_variations + self.best_latent
        return self.latents

    def ranking_feedback(self, ranking: str):
        # Grad estimation
        if self._step == "grad":
            # Parse the ranking
            ranking = self._parse_ranking(ranking)

            # Computes
            update_direction = torch.zeros_like(self.best_latent)
            for image_idx, rank in ranking.items():
                latent_variation = self.latents[image_idx] * self.best_latent
                if rank >= 0:
                    update_direction += (self._n - 2*rank) * latent_variation
                else:
                    # TODO (not number of images but number of ranked images)
                    update_direction += -len(ranking) * latent_variation

            # Sets
            self._step = "search"

        # Search direction
        else:
            pass
            self._step = "grad"

    def _parse_ranking(self, ranking: str) -> Dict[int, int]:
        ranking = [int(e) for e in ranking.split("-")]

        ranking_as_dict = dict()
        for image_idx in range(1, self._n+1):
            if image_idx in ranking:
                ranking_as_dict[image_idx] = ranking.index(image_idx)
            else:
                ranking_as_dict[image_idx] = -1

        return ranking_as_dict
