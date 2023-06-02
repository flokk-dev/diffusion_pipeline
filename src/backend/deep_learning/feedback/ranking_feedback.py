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
from PIL import Image
import torch


class RankingFeedback:
    """
    Represents the object allowing to improve images using ranking feedback.

    Attributes
    ----------
        _n: int
            ...
        _lr: float
            ...
        _smoothing_factor: float
            ...
        step: str
            ...

    """

    @torch.no_grad()
    def __init__(
        self,
        latent: torch.Tensor,
        image: Image.Image,
        num_latents: int = 4,
        lr: float = 2.0,
        smoothing_factor: float = 0.1
    ):
        """
        Initializes the object allowing to improve images using ranking feedback.

        Parameters
        ----------
            latent: torch.Tensor
                latent from which to start the procedure
            image: Image.Image
                image generated from the latent
            num_latents: int
                number of latents to generate at each iteration
            smoothing_factor: float
                factor to apply to the latent variations
            lr: float
                step to use during the line search
        """
        # ----- Attributes ----- #
        # Parameters
        self._n: int = num_latents  # Number of latents to generate
        self._lr: float = lr  # Learning rate of the line search
        self._smoothing_factor: float = smoothing_factor  # Strength of the latent variations

        # Current step of the ranking feedback
        self.step: str = "gradient_estimation"

        # Latents
        self._best_latent: torch.Tensor = latent.unsqueeze(0)  # Best latent of the iteration
        self._best_image: Image.Image = image  # Best image of the iteration

        self._gradient_best_latent: torch.Tensor = None  # Best latent after the gradient estimation
        self._gradient_best_image: Image.Image = None  # Best image after the gradient estimation

        self._shape: Tuple[int] = self._best_latent.shape  # Shape of the latents

        # Variations of the best latent updated at each step
        self._latents: torch.Tensor = None

        # ...
        self._search_direction: torch.Tensor = torch.zeros_like(self._best_latent)

    @torch.no_grad()
    def gen_new_latents(self) -> torch.Tensor:
        """
        Generates new latents (depends on the current step).

        Returns
        ----------
            torch.Tensor
                generated latents
        """
        # Creates the variations to add to the current best latent in order to generate new latents
        latent_variations: torch.Tensor = None

        # Gradient estimation
        if self.step == "gradient_estimation":
            # Creates the variations needed to create the new latents
            latent_variations = torch.randn((self._n, *self._shape[1:])) * self._smoothing_factor

        # Line search
        elif self.step == "line_search":
            # Computes the search directions
            latent_variations = torch.cat(
                tensors=[
                    self._search_direction.clone() * (self._lr * (0.5 ** i))
                    for i in range(self._n - 2)
                ],
                dim=0
            )

        # Combines the best latent with the variations to creates the new latents
        self._latents = latent_variations + self._best_latent
        return self._latents

    @torch.no_grad()
    def ranking_feedback(self, ranking: str, generated_images: List[Image.Image]):
        """
        Computes the ranking feedback (depends on the current step).

        Parameters
        ----------
            ranking: str
                ranking from the user
            generated_images: List[Image.Image]
                images generated during the current step
        """
        # Gradient estimation
        if self.step == "gradient_estimation":
            # Parse the ranking
            ranked, unranked = self._parse_ranking(ranking)
            k = len(ranked)

            # Computes the update of the search direction
            update_direction: torch.Tensor = torch.zeros_like(self._best_latent)

            # For the ranked images
            for rank, image_idx in ranked.items():
                update_direction += (self._n - (2 * rank)) * \
                                    (self._latents[image_idx] - self._best_latent)

            # For the unranked images
            for image_idx in unranked:
                update_direction += -k * (self._latents[image_idx] - self._best_latent)

            # Updates the search direction
            update_direction /= (k * (k - 1)) / 2 + k * (self._n - k)
            self._search_direction += update_direction

            # Stores the best latent
            self._gradient_best_latent = self._latents[ranked[0]].unsqueeze(0)
            self._gradient_best_image = generated_images[ranked[0]]

            # Step
            self.step = "line_search"

        # Line search
        elif self.step == "line_search":
            # Parse the ranking
            best_latent_idx = int(ranking) - 1

            # If the index of the best latent is not the one of the current best image
            if best_latent_idx != len(self._latents) - 1:
                self._best_latent = self._latents[best_latent_idx].unsqueeze(0)
                self._best_image = generated_images[best_latent_idx]

                self._search_direction = torch.zeros_like(self._best_latent)

            # Step
            self.step = "gradient_estimation"

    @torch.no_grad()
    def adjust_generated_images(self, generated_images: List[Image.Image]) -> List[Image.Image]:
        """
        Modifies the latents and the generated images before the line search procedure.

        Parameters
        ----------
            generated_images: List[Image.Image]
                images generated during current step

        Returns
        ----------
            List[Image.Image]
                modified generated images
        """
        if self.step == "gradient_estimation":
            return generated_images

        # Adds the best latents to the latents of the current step
        self._latents = torch.cat(
            [self._latents, self._gradient_best_latent, self._best_latent],
            dim=0
        )

        # Adds the best images to the generated images
        return generated_images + [self._gradient_best_image] + [self._best_image]

    def _parse_ranking(self, ranking: str) -> Tuple[Dict[int, int], List[int]]:
        """
        Computes the ranking feedback (depends on the current step).

        Parameters
        ----------
            ranking: str
                ranking from the user

        Returns
        ----------
            Dict[int, int]
                ranked latents
            List[int]
                unranked latents
        """
        # Transform the string into a list
        ranking: List[int] = [int(e) for e in ranking.split("-")]

        ranked: Dict[int, int] = dict()  # Dict of ranked images
        unranked: List[int] = list()  # List of unranked images

        # For each image index between 1 and the number of images
        for image_idx in range(self._n):
            # If the image has been ranked
            if image_idx + 1 in ranking:
                # Adds the image to the ranked images list
                ranked[ranking.index(image_idx + 1)] = image_idx

            # If the image has not been ranked
            else:
                # Adds the image to the unranked images list
                unranked.append(image_idx)

        return ranked, unranked
