"""
Creator: Flokk
Date: 19/05/2023
Version: 1.0

Purpose:
"""

# IMPORT: utils
from typing import *


class Component:
    """ Represents a component. """
    def __init__(self, parent: Any):
        """
        Initializes a component.

        Parameters
        ----------
            parent: Any
                parent of the component
        """
        # ----- Attributes ----- #
        self.parent: Any = parent

    def retrieve_info(self) -> List[Any]:
        """
        Runs the image processing into the image.

        Returns
        ----------
            List[Any]
                info within the component

        Raises
        ----------
            NotImplementedError
                function isn't implemented yet
        """
        raise NotImplementedError()
