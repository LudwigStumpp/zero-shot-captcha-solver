"""
Collection of functions to prepare the captcha images.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ImageGrid:
    """Class to represent a grid of images.

    Attributes:
        grid_img (np.ndarray): Grid image of shape (H, W, C).
        num_rows (int): Number of rows in the grid.
        num_cols (int): Number of columns in the grid.
        grid_img_split (np.ndarray): Grid image split into a 5d array of shape (num_rows, num_cols, H, W, C).
    """

    grid_img: np.ndarray
    num_rows: int
    num_cols: int

    @property
    def grid_img_split(self):
        return split_img(
            raw_img=self.grid_img,
            num_rows=self.num_rows,
            num_cols=self.num_cols,
        )


@dataclass
class Captcha:
    """Class to represent a captcha.

    Attributes:
        target_object (str): The target object in the captcha.
        image_grid (ImageGrid): The grid of images in the captcha.
    """

    target_object: str
    image_grid: ImageGrid

    @classmethod
    def from_image_grid(cls, grid_img: np.ndarray, target_object: str, num_rows: int, num_cols: int):
        """Creates a Captcha object from a grid image.

        Args:
            grid_img (np.ndarray): Grid image of shape (H, W, C).
            target_object (str): The target object in the captcha.
            num_rows (int): Number of rows in the grid.
            num_cols (int): Number of columns in the grid.
        """
        return cls(
            target_object=target_object,
            image_grid=ImageGrid(
                grid_img=grid_img,
                num_rows=num_rows,
                num_cols=num_cols,
            ),
        )


def split_img(raw_img: np.ndarray, num_rows: int, num_cols: int) -> np.ndarray:
    """Splits an image equally into a grid of images.

    For example if we have an image which contains 6 images in a 2x3 grid, we can call this function as follows:
    >>> split_img(img, num_rows=2, num_cols=3)
    to get an array of shape (2, 3, H, W, C), so to access the image in the first row and second column we can do:
    >>> split_img(img, num_rows=2, num_cols=3)[0, 1]

    Args:
        raw_img (np.ndarray): Image to split, shape (H, W, C).
        num_rows (int): Number of splits along the heights dimension.
        num_cols (int): Number of splits along the widths dimension.

    Returns:
        np.ndarray: Array of images of shape (num_rows, num_cols, H, W, C).
    """
    return np.array(
        [
            np.split(img, indices_or_sections=num_cols, axis=1)
            for img in np.split(raw_img, indices_or_sections=num_rows, axis=0)
        ]
    )
