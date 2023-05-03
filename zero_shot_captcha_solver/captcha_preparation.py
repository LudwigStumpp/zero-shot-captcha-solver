"""
Collection of functions to prepare the captcha images.
"""

import numpy as np


def split_img(raw_img: np.ndarray, num_h: int, num_w: int) -> np.ndarray:
    """Splits an image equally into a grid of images.

    For example if we have an image which contains 6 images in a 2x3 grid, we can call this function as follows:
    >>> split_img(img, num_h=2, num_w=3)
    to get an array of shape (2, 3, H, W, C), so to access the image in the first row and second column we can do:
    >>> split_img(img, num_h=2, num_w=3)[0, 1]

    Args:
        raw_img (np.ndarray): Image to split, shape (H, W, C).
        num_h (int): Number of horizontal splits.
        num_w (int): Number of vertical splits.

    Returns:
        np.ndarray: Array of images of shape (num_h, num_w, H, W, C).
    """
    return np.array(
        [
            np.split(img, indices_or_sections=num_w, axis=1)
            for img in np.split(raw_img, indices_or_sections=num_h, axis=0)
        ]
    )
