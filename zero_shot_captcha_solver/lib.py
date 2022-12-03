import os
from typing import Optional

import numpy as np
import sentence_transformers
from PIL import Image


def cluster_into_two(nums_1d: np.ndarray) -> np.ndarray:
    """Takes a 1d array of numbers and clusters them into two groups.

    Args:
        nums_1d (np.ndarray): 1d array of numbers.

    Returns:
        np.ndarray: 1d array of False and True where True corresponds to the group of images that contain the object.
    """
    nums_1d = nums_1d.flatten()
    nums_sorted, nums_sorted_args = np.sort(nums_1d), np.argsort(nums_1d)
    diffs = np.diff(nums_sorted)
    return nums_1d > nums_1d[nums_sorted_args[np.argmax(diffs)]]


def split_img(raw_img: np.ndarray, num_h: int, num_w: int) -> np.ndarray:
    """Splits an image equally into a grid of images.

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


def solve_captcha(
    captcha_img: np.ndarray, object: str, model: Optional[sentence_transformers.SentenceTransformer] = None
) -> np.ndarray:
    """Solves a captcha by comparing all the images in the grid to the text.

    Assumes:
    - that there is a grid of 3x3 images
    - that all individual images are of the same size
    - that there is at least one image that contains the object

    Args:
        captcha_img (np.ndarray): Captcha image of 3x3 grid, shape (H, W, C).
        object (str): Object to find in the captcha.
        model (Optional[sentence_transformers.SentenceTransformer], optional): Sentence transformer model that works
            both with images and texts. Defaults to None.

    Returns:
        np.ndarray: 2d array of True and False corresponding to the grid where True are the images that contain the
            object.
    """
    imgs = split_img(captcha_img, num_h=3, num_w=3)
    imgs_flattened = imgs.reshape(-1, *imgs.shape[-3:])

    if model is None:
        model = sentence_transformers.SentenceTransformer("clip-ViT-B-32")

    img_emb_arr = model.encode([Image.fromarray(img) for img in imgs_flattened])
    text_emb = model.encode([object])
    cos_scores_flattened = np.array(sentence_transformers.util.cos_sim(img_emb_arr, text_emb))

    return cluster_into_two(cos_scores_flattened).reshape(3, 3)


def solve_captcha_from_path(
    path: str, object: Optional[str] = None, model: Optional[sentence_transformers.SentenceTransformer] = None
) -> np.ndarray:
    """Solves a captcha by comparing all the images in the grid to the text.

    Assumes:
    - that there is a grid of 3x3 images
    - that all individual images are of the same size
    - that there is at least one image that contains the object
    - that the filename of the path is the object to find

    Args:
        path (str): Path to the captcha image
        object (Optional[str], optional): Object to find in the captcha. Defaults to None.
            if None, the object is inferred from the filename of the path.
        model (Optional[sentence_transformers.SentenceTransformer], optional): Sentence transformer model that works
            both with images and texts. Defaults to None.

    Returns:
        np.ndarray: 2d array of 0s and 1s corresponding to the grid where 1s are the images that contain the object
    """
    if object is None:
        object = path.split(os.sep)[-1].split(".")[0]
    img = Image.open(path)
    return solve_captcha(np.array(img), object, model)


def load_clip() -> sentence_transformers.SentenceTransformer:
    """Loads the CLIP model.

    Returns:
        sentence_transformers.SentenceTransformer: CLIP model
    """
    return sentence_transformers.SentenceTransformer("clip-ViT-B-32")
