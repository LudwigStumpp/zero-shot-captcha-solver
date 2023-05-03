import os

import numpy as np
import transformers
from PIL import Image

from zero_shot_captcha_solver import (
    captcha_preparation,
    clustering,
    text_image_similarity,
)


def solve_captcha(
    captcha_img: np.ndarray, target_object: str, model: transformers.CLIPModel | None = None
) -> np.ndarray:
    """Solves a captcha by comparing all the images in the grid to the text.

    Assumes:
    - that there is a grid of 3x3 images
    - that all individual images are of the same size
    - that there is at least one image that contains the target_object

    Args:
        captcha_img (np.ndarray): Captcha image of 3x3 grid, shape (H, W, C).
        target_object (str): Object to find in the captcha.
        model (transformers.CLIPModel | None, optional): Pretrained CLIP model. Defaults to None.

    Returns:
        np.ndarray: 2d array of True and False corresponding to the grid where True are the images that contain the
            target_object.
    """
    imgs = captcha_preparation.split_img(captcha_img, num_h=3, num_w=3)
    imgs_flattened = imgs.reshape(-1, *imgs.shape[-3:])  # shape (image, height, width, channel)
    imgs_list = [Image.fromarray(img) for img in imgs_flattened]

    similarity_scores = text_image_similarity.compute_texts_images_similarity(
        texts=[target_object], images=imgs_list, clip_model=model
    )  # shape (1, image)
    similarity_scores_flattened = similarity_scores.flatten()  # shape (image,)

    return clustering.cluster_into_two(similarity_scores_flattened).reshape(3, 3)


def solve_captcha_from_path(
    path: str, target_object: str | None = None, model: transformers.CLIPModel | None = None
) -> np.ndarray:
    """Solves a captcha by comparing all the images in the grid to the text.

    Assumes:
    - that there is a grid of 3x3 images
    - that all individual images are of the same size
    - that there is at least one image that contains the target object
    - that the filename of the path is the object to find

    Args:
        path (str): Path to the captcha image
        target_object (str | None, optional): Object to find in the captcha. Defaults to None.
            if None, the object is inferred from the filename of the path.
        model (transformers.CLIPModel | None, optional): Pretrained CLIP model. Defaults to None.

    Returns:
        np.ndarray: 2d array of 0s and 1s corresponding to the grid where 1s are the images that contain the object
    """
    if target_object is None:
        target_object = path.split(os.sep)[-1].split(".")[0]
    img = Image.open(path)
    return solve_captcha(np.array(img), target_object, model)


load_clip = text_image_similarity.load_clip


__all__ = ["solve_captcha_from_path", "solve_captcha", "load_clip"]
