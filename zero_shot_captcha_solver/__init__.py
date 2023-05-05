import numpy as np

from zero_shot_captcha_solver import captcha, clustering, text_image_similarity


def solve_captcha(
    captcha_img: captcha.Captcha, model: text_image_similarity.TextImageSimilarityModel | None = None
) -> np.ndarray:
    """Solves a captcha by comparing all the images in the grid to the text.

    Args:
        captcha (captcha.Captcha): Captcha object.
        model (text_image_similarity.TextImageSimilarityModel | None, optional): Pretrained text-image similarity
            model. Defaults to None.

    Returns:
        np.ndarray: 2d array of True and False corresponding to the grid where True are the images that contain the
            target_object.
    """
    if model is None:
        model = text_image_similarity.CLIPTextImageSimilarityModel()

    imgs = captcha_img.image_grid.grid_img_split  # shape (num_rows, num_cols, height, width, channel)
    imgs_flattened = imgs.reshape(-1, *imgs.shape[-3:])  # shape (image, height, width, channel)
    imgs_list = [img for img in imgs_flattened]

    similarity_scores = text_image_similarity.compute_texts_images_similarity(
        texts=[captcha_img.target_object], images=imgs_list, text_image_similarity_model=model
    )  # shape (1, image)
    similarity_scores_flattened = similarity_scores.flatten()  # shape (image,)

    return clustering.cluster_into_two(similarity_scores_flattened).reshape(
        *imgs.shape[:2]
    )  # shape (num_rows, num_cols)


CLIPTextImageSimilarityModel = text_image_similarity.CLIPTextImageSimilarityModel
CaptchaImage = captcha.Captcha

__all__ = ["solve_captcha", "CLIPTextImageSimilarityModel", "CaptchaImage"]
