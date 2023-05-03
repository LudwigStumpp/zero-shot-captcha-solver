"""
Collection of functions to compute the similarity between texts and images.
Used to find images inside the captcha that contain the object of interest.
"""

import numpy as np
import transformers

MODEL_NAME = "openai/clip-vit-base-patch32"


def load_clip() -> transformers.CLIPModel:
    """Loads the pretrained CLIP model.

    Returns:
        CLIPModel: CLIP model
    """
    return transformers.CLIPModel.from_pretrained(MODEL_NAME)


def compute_texts_images_similarity(
    texts: list[str], images: list[np.ndarray], clip_model: transformers.CLIPModel | None = None
) -> np.ndarray:
    """Computes the similarity between texts and images using CLIP.

    Args:
        texts (List[str]): List of texts.
        images (List[np.ndarray]): List of images. Each image is a numpy array of shape (H, W, C).
        clip_model (transformers.CLIPModel | None, optional): Pretrained CLIP model. Defaults to None.
            If not specified, will load the default model.

    Returns:
        np.ndarray: Array of similarity scores of shape (text, image).
            For example similarity_scores[0, 1] is the similarity score between texts[0] and images[1].
    """
    if clip_model is None:
        clip_model = load_clip()
    processor = transformers.CLIPProcessor.from_pretrained(MODEL_NAME)
    inputs = processor(text=texts, images=images, return_tensors="pt")
    outputs = clip_model(**inputs)
    return outputs.logits_per_text.detach().cpu().numpy()  # shape (text, image)
