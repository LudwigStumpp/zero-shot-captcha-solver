"""
Collection of functions to compute the similarity between texts and images.
Used to find images inside the captcha that contain the object of interest.
"""

from typing import Protocol

import numpy as np
import transformers


class CLIPTextImageSimilarityModel:
    """Pretrained CLIP model to compute the similarity between texts and images.

    Args:
        model_name (str, optional): Pretrained model name. Defaults to "openai/clip-vit-base-patch32".

    Attributes:
        model_name (str): Pretrained model name.
        model (transformers.CLIPModel): CLIP model.
        processor (transformers.CLIPProcessor): CLIP processor.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self.model = self.load_clip(model_name)
        self.processor = transformers.CLIPProcessor.from_pretrained(model_name)

    def __call__(self, texts: list[str], images: list[np.ndarray]) -> np.ndarray:
        inputs = self.processor(text=texts, images=images, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.logits_per_text.detach().cpu().numpy()

    @staticmethod
    def load_clip(model_name: str) -> transformers.CLIPModel:
        """Loads the pretrained CLIP model.

        Returns:
            CLIPModel: CLIP model
        """
        return transformers.CLIPModel.from_pretrained(model_name)


class TextImageSimilarityModel(Protocol):
    def __call__(self, texts: list[str], images: list[np.ndarray]) -> np.ndarray:
        ...


def compute_texts_images_similarity(
    texts: list[str], images: list[np.ndarray], text_image_similarity_model: TextImageSimilarityModel
) -> np.ndarray:
    """Computes the similarity between texts and images using a text-image similarity model (e.g. CLIP).

    Args:
        texts (List[str]): List of texts.
        images (List[np.ndarray]): List of images. Each image is a numpy array of shape (H, W, C).
        text_image_similarity_model (TextImageSimilarityModel): Model that computes the similarity between texts and
            images.

    Returns:
        np.ndarray: Array of similarity scores of shape (text, image).
            For example similarity_scores[0, 1] is the similarity score between texts[0] and images[1].
    """
    return text_image_similarity_model(texts, images)
