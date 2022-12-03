import os
from typing import Optional

import numpy as np
import sentence_transformers
from PIL import Image


def cluster_into_two(nums: np.ndarray) -> np.ndarray:
    nums = nums.flatten()
    nums_sorted, nums_sorted_args = np.sort(nums), np.argsort(nums)
    diffs = np.diff(nums_sorted)
    return nums > nums[nums_sorted_args[np.argmax(diffs)]]

    # km = ckwrap.ckmeans(nums, 2)
    # return km.labels


def split_img(raw_img, num_h, num_w) -> np.ndarray:
    # shape (num_h, num_w, h, w, c)
    return np.array(
        [
            np.split(img, indices_or_sections=num_w, axis=1)
            for img in np.split(raw_img, indices_or_sections=num_h, axis=0)
        ]
    )


def solve_captcha(captcha_img: np.ndarray, text: str, model=None) -> np.ndarray:
    imgs = split_img(captcha_img, num_h=3, num_w=3)
    imgs_flattened = imgs.reshape(-1, *imgs.shape[-3:])

    if model is None:
        model = sentence_transformers.SentenceTransformer("clip-ViT-B-32")

    img_emb_arr = model.encode([Image.fromarray(img) for img in imgs_flattened])
    text_emb = model.encode([text])
    cos_scores_flattened = np.array(sentence_transformers.util.cos_sim(img_emb_arr, text_emb))

    return cluster_into_two(cos_scores_flattened).reshape(3, 3)


def solve_captcha_from_path(path: str, text: Optional[str] = None, model=None) -> np.ndarray:
    if text is None:
        text = path.split(os.sep)[-1].split(".")[0]
    img = Image.open(path)
    return solve_captcha(np.array(img), text, model)


def load_clip():
    return sentence_transformers.SentenceTransformer("clip-ViT-B-32")
