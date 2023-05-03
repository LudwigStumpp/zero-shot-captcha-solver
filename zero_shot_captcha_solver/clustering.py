"""
Collection of functions for clustering.
Used to cluster the images into two groups, one group containing the object of interest and the other group not.
"""

import numpy as np


def cluster_into_two(nums_1d: np.ndarray) -> np.ndarray:
    """Takes a 1d array of numbers and clusters them into two groups.

    Works by finding the largest gap between two consecutive numbers and then splitting the numbers into two groups.
    Top group is assigned True and bottom group is assigned False.

    Args:
        nums_1d (np.ndarray): 1d array of numbers.

    Returns:
        np.ndarray: 1d array of False and True where True corresponds to the top group.
    """
    nums_1d = nums_1d.flatten()
    nums_sorted, nums_sorted_args = np.sort(nums_1d), np.argsort(nums_1d)
    diffs = np.diff(nums_sorted)
    return nums_1d > nums_1d[nums_sorted_args[np.argmax(diffs)]]
