__author__ = "Daeyoung Hong, Woohwan Jung"
__date__ = "2022.02.26."
"""
Not requiring config
"""
import json
import operator
import os
import pickle
import random
from functools import reduce
from math import ceil
from typing import Iterator

import numpy as np
import scipy as sp
from scipy.sparse import lil_matrix

from tqdm import tqdm


def load_json(path: str):
    """
    Args:
        path: the JSON file path to load.
    Returns:
        The load dict object.
    """
    print('Try to load: {}'.format(path))
    with open(path) as f:
        out = json.load(f)
    print('Loaded: {}'.format(path))
    return out


def dump_pickle(obj, path: str):
    """
    Args:
        obj: the object to save.
        path: the file path where `obj` is saved.
    """
    tmp_path = path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(obj, f)
    os.rename(tmp_path, path)
    print('Saved: {}'.format(path))


def load_pickle(path: str):
    """
    Args:
        path: the pickle file path to load.
    Returns:
        The loaded object.
    """
    try:
        with open(path, 'rb') as f:
            out = pickle.load(f)
        print('Loaded: {}'.format(path))
    except:
        print("Failed to load: {}".format(path))
        raise
    return out


def my_tqdm(*args, desc=None, total=None, **kwargs):
    """
    Customized tqdm function with preferred parameters.
    Args:
        desc: the string to display
        total: the total counts
    """
    return tqdm(*args, **kwargs, dynamic_ncols=True, ascii=True, desc=desc, total=total)


def set_seed(seed):
    """
    initialize the random number generator.
    Args:
        seed: the seed used to initialize the random number generator.
    """
    np.random.seed(seed=seed)
    random.seed(seed)
    sp.random.seed(seed)


def truncate(data, min_, max_):
    """
    Args:
        data: data to truncate
        min_: the minimum value used for truncation
        max_: the maximum value used for truncation
    Returns:
        The truncated data where each value lie between input `min_` and input `max_`
    """
    return np.maximum(np.minimum(max_, data), min_)


def split_arr(arr, n_splits=None, chunk_size=None):
    """
    Iterate divided arrays.
    Args:
        arr: the original data array
        n_splits: the number of splits
        chunk_size: the size of each chunk
    """
    if n_splits is not None:
        split_size = ceil(len(arr) / n_splits)
    elif chunk_size is not None:
        split_size = chunk_size
    else:
        raise Exception('n_splits=None, chunk_size=None')
    i = 0
    while i * split_size < len(arr):
        yield arr[i * split_size: (i + 1) * split_size]
        i += 1
    # return (arr[i * split_size: (i + 1) * split_size] for i in range(n))


def iter_idx(ranges) -> Iterator[tuple]:  # Include high idx
    """
    Iterate indices inside an index range (multi-dimensional).
    Args:
        ranges: the list of index ranges each of which is (lower bound, upper bound) for each dimension
    """
    for i in range(ranges[0][0], ranges[0][1] + 1):
        if len(ranges) == 1:
            yield i,
        else:
            for idx in iter_idx(ranges[1:]):
                yield (i,) + idx


def reduce_mul(x):
    """
    Args:
        x: the values to be multiplied
    Returns:
        Multiplied value for the values in the input `x`
    """
    return reduce(operator.mul, x, 1)


def is_valid_idx(idx, shape):
    """
    Args:
        idx: an index
        shape: a shape used for checking the input index `idx`
    Returns:
        Whether the input index `idx` is valid in terms of the shape `shape`
    """
    idx = np.array(idx)
    shape = np.array(shape)
    assert len(idx.shape) == 1 and len(idx) == len(shape)
    return bool(np.all(0 <= idx)) and bool(np.all(idx < shape))


class Cumu:
    @classmethod
    def range_to_indices(cls, range_):
        range_ = np.array(range_)
        yield range_[:, 1], 1.
        if range_[0, 0] >= 1 and range_[1, 0] >= 1:
            yield (range_[0, 0] - 1, range_[1, 0] - 1), 1.
        if range_[0, 0] >= 1:
            yield (range_[0, 0] - 1, range_[1, 1]), -1.
        if range_[1, 0] >= 1:
            yield (range_[0, 1], range_[1, 0] - 1), -1.

    @classmethod
    def range_to_flattened_indices(cls, range_, shape):
        for i, sign in cls.range_to_indices(range_):
            yield np.ravel_multi_index(i, shape), sign

    @classmethod
    def build_mat_to_convert_to_ori(cls, grid_shape):
        """
        :returns a sparse matrix which converts to original frequency vector
        when multiplied to the cumulative frequency vector
        """
        n = reduce_mul(grid_shape)
        mat = lil_matrix((n, n), dtype=np.float)
        for ref_idx in np.ndindex(*grid_shape):
            for idx, sign in Cumu.range_to_flattened_indices(((ref_idx[0], ref_idx[0]), (ref_idx[1], ref_idx[1])),
                                                             grid_shape):
                mat[np.ravel_multi_index(ref_idx, grid_shape), idx] += sign
        return mat.tocsc()

    @classmethod
    def to_cumu(cls, freq: np.ndarray):
        cumu = np.empty(freq.shape)
        cumu[0, 0] = freq[0, 0]
        for i in range(1, cumu.shape[0]):
            cumu[i, 0] = freq[i, 0] + cumu[i - 1, 0]
        for j in range(1, cumu.shape[1]):
            cumu[0, j] = freq[0, j] + cumu[0, j - 1]
        for i in range(1, cumu.shape[0]):
            for j in range(1, cumu.shape[1]):
                cumu[i, j] = freq[i, j] + cumu[i, j - 1] + cumu[i - 1, j] - cumu[i - 1, j - 1]
        return cumu


def calc_mse(original_data: np.ndarray, perturbed_data: np.ndarray):
    """
    Args:
        original_data: the original data
        perturbed_data: the perturbed data
    Returns:
        The MSE between `original_data` and `perturbed_data`
    """
    return ((original_data - perturbed_data) ** 2).mean()
