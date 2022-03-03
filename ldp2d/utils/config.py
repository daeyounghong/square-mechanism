__author__ = "Daeyoung Hong, Woohwan Jung"
__date__ = "2022.02.26."

"""
Requires config information
"""

import os
import pickle

import numpy as np

from ldp2d.config.config import local_data_dir, re_sanity_bound
from ldp2d.config.keys.result import re


class TmpSaveLoader:
    """
    with TmpSaveLoader(name, obj) as loader:
        x = loader.load()
    """

    def __init__(self, name, obj):
        pid = os.getpid()
        self._path = os.path.join(local_data_dir, f'{name}_{pid}.pkl')
        path = self._path
        tmp_path = path + '.tmp'
        assert not os.path.isfile(path) and not os.path.isfile(tmp_path)
        with open(tmp_path, 'wb') as f:
            pickle.dump(obj, f)
        os.rename(tmp_path, path)
        print('Saved: {}'.format(path))

    def load(self):
        path = self._path
        with open(path, 'rb') as f:
            out = pickle.load(f)
        print('Loaded: {}'.format(path))
        return out

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        os.remove(self._path)
        print('Removed: {}'.format(self._path))


def smoothed_relative_err(true, estimated, total_cnt=1):
    """
    Args:
        true: the true value
        estimated: the estimated value
        total_cnt: the total count
    Returns:
        The relative error of input `estimated` with sanity bound that is computed by `re_sanity_bound * total_cnt`.
    """
    return np.abs(estimated - true) / np.maximum(true, re_sanity_bound * total_cnt)


error_f = {re: smoothed_relative_err}


def get_large_i(small_i: tuple, large_shape, small_shape):
    """
    Iterates cell indices of a fine-grained grid corresponds to `small_i`.
    Args:
        small_i: the index of cell in coarse-grained grid
        large_shape: the shape of fine-grained grid
        small_shape: the shape of coarse-grained grid
    """
    dim = len(large_shape)
    multiples = np.empty(dim, dtype=int)
    out = tuple()
    for i in range(dim):
        assert large_shape[i] % small_shape[i] == 0
        multiples[i] = large_shape[i] // small_shape[i]
        out += (small_i[i] * multiples[i],)
    for offset_0 in range(multiples[0]):
        for offset_1 in range(multiples[1]):
            yield out[0] + offset_0, out[1] + offset_1


def convert_uni_histo(histo: np.ndarray, tgt_shape: tuple):
    """
    Args:
        histo: the original histogram
        tgt_shape: the target shape for transforming `histo`
    Returns:
        The transformed histogram with the shape `tgt_shape` from the original histogram `histo`
            by assuming each bin of the histogram is uniformly distributed.
    """
    # Converts uniform histogram
    dim = len(histo.shape)
    if histo.shape == tgt_shape:
        return histo
    # make larger
    out_hist = np.zeros(tgt_shape)
    if tgt_shape[0] > histo.shape[0]:
        divisor = 1
        for i in range(dim):
            assert tgt_shape[i] % histo.shape[i] == 0
            divisor *= tgt_shape[i] // histo.shape[i]
        for i, v in np.ndenumerate(histo):
            for large_i in get_large_i(i, tgt_shape, histo.shape):
                out_hist[large_i] = histo[i] / divisor
    # make smaller
    else:
        multiples = np.empty(dim, dtype=int)
        for i in range(dim):
            assert histo.shape[i] % tgt_shape[i] == 0
            multiples[i] = histo.shape[i] // tgt_shape[i]
        for i, v in np.ndenumerate(histo):
            tgt_i = np.empty(dim, dtype=int)
            for d in range(dim):
                tgt_i[d] = i[d] // multiples[d]
            out_hist[tuple(tgt_i)] += histo[i]
    return out_hist


def freq_est_err(ori_hist: np.ndarray, est_hist: np.ndarray, total_cnt: int = 1, truncate=True):
    """
    Args:
        ori_hist: the original histogram
        est_hist: the estimated histogram
        total_cnt: the total frequency
        truncate: whether truncate the output to make it non-negative.
    Returns:
        A dictionary that consists of key-value pairs of
            (the name of error measure, the array of errors of the bins in histogram `est_hist`)
    """
    if not (est_hist.sum() <= 1.5 * total_cnt, est_hist.sum()):
        print('est_hist.sum() = {}'.format(est_hist.sum()))
    assert ori_hist.sum() <= 1.13 * total_cnt, ori_hist.sum()
    if truncate:
        est_hist = np.minimum(total_cnt, np.maximum(0., est_hist))
    if est_hist.shape != ori_hist.shape:
        est_hist = convert_uni_histo(est_hist, ori_hist.shape)
    out = {}
    for err_name, err_f in error_f.items():
        if err_name in {re}:
            out[err_name] = err_f(ori_hist, est_hist, total_cnt)
        else:
            out[err_name] = err_f(ori_hist, est_hist)
    return out
