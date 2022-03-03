__author__ = "Daeyoung Hong, Woohwan Jung"
__date__ = "2022.02.26."

import gc
import os
from copy import deepcopy
from typing import Union

import numpy as np

from classes import Range, Range1D
from ldp2d.config.config import dim, gowalla_path, processed_data_path
from ldp2d.config.paths import DataPaths
from ldp2d.utils.others import load_pickle, dump_pickle, my_tqdm


def get_domain(arr: np.ndarray) -> Range:
    """
    :param arr: a data array with shape (data size, dimensionality)
    """
    low = arr.min(axis=0)
    high = arr.max(axis=0)
    return Range([Range1D(low[d], high[d]) for d in range(arr.shape[1])])


class OriginalData:
    """
    The member function `load` must be called before using the data.
    Attributes:
        loaded: whether the data is loaded in memory.
        arr: the data array parsed from the original dataset file.
        original_domain: the original domain of the data
        domain: the domain with a very small margin added from the original domain
            in order to deal with the points on the border of the domain.
        dim: the dimensionality of the dataset.
    """
    name: str
    arr: Union[np.ndarray, None]
    original_domain: Range
    domain: Range
    dim: int

    def __init__(self):
        self.loaded = False

    @classmethod
    def save_arr(cls, path: DataPaths, arr: np.ndarray):
        """
        Save the preprocessed data.
        Args:
            path: `DataPaths` object that contains the path to save the preprocessed data for the target dataset
            arr: the array of the data parsed from original dataset files
        """
        os.makedirs(path.dir, exist_ok=True)
        dump_pickle(arr, path.data)


class GowallaData(OriginalData):
    name = 'Gowalla'  # the name of the dataset
    file_path = gowalla_path  # the file path of the downloaded data file

    @classmethod
    def filter_errors(cls, arr):
        """
        :param arr: the original location data
        :return the filtered location data which has valid longitude and latitude
        """
        is_valid = np.all((-180.0 <= arr[:, 0], arr[:, 0] <= 180.0, -90.0 <= arr[:, 1], arr[:, 1] < 90.0), axis=0)
        print('valid tuples = {} / {}'.format(is_valid.sum(), len(is_valid)))
        arr = np.compress(is_valid, arr, axis=0)
        return arr

    def get_arr(self):
        """
        Returns:
            The data array parsed from the original dataset file.
        """
        file_path = self.file_path
        arr = []
        with open(file_path) as f:
            lines = f.readlines()
            for line in my_tqdm(lines, desc="parsing"):
                line = line.strip().split()

                # 'reversed' is used to exchange X and Y coordinates
                loc = np.array([float(word) for word in reversed(line[2:2 + dim])])
                if loc.shape != (2,):
                    continue

                arr.append(loc)

        arr = np.array(arr)  # (the number of points, the data dimension)
        return arr

    def preprocess(self):
        """
        The data is parsed from the original dataset file and saved the parsed data in the path of `self.get_paths()`.
        """
        data = self.get_arr()
        print('Array shape: {}'.format(data.shape))
        print(f'Parsed: {len(data)} tuples')
        self.save_arr(self.get_paths(), data)

    def short_name(self):
        """
        Returns:
            The short name of the dataset.
        """
        return self.name

    def long_name(self) -> str:
        """
        Returns:
            The long name of the dataset.
        """
        return '{}'.format(self.name)

    def load_the_original(self):
        """
        Parse the data from the original dataset file, and save the parsed data in `self.arr`.
        """
        arr = load_pickle(self.get_paths().data)
        self.arr = arr
        self.arr = self.filter_errors(self.arr)
        self.arr = self.arr.astype(float)

    def get_paths(self) -> DataPaths:
        """
        Returns:
            `DataPaths` object contains path where the preprocessed data will be saved.
        """
        return DataPaths(os.path.join(processed_data_path(), self.long_name()))

    def load(self):
        """
        Load the data in memory by reading the preprocessed file.
        In addition, the data is shifted so that the center of the domain becomes the origin (0, 0).
        The following class attributes are set to proper values: `arr`, `domain`, `original_domain`, `loaded`.
        """
        if not self.loaded:
            self.load_the_original()

            # Calculate the range of the data domain
            self.domain = get_domain(self.arr)
            print('Original domain: {}'.format(self.domain))
            self.original_domain = deepcopy(self.domain)
            self.domain.add_extra_margin()
            center = self.domain.center()
            self.domain.shift(-center)
            center = center.reshape(1, -1)
            self.arr -= center

            print('Processed domain: {}'.format(self.domain))
            print('Loaded: {}'.format(self.long_name()))
            self.loaded = True

    def free(self):
        """
        Free the data in the memory.
        """
        self.arr = None
        gc.collect()
        self.loaded = False

    def __enter__(self):
        """ Load the data in memory when entering the `with` statement """
        self.load()

    def __exit__(self, *args, **kwargs):
        """ Free the data in memory when excaping the `with` statement """
        self.free()
