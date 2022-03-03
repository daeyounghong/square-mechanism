__author__ = "Daeyoung Hong, Woohwan Jung"
__date__ = "2022.02.26."

import operator
from functools import reduce
from math import isclose, floor
from typing import List, Iterable, Tuple

import numpy as np

from ldp2d.config.config import domain_margin_ratio
from ldp2d.utils.others import truncate, iter_idx


class Domain1D:
    """
    Args:
        min_: the lower bound of a range
        max_: the upper bound of a range
    Attributes:
        _min: the lower bound of a range
        _max: the upper bound of a range
    """

    def __init__(self, min_, max_):
        max_ = float(max_)
        min_ = float(min_)
        self._min = min_
        self._max = max_ if max_ >= min_ else min_  # when len = 0

    def from_points(self, arr: np.ndarray):
        """
        Compute the range of data points and save the results in attributes of `self`.
        """
        self._min = arr.min()
        self._max = arr.max()
        self.add_extra_margin()

    def add_extra_margin(self):
        """
        Add a very small margin to the lower and upper bounds.
        """
        self._min -= self.len() * domain_margin_ratio
        self._max += self.len() * domain_margin_ratio

    def max(self):
        """
        Returns:
            The upper bound value
        """
        return self._max

    def min(self):
        """
        Returns:
            The lower bound value
        """
        return self._min

    def len(self):
        """
        Returns:
            The length of the range.
        """
        return self._max - self._min

    def cent(self):
        """
        Returns:
            The center value of the range `self`
        """
        return (self._max + self._min) / 2

    def __eq__(self, other):
        """
        Returns:
            Whether `self` is equal to the input object `other`
        """
        return (isclose(self.min(), other.min())) and (isclose(self.max(), other.max()))

    def __iter__(self):
        """
        Returns:
            An iterator iterating `self._min` and `self._max`
        """
        return iter((self.min(), self.max()))


class Range1D(Domain1D):
    def is_in(self, x):
        """
        Returns:
            Whether the input point `x` is inside the range `self`
        """
        return self._min <= x < self._max

    def __repr__(self):
        """
        Returns:
            The string representations of the `self`
        """
        return '({:.7f}, {:.7f})'.format(self.min(), self.max())

    def arr(self):
        """
        Returns:
            The array of (lower bound, upper bound)
        """
        return np.array((self.min(), self.max()))

    def shift(self, offset: float):
        """
        Shift the range with the input offset.
        Args:
            offset: the offset used to shift the range.
        """
        self._min += offset
        self._max += offset


class Range:
    """
    Args:
        range_ (List[Range1D]): the list of (Range1D for each dimension) or
            the list of tuples consisting of lower and upper bounds of each dimension
        arr: the data array which is user to compute range tightly containing the entire data array.
    Attributes:
        range: the list of (Range1D for each dimension)
    """
    range: List[Range1D]  # the list structure is require for getting the dimension

    def __init__(self, range_, arr=None):
        if range_ is not None:
            assert len(range_) >= 1
            if type(range_[0]) is Range1D:
                self.range = list(range_)
            else:
                self.range = [Range1D(r_1d[0], r_1d[1]) for r_1d in range_]
        elif arr is not None:
            self.from_points(arr)
        else:
            raise Exception('Invalid arguments')

    def from_points(self, arr: np.ndarray):
        """
        Compute the range of data points and save the results in attributes of `self`.
        """
        assert arr.shape[-1] == len(self)
        for dim_i, range_1d in enumerate(self):
            range_1d.from_points(arr[:, dim_i])

    def __getitem__(self, i) -> Range1D:
        """
        Returns:
            The `Range1D` object for the i-th dimension.
        """
        return self.range[i]  # required for iterating over the nested

    def __len__(self):
        """
        Returns:
            The dimensionality of the range
        """
        return len(self.range)

    def __repr__(self):
        """
        Returns:
            The string representation of `self`
        """
        return f'{self.range}'

    def __iter__(self):
        """
        Returns:
            The string representation of `self`
        """
        return iter(self.range)

    def __eq__(self, other):
        """
        Returns:
            Whether `self` is equal to the input `Range` object `other`
        """
        other: Range
        return all([my_r == o_r for my_r, o_r in zip(self, other)])

    def add_extra_margin(self):
        """
        Add a very small margin to the upper bounds of the range.
        """
        for range_1d in self:
            range_1d.add_extra_margin()
        self[0]._min = max(-180., self[0].min())
        self[1]._min = max(-90., self[1].min())

    def dim(self):
        """
        Returns:
            The dimensionality of the range
        """
        return len(self)

    def area(self):
        """
        Returns:
            The area of the range.
        """
        return reduce(operator.mul, (r.len() for r in self))

    def low(self) -> np.ndarray:
        """
        Returns:
            The lower bound value of each dimension.
        """
        return np.array([r_1d.min() for r_1d in self.range])

    def high(self) -> np.ndarray:
        """
        Returns:
            The upper bound value of each dimension.
        """
        return np.array([r_1d.max() for r_1d in self.range])

    def is_in(self, x):
        """
        Returns:
            Whether the input point `x` is inside the range `self`
        """
        return all(range_1d.is_in(x[d]) for d, range_1d in enumerate(self.range))

    def center(self) -> np.ndarray:
        """
        Returns:
            The center of the range.
        """
        return np.array([r_1d.cent() for r_1d in self.range])

    def vertices(self):
        """
        Returns:
            An iterator over the list of coordinates of the vertices
        """
        assert self.dim() == 2
        for x0 in self[0]:
            for x1 in self[1]:
                yield np.array((x0, x1))

    def arr(self):
        """
        Returns:
            The array where each element corresponds a tuple of (lower bound, upper bound) of each dimension
        """
        return np.array([range1d.arr() for range1d in self])

    def sample_pts(self, n) -> np.ndarray:
        """
        :param n: the number of samples
        """
        return np.random.uniform(self.low(), self.high(), (n, 2))

    def truncate(self, points: np.ndarray) -> np.ndarray:
        """
        Args:
            points: the input data points
        Returns:
            Truncated data points by the range `self`
        """
        original_shape = points.shape
        dim = self.dim()
        points = points.reshape(-1, dim)
        low = self.low().reshape(-1, dim)
        high = self.high().reshape(-1, dim)
        return truncate(points, low, high).reshape(original_shape)

    def shift(self, offsets):
        """
        Shift the range `self` by the input `offsets`, which is the offset for each dimension
        Args:
            offsets: the offset for each dimension
        """
        assert len(self) == len(offsets)
        for range1, offset in zip(self, offsets):
            range1: Range1D
            range1.shift(offset)


def split_idx_1d(range_1d: Range1D, b, x):
    """
    Args:
        range_1d: the entire range before dividing
        b: the number of divided ranges.
        x: a query point
    Returns:
        The index (zero-based) of divided range to which `x` is belongs to when `range_1d` is equally divided by `b`
    """
    offset = x - range_1d.min()
    width = range_1d.len() / b
    return int(floor(offset / width))


def split_idx(range_: Range, b, x_list) -> tuple:
    """
    Args:
        range_: the entire multi-dimensional range before dividing
        b: the number of divided ranges for each dimension.
        x_list: a query point (multi-dimensional)
    Returns:
        The index (zero-based) of divided range to which `x` is belongs to when `range` is equally divided with `b`
    """
    return tuple(split_idx_1d(range_1d, b_1d, x) for x, range_1d, b_1d in zip(x_list, range_, b))


def overlap_1d(a: Range1D, b: Range1D):
    """
    Args:
        a: a range
        b: a range
    Returns:
        The overlapping range of `a` and `b`.
    """
    return Range1D(max(a.min(), b.min()), min(a.max(), b.max()))


def overlap(a: Range, b: Range):
    """
    Args:
        a: a range
        b: a range
    Returns:
        The overlapping range of `a` and `b`.
    """
    return Range([overlap_1d(a_1d, b_1d) for a_1d, b_1d in zip(a, b)])


def answer_range_query_with_uniform_grid_histogram(domain, grid_regions, q_range, freq, truncate=True):
    """
    Args:
        domain: the entire domain.
        grid_regions: the array of `Range` objects
        q_range: the query range.
        freq: the frequency array for the grid.
        truncate: whether truncate the output value to make it non-negative.
    Returns:
        The frequency for a query range with using the histogram
            whose grid consists of `grid_regions` and frequencies consists of `freq`.
            Here, we assume that each bin is uniformly distributed.
    """
    start_idx = split_idx(domain, grid_regions.shape, q_range.low())
    end_idx = split_idx(domain, grid_regions.shape, q_range.high())
    end_idx = tuple(np.minimum(np.array(end_idx), np.array(grid_regions.shape) - 1))
    ans = 0
    for idx in iter_idx([[start_idx[0], end_idx[0]], [start_idx[1], end_idx[1]]]):
        try:
            ans += freq[idx] * overlap(q_range, grid_regions[idx]).area() / grid_regions[idx].area()
        except:
            print(idx)
            raise
    return max(ans, 0) if truncate else ans


class Grid:
    """
    Args:
        domain: the range of the entire domain.
        shape: the shape of the grid
    Attributes:
        domain: the range of the entire domain.
        regions: the array for the ranges of grid cells
        shape: the shape of the grid
        dim: the dimensionality of the grid
        cell_lengths: the length of cell for each dimension
        _centers: the array containing the coordinates of centers of grid cells
    """
    regions: np.ndarray  # array of Range
    shape: tuple
    domain: Range

    def __init__(self, domain: Range, shape):
        self.domain = domain
        self.regions: np.ndarray = build_grid(domain, shape)
        self.shape = shape
        cell_lengths = []
        dim = len(shape)
        self.dim = dim
        for i in range(dim):
            cell_lengths.append(domain[i].len() / shape[i])
        self.cell_lengths = np.array(cell_lengths)
        self._centers = None

    def get_cell_i_from_pts(self, points: np.ndarray) -> np.ndarray:
        """
        Args:
            points: the coordinates of points
        Returns:
            The cell indices to which the points belongs
        """
        assert len(points.shape) == 2
        shifted_pts = points - np.expand_dims(self.domain.low(), axis=0)
        cell_lengths = np.expand_dims(self.cell_lengths, axis=0)
        return np.floor(shifted_pts / cell_lengths).astype(int)

    def truncated_cell_i_from_pts(self, points: np.ndarray) -> np.ndarray:
        """
        Args:
            points: the coordinates of points
        Returns:
            The cell indices to which the points belongs,
                where the cell indices truncated by maximum value of cell index for each dimension.
        """
        points = self.domain.truncate(points)
        cell_i = self.get_cell_i_from_pts(points)
        return np.minimum((np.array(self.shape) - 1).reshape(1, 2), cell_i)

    def get_centers(self):
        """ Compute `self._centers` (the array containing the coordinates of centers of grid cells) """
        if self._centers is None:
            centers = np.empty(self.shape + (self.dim,))
            for i, range_ in np.ndenumerate(self.regions):
                range_: Range
                centers[i] = range_.center()
            self._centers = centers
        return self._centers

    def get_normalized_histo(self, pts) -> np.ndarray:
        """
        Args:
            pts: the coordinates of points
        Returns:
            The frequency array where each element corresponds to frequency of each bin of the grid,
                where the sum of frequencies is one.
        """
        return get_normalized_histo(pts, self.domain, self.shape)


class Histogram:
    """
    Args:
        grid: the grid information, which is an array of `Range` objects
        freq: the frequencies for the grid
    Attributes:
        grid: the grid information, which is an array of `Range` objects
        freq: the frequencies for the grid
    """
    grid: Grid
    freq: np.ndarray

    def __init__(self, grid: Grid, freq: np.ndarray):
        self.grid = grid
        self.freq = freq.reshape(grid.shape)


def answer_range_query_from_uniform_hist(hist: Histogram, q_range: Range, truncate=True):
    """
    Args:
        hist: a histogram
        q_range: a query range
        truncate: whether truncate the output value to make it non-negative.
    Returns:
        The frequency for a query range with using the input `hist` by assuming that each bin is uniformly distributed.
    """
    return answer_range_query_with_uniform_grid_histogram(hist.grid.domain, hist.grid.regions, q_range, hist.freq,
                                                          truncate)


def grid_freq_from_uniform_hist(src_histo: Histogram, tgt_grid: Grid, truncate=True):
    """
    Args:
        src_histo: a histogram
        tgt_grid: a grid information
        truncate: whether truncate the frequencies to make them non-negative.
    Returns:
        The frequencies for `tgt_grid` from `src_histo`
    """
    freq = np.empty(tgt_grid.shape)
    for i, range_ in np.ndenumerate(tgt_grid.regions):
        freq[i] = answer_range_query_from_uniform_hist(src_histo, range_, truncate)
    return freq


def build_grid(domain, shape):
    """
    Args:
        domain: the entire domain
        shape: the shape of the output grid
    Returns:
        grid: the array of `Range` object that make up the grid.
    """
    grid = np.empty(shape, Range)
    for idx, sub_range in iter_nested_split(domain, shape):
        grid[idx] = sub_range
    return grid


def get_normalized_histo(points: np.ndarray, domain: Range, shape) -> np.ndarray:
    """
    Args:
        points: the data points
        domain: the data domain
        shape: the grid shape to build histogram for the domain
    Returns:
        The numpy array where each element corresponds to frequency of each bin of the normalized histogram,
            where the sum of frequencies is one.
    """
    grid_freq = np.histogram2d(points[:, 0], points[:, 1], shape, range=domain.arr())[0]
    sum_ = grid_freq.sum()
    if sum_ > 0.:
        grid_freq /= sum_
    return grid_freq


def iter_nested_split(range_: Range, b: List[int]) -> Iterable[Tuple[Tuple, Range]]:
    """
    Iterate equally divided ranges from a multi-dimensional range.
    Args:
        range_: the entire range.
        b: a list containing the number of splits for each dimension.
    """
    for i, range_1d in iter_1d_split(range_[0], b[0]):
        if len(range_) == 1 and len(b) == 1:
            yield (i,), Range([range_1d])
        elif len(range_) > 1 and len(b) > 1:
            for indices, sub_range in iter_nested_split(Range(range_[1:]), b[1:]):
                yield (i,) + indices, Range([range_1d] + sub_range.range)
        else:
            raise Exception(f'Invalid: len(range_) = {len(range_)}, len(b) = {len(b)}')


def iter_1d_split(range_: Range1D, b):
    """
    Iterate equally divided ranges from a 1-D range.
    Args:
        range_: the entire range.
        b: the number of splits
    """
    width = range_.len() / b
    start = range_.min()
    stop = start + width
    for i in range(b):
        yield i, Range1D(start, stop)
        start += width
        stop += width
