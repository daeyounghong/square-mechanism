__author__ = "Daeyoung Hong, Woohwan Jung"
__date__ = "2022.02.26."

from abc import ABC, abstractmethod
from typing import Union

from classes import Range, Grid, Histogram, get_normalized_histo, grid_freq_from_uniform_hist
from data import OriginalData
from ldp2d import key
from ldp2d.utils.others import set_seed
from ldp2d.utils.config import freq_est_err


class Alg(ABC):
    """
    Args:
        dataset (OriginalData): the input dataset
        ans_grid_shape: the grid shape used when computing the accuracy of frequency estimation (e.g., (600, 300)).
        eps (float): the privacy budget
        seed (int): the random seed.
        re_sanity_bound (float): the sanity bound for computing relative errors

    Attributes:
        dataset (OriginalData): the input dataset
        domain (Range): the domain of the dataset
        original_pts (np.ndarray): the array of coordinates of the original points
        preprocessed (bool): whether the input data is preprocessed
        ans_grid_shape: the grid shape used when computing the accuracy of frequency estimation (e.g., (600, 300)).
        estimated_histo (Hist): the estimated histogram by the algorithm
        eps (float): the privacy budget
        seed (int): the random seed
        re_sanity_bound (float): the sanity bound for computing relative errors
        results (dict): the evaluate measures will be saved
    """
    domain: Range
    estimated_histo: Union[Histogram, None]
    name: str

    def __init__(self, dataset: OriginalData, ans_grid_shape, eps: float, seed: int, re_sanity_bound: float, **kwargs):
        # Only parameters are set
        self.dataset = dataset
        self.domain = dataset.domain
        self.original_pts = dataset.arr
        self.preprocessed = False
        self.ans_grid_shape = ans_grid_shape
        self.estimated_histo = None
        self.eps = eps
        self.seed = seed
        self.re_sanity_bound = re_sanity_bound
        self.results = {}
        self._kwargs = kwargs

    @abstractmethod
    def _run(self, **kwargs):
        """ Run the algorithm """
        pass

    @abstractmethod
    def _collect(self):
        """ Collect the perturbed data """
        pass

    def initialize(self, eps: float):  # set parameters for eps for plotting parameters
        """
        Prepare data perturbation given privacy budget eps.
        :param eps: the privacy budget epsilon
        """
        pass

    def run(self, **kwargs) -> dict:
        """ Run the algorithm and compute the measures such as MSE and MRE. """

        # Set the random seed and run the algorithm
        set_seed(self.seed)
        self._run(**kwargs)

        # Evaluate the results
        self.evaluate()

        return self.results

    def evaluate(self):
        """
        Compute the measures, such as MSE and MRE, based on results of the algorithm.
        The measures and values are saved in the dictionary `self.results`.
        Here, the accuracy of the frequency estimation is saved as `{key.freq_est: {'re': re_array}}`
            where re_array consists of relative error for each bin.
        """
        domain = self.domain
        ans_grid_shape = self.ans_grid_shape

        ans_histo = get_normalized_histo(self.original_pts, domain, ans_grid_shape)

        estimated_histo = self.estimated_histo
        assert estimated_histo is not None
        assert isinstance(estimated_histo, Histogram)

        transformed_estimated_histo = grid_freq_from_uniform_hist(
            estimated_histo, Grid(domain, ans_grid_shape), truncate=False)

        processed_ori_histo = ans_histo
        processed_est_histo = transformed_estimated_histo
        total_cnt = 1
        freq_est_res = freq_est_err(processed_ori_histo, processed_est_histo, total_cnt, truncate=False)
        self.results.update({key.freq_est: freq_est_res})
