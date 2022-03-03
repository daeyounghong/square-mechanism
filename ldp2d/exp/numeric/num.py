__author__ = "Daeyoung Hong, Woohwan Jung"
__date__ = "2022.02.26."

from abc import abstractmethod
from multiprocessing import Pool

import numpy as np

from classes import Grid, Histogram, get_normalized_histo
from data import OriginalData
from ldp2d.config.config import num_cpus, num_of_chunks_per_core, parallel
from ldp2d.exp.alg import Alg
from ldp2d.utils.others import split_arr, calc_mse


class NumAlg(Alg):
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
        preprocessed (bool): whether the input data is preprocessed
        ans_grid_shape: the grid shape used when computing the accuracy of frequency estimation (e.g., (600, 300)).
        estimated_histo (Hist): the estimated histogram by the algorithm
        eps (float): the privacy budget
        seed (int): the random seed
        re_sanity_bound (float): the sanity bound for computing relative errors
        results (dict): the evaluate measures will be saved
        perturbed_pts (np.ndarray): the perturbed points
        preprocessed_pts (np.ndarray): the preprocessed input points
    """
    perturbed_pts: np.ndarray
    preprocessed_pts: np.ndarray  # the preprocessed data can be used for multiple epsilons, e.g. normalization for PM.

    @abstractmethod
    def _perturb_by_one_cpu(self, arr: np.ndarray, eps) -> np.ndarray:
        """
        Args:
            arr: the data array
            eps (float): the privacy budget
        Returns:
            The perturbed data array.
        """
        pass

    def _run(self, **kwargs):
        """
        1. Run the perturbation algorithm
        2. Compute the estimated frequency distribution directly using `self.perturbed_pts`
        """
        # Initialize the perturbation algorithm
        eps = self.eps
        self.initialize(eps)

        # Collect the perturbed points
        self._collect()
        self.perturbed_pts = self.domain.truncate(self.perturbed_pts)

        # Compute the frequency matrix based on the perturbed points
        self.estimated_histo = Histogram(
            Grid(self.domain, self.ans_grid_shape),
            get_normalized_histo(self.perturbed_pts, self.dataset.domain, self.ans_grid_shape))

    def _collect(self):
        """
        Perturb the preprocessed data points, and perform the postprocessing the perturbed data if needed.
        As a result, the following attribute is computed: `self.perturbed_pts`
        """
        eps = self.eps
        if not self.preprocessed:
            self.preprocessed_pts = self.dataset.arr
            self.preprocessed = True

        if parallel is False:
            perturbed = self._perturb_by_one_cpu(self.preprocessed_pts, eps)
        else:
            arr = self.preprocessed_pts
            num_split = num_cpus * num_of_chunks_per_core
            original_shape = arr.shape
            with Pool(num_cpus) as pool:
                perturbed = pool.starmap(self._perturb_by_one_cpu, ((t, eps) for t in split_arr(arr, num_split)))
            perturbed = np.concatenate(perturbed, axis=0)
            perturbed = np.array(perturbed).reshape(original_shape)
        self.perturbed_pts = self._postprocess(perturbed)

    def _postprocess(self, perturbed: np.ndarray):
        """
        Args:
            perturbed: the perturbed data
        Returns:
            The postprocessed data array. By default, no postprocessing is applied.
        """
        return perturbed

    def evaluate(self):
        """
        Compute the measures, such as MSE and MRE, based on results of the algorithm.
        The measures and values are saved in the dictionary `self.results`.
        Here, the mse of the perturbed locations is saved as `{'mse': mse_value}`
            where mse_value is the mse of the perturbed locations.
        """
        super().evaluate()
        if self.perturbed_pts is not None:
            mse = calc_mse(self.dataset.arr, self.perturbed_pts)
            self.results.update({'mse': mse})
        else:
            self.results.update({'mse': None})
