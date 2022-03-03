__author__ = "Daeyoung Hong, Woohwan Jung"
__date__ = "2022.02.26."

import gc
from itertools import product
from multiprocessing import Pool
from typing import Union

import matlab.engine
import numpy as np
from numpy import exp, ceil, floor
from scipy import sparse
from scipy.sparse import lil_matrix, vstack, spmatrix

from classes import Range, Range1D, overlap, Histogram, Grid, get_normalized_histo
from data import OriginalData
from ldp2d.alg.func.tls import calc_square_center, SquareMecAlg
from ldp2d.config.config import num_cpus, num_of_chunks_per_core, dim, parallel, prec_bnd
from ldp2d.exp.numeric.num import NumAlg
from ldp2d.utils.config import TmpSaveLoader, smoothed_relative_err
from ldp2d.utils.others import my_tqdm, reduce_mul, is_valid_idx, Cumu


def count_nnz(list_):
    """
    Args:
        list_: the list of matrics or vectors
    Returns:
        The sum where the number of non-zero elements is added for sparse data structures and
            the number of elements is added for dense data structures.
    """
    nnz = 0
    for x in list_:
        if isinstance(x, np.ndarray):
            nnz += reduce_mul(x.shape)
        elif isinstance(x, spmatrix):
            nnz += x.nnz
        else:
            raise Exception('Invalid vector or matrix type: {}'.format(type(x)))
    return nnz


class SquareMechanism(NumAlg):
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
        l_x: the length of domain in the x-axis
        l_y: the length of domain in the y-axis
        b: the optimal side length of the square region will be saved
        alpha: the probability density (alpha) outside the square region.
    """
    name = 'SM_avg'
    b: float  # the side-length
    alpha: float

    def __init__(self, dataset: OriginalData, ans_grid_shape, eps, seed, re_sanity_bound, **kwargs):
        super().__init__(dataset, ans_grid_shape, eps, seed, re_sanity_bound, **kwargs)

        # Compute l_x (the length of domain in the x-axis) and l_y (the length of domain in the y-axis)
        l_x, l_y = SquareMecAlg.get_domain_lengths(self.domain)
        self.l_x = l_x
        self.l_y = l_y

    def _compute_b(self, eps: float):
        """
        `self.b` is set to the optimal side length of the square region.
        :param eps: the privacy budget epsilon
        """
        self.b = SquareMecAlg.optimized_b_for_avg(self.l_x, self.l_y, eps)

    def initialize(self, eps: float):
        """
        `self.b` is set to the optimal side length of the square region.
        In addition, `self.alpha` is set to the probability density (alpha) outside the square region.
        :param eps: the privacy budget epsilon
        """
        self._compute_b(eps)
        self._compute_alpha(eps)

    def _perturb_by_one_cpu(self, arr: np.ndarray, eps: float) -> np.ndarray:
        """
        Args:
            arr: the data array
            eps (float): the privacy budget
        Returns:
            The perturbed data array by the square mechanism.
        """
        return SquareMecAlg.perturb(eps, self.get_out_domain(), self.b, arr)

    def _compute_alpha(self, eps: float):
        """
        `self.alpha` is set to the probability density (alpha) outside the square region.
        :param eps: the privacy budget epsilon
        """
        # Compute alpha according to Equation (3)
        out_domain = self.get_out_domain()
        l_x = out_domain[0].len()
        l_y = out_domain[1].len()
        self.alpha = SquareMecAlg.low_prob(l_x * l_y, eps, self.b)

    @classmethod
    def compute_square_region(cls, t: np.ndarray, out_domain: Range, b: float) -> Range:
        """
        Args:
            t: an original point to perturb. Note that it is only one data point.
            out_domain: the output domain
            b: the side length of the square region
        Returns:
            A `Range` object indicating the square region for the input `t`
        """
        t = t.flatten()
        assert len(t) == dim
        center = calc_square_center(t, out_domain, b).reshape(2)
        return Range([Range1D(center_1d - b / 2, center_1d + b / 2) for center_1d in center])

    def get_out_domain(self) -> Range:
        """
        Returns:
            The output domain.
        """
        return self.domain

    def self_square_region(self, t: np.ndarray) -> Range:
        """
        Args:
            t: an original point to perturb. Note that it is only one data point.
        Returns:
            A `Range` object indicating the square region for the input `t`
                with the current output domain and side length of the square region.
        """
        return self.compute_square_region(t, self.get_out_domain(), self.b)


class GeospatialDataCollectorLDP(NumAlg):
    """
    Args:
        dataset (OriginalData): the input dataset
        ans_grid_shape: the grid shape used when computing the accuracy of frequency estimation (e.g., (600, 300)).
        eps (float): the privacy budget
        seed (int): the random seed.
        re_sanity_bound (float): the sanity bound for computing relative errors
        grid_shape: the grid shape used for frequency estimation (e.g., (600, 300)).
        prob (str): 'cumu' (use prefix-sums) or 'vanilla' (without prefix-sums)
        l2_coeff (float): L2 regularization coefficient (denoted by lambda in the paper)
        kwargs (dict): other parameters for convex optimization solver containing the following fields:
            'optimality_tolerance', 'constraint_tolerance', 'max_iter', 'step_tol'.
            Please see https://www.mathworks.com/help/optim/ug/quadprog.html for the details of the parameters.
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
        grid_shape: the grid shape used for frequency estimation (e.g., (600, 300)).
        cell_lengths: contains the cell length of each dimension
        prob (str): 'cumu' (use prefix-sums) or 'vanilla' (without prefix-sums)
        l2_coeff (float): L2 regularization coefficient (denoted by lambda in the paper)
        grid: the grid generated with `grid_shape` and `domain`
        original_histo: the original frequency distribution
        perturbed_histo: the perturbed histogram computed directly from the perturbed points
        cell_area: the area of one cell
        mec: the object for square mechanism
    """
    grid: Grid
    original_histo: Histogram
    perturbed_histo: Union[Histogram, None]
    cell_area: float
    mec: SquareMechanism
    name = 'matlab'

    def __init__(self, dataset: OriginalData, ans_grid_shape, eps, seed, re_sanity_bound, grid_shape,
                 prob: str, l2_coeff: float, **kwargs):
        assert all((k in kwargs) for k in ['optimality_tolerance', 'constraint_tolerance', 'max_iter', 'step_tol'])
        super().__init__(dataset, ans_grid_shape, eps, seed, re_sanity_bound, **kwargs)
        # Set the parameters
        dim = 2
        self.grid_shape = grid_shape
        self.perturbed_histo = None
        self.prob = prob
        self.l2_coeff = l2_coeff
        self.cumu = (prob == 'cumu')

        # Compute the cell length of each dimension
        cell_lengths = []
        for i in range(dim):
            cell_lengths.append(self.domain[i].len() / grid_shape[i])
        self.cell_lengths = cell_lengths

    def _prepare_freq_estimation(self, **kwargs):
        """
        - `self.grid` is set
        - `self.original_histo` is set
        """
        dataset = self.dataset
        grid_shape = self.grid_shape
        domain = self.domain

        # Construct the grid of the domain
        self.grid = grid = Grid(domain, grid_shape)

        # Build the frequency matrix of the original points with the grid
        self.original_histo = Histogram(grid, get_normalized_histo(dataset.arr, dataset.domain, grid_shape))

    def solve_quadratic_programming(self, grid_freq, eps: float, loader: TmpSaveLoader):
        """
        Args:
            grid_freq: perturbed frequency matrix
            eps: the privacy budget epsilon
            loader: the loader to load the S matrix from the saved file to save memory
        Returns: estimated_freq, info_dict
            estimated_freq: estimated frequency matrix
            info_dict:
               {'status': (the status information from the MATLAB solver when its operation is finished),
                'iter': (the number of iterations taken in the solver),
                'qp_nnz': (the number of non-zero elements in matrices and vectors used in the solver)}
        """
        flattened_grid_freq = grid_freq.flatten()
        l2_coeff = self.l2_coeff
        grid_shape = self.grid_shape
        cell_area = self.cell_area
        mec = self.mec
        cumu = self.cumu
        kwargs = self._kwargs
        step_tol = kwargs['step_tol']

        neighbor_mat = loader.load()

        alpha = mec.alpha

        H, f, A, b, Aeq, beq, lb, ub = self.for_matlab(grid_shape, alpha, cell_area, eps, neighbor_mat,
                                                       flattened_grid_freq, l2_coeff, self.cumu)
        qp_nnz = count_nnz([H, f, A, b, Aeq, beq, lb, ub])
        del neighbor_mat
        x0 = Cumu.to_cumu(grid_freq).flatten() if cumu else grid_freq.flatten()  # initial guess
        params = {'H': H, 'f': f, 'A': A, 'b': b, 'Aeq': Aeq, 'beq': beq, 'lb': lb, 'ub': ub, 'x0': x0}

        # process sparse matrices
        for k in ['H', 'A', 'Aeq']:
            if isinstance(params[k], np.ndarray):
                coo = sparse.coo_matrix(params[k])
            else:
                coo = params[k].tocoo()
            for sub_k in ('row', 'col', 'data'):
                params[k + '_' + sub_k] = getattr(coo, sub_k)
                if sub_k in {'row', 'col'}:
                    params[k + '_' + sub_k] += 1  # for matlab, 1-based indexing is used
            params[k + '_' + 'shape'] = np.array(coo.shape)
            params.pop(k)

        for k, v in params.items():
            params[k] = matlab.double(v.tolist())
        if True or num_cpus == 1:
            eng = matlab.engine.start_matlab('-singleCompThread')
            print("MATLAB is started with single thread mode.")
        else:
            eng = matlab.engine.start_matlab()
        gc.collect()
        x, status, iter_ = eng.qp(
            params['H_shape'], params['H_row'], params['H_col'], params['H_data'], params['f'],
            params['A_shape'], params['A_row'], params['A_col'], params['A_data'], params['b'],
            params['Aeq_shape'], params['Aeq_row'], params['Aeq_col'], params['Aeq_data'], params['beq'],
            params['lb'], params['ub'], params['x0'], kwargs['optimality_tolerance'],
            kwargs['constraint_tolerance'], 1, kwargs['max_iter'], step_tol, nargout=3)
        if status in {0, 1, 2}:
            x = np.array(x)
            status = status
        else:
            status = 'failed'

        estimated_freq = (Cumu.build_mat_to_convert_to_ori(grid_shape) @ x).reshape(grid_shape) \
            if cumu else x.reshape(grid_shape)
        return estimated_freq, dict(status=status, iter=iter_, **{'qp_nnz': qp_nnz})

    @classmethod
    def for_matlab(cls, grid_shape, alpha, cell_area, eps, neigh_mat, perturbed_freq, l2_coeff: float, cumu: bool):
        """
        Args:
            grid_shape: the grid shape used for frequency estimation (e.g., (600, 300)).
            alpha: the probability density (alpha) outside the square region.
            cell_area: the area of one cell
            eps: the privacy budget epsilon
            neigh_mat: S matrix in the paper
            perturbed_freq: the perturbed frequency matrix computed directly from the perturbed points
            l2_coeff (float): L2 regularization coefficient (denoted by lambda in the paper)
            cumu: whether to use the prefix-sums method
        Returns:
            Matrices and vectors for arguments of MATLAB quadratic programming solver. See
                https://www.mathworks.com/help/optim/ug/quadprog.html.
        """
        n = reduce_mul(grid_shape)
        neigh_mat = sparse.csc_matrix(neigh_mat)
        if cumu:
            r_mat = Cumu.build_mat_to_convert_to_ori(grid_shape).tocsc()
            a_mat_bottom = lil_matrix((1, n))
            a_mat_bottom[0, -1] = 1.
        else:  # vanilla
            r_mat = sparse.identity(n)
            a_mat_bottom = sparse.csc_matrix(np.ones((1, n)))

        p_mat = (2 * (alpha * cell_area * (exp(eps) - 1)) ** 2 * neigh_mat.transpose() @ neigh_mat).tocsc()
        p_mat += 2 * l2_coeff * r_mat.transpose() @ r_mat  # for regularization
        q_vec = 2 * alpha * cell_area * (exp(eps) - 1) * (alpha * cell_area - perturbed_freq).T @ neigh_mat

        # a_mat = lil_matrix((n + 1, n), dtype=np.float)
        a_mat = vstack([r_mat, a_mat_bottom]).tocsc()
        # a_mat[-1, -1] = 1.
        l_vec = np.zeros(n + 1)
        l_vec[-1] = 1.
        u_vec = np.ones(n + 1)
        if not cumu:
            A = np.array([])
            b = np.array([])
            lb = np.zeros(n)
            ub = np.ones(n)
        else:
            A = - r_mat
            b = np.zeros(n)
            lb = np.array([])
            ub = np.array([])
        beq = np.ones(1)
        Aeq = a_mat_bottom
        H = p_mat
        f = q_vec
        return H, f, A, b, Aeq, beq, lb, ub

    def _run(self, **kwargs):
        """
        - `self._prepare_freq_estimation()` is called.
        - `self.mec` is set and performs perturbation.
        - `self.perturbed_pts` is set.
        - `self.cell_area` is computed.
        - `self.perturbed_histo` is set to the perturbed histogram.
        - In `self.results` is updated with the following:
               {'freq': (estimated frequency matrix),
                'status': (the status information from the MATLAB solver when its operation is finished),
                'iter': (the number of iterations taken in the solver),
                'qp_nnz': (the number of non-zero elements in matrices and vectors used in the solver)}
        - `self.estimated_histo` is computed.
        """
        # Initialize and set parameters
        self._prepare_freq_estimation()
        dataset = self.dataset
        ans_grid_shape = self.ans_grid_shape
        eps = self.eps
        seed = self.seed
        re_sanity_bound = self.re_sanity_bound
        self.cell_area = self.grid.regions[0, 0].area()

        # Collect perturbed data using the square mechanism
        mec = SquareMechanism(dataset, ans_grid_shape, eps, seed, re_sanity_bound, **self._kwargs)
        self.mec = mec
        self.mec.run()
        perturbed_pts = mec.perturbed_pts
        self.perturbed_pts = perturbed_pts

        # Compute the perturbed frequency distribution
        self.perturbed_histo = Histogram(self.grid, self.grid.get_normalized_histo(perturbed_pts))

        # Estimate the original frequency distribution by Copt
        estimated_freq, results = self.run_copt(eps)
        self.estimated_histo = Histogram(self.grid, estimated_freq)
        if results is not None:
            self.results.update(results)
        self.results.update({'freq': estimated_freq})

    def run_copt(self, eps: float):
        """
        Args:
            eps: the privacy budget epsilon
        Returns: estimated_freq, results
            estimated_freq: the estimated frequency matrix
            results (dict):
               {'status': (the status information from the MATLAB solver when its operation is finished),
                'iter': (the number of iterations taken in the solver),
                'qp_nnz': (the number of non-zero elements in matrices and vectors used in the solver)}
        """
        grid_freq = self.perturbed_histo.freq

        # Build the matrix S for the quadratic programming (Please refer to Section 5.3 in the paper)
        neighbor_mat = self._build_neighbor_mat(parallel=parallel, cumu=False if self.prob == 'vanilla' else True)
        obj_nnz = neighbor_mat.nnz
        loader = TmpSaveLoader('neigh_mat', neighbor_mat)
        del neighbor_mat
        gc.collect()

        # Solve the quadratic programming by MATLAB solver
        with loader:
            estimated_freq, results = self.solve_quadratic_programming(grid_freq, eps, loader)
            print('{}'.format(results))
            gc.collect()

        results.update({'obj_nnz': obj_nnz})
        return estimated_freq, results

    def _build_neighbor_mat(self, parallel: bool, cumu: bool, sparse=True):
        """
        Set the matrix S in Equation (17) in Section 5.3 of the paper.
        Args:
            parallel: whether to use multi-processing
            cumu: whether to use the prefix-sum technique
            sparse: whether to use sparse matrix
        Returns:
            S matrix in the paper
        """
        print('Start to build the neighbor matrix')
        grid_shape = self.grid_shape
        mat_shape = 2 * grid_shape
        n = reduce_mul(grid_shape)

        # Initialize the matrix S
        neighbor_mat = lil_matrix((n, n), dtype=np.float) if sparse else np.zeros(mat_shape)

        # For each cell (i, j)
        for ref_idx, (range_list, overlap_ratio_list) in my_tqdm(
            self._compute_neighbor_range_and_ratio(parallel), desc='_build_neighbor_mat', total=n):

            # Compute the $v_B_{i, j}$
            # Set S[(i − 1)n2 + j] to $v_B_{i, j}$
            for range_, overlap_ratio in zip(range_list, overlap_ratio_list):

                if cumu is False:
                    # Do not use the prefix-sum technique
                    if isinstance(neighbor_mat, np.ndarray):
                        neighbor_mat[tuple(ref_idx)][range_[0, 0]:range_[0, 1] + 1, range_[1, 0]:range_[1, 1] + 1] \
                            += overlap_ratio
                    else:
                        for i in range(range_[0, 0], range_[0, 1] + 1):
                            for j in range(range_[1, 0], range_[1, 1] + 1):
                                neighbor_mat[
                                    np.ravel_multi_index(ref_idx, grid_shape), np.ravel_multi_index((i, j), grid_shape)
                                ] += overlap_ratio
                else:
                    # Use the prefix-sum technique
                    for idx, sign in Cumu.range_to_flattened_indices(range_, grid_shape):
                        neighbor_mat[np.ravel_multi_index(ref_idx, grid_shape), idx] += sign * overlap_ratio
        if isinstance(neighbor_mat, np.ndarray):
            neighbor_mat = neighbor_mat.reshape((n, n))
        else:
            neighbor_mat = neighbor_mat.tocsc()
            print('NNZ = {}'.format(neighbor_mat.nnz))
        gc.collect()
        return neighbor_mat

    def _compute_neighbor_range_and_ratio(self, parallel: bool):
        """
        Args:
            parallel: whether to use multi-processing
        Returns:
            the list where each element is tuple ((cell index), (the tuple of ('ranges', 'sigmas')))
                where 'ranges' is the list of ranges of subgrids for the cell
                        (e.g., A range [[ 0, 10], [ 0, 11]] means G(0, 0, 10, 11) in the paper)
                    and 'sigmas' is the list of sigma values for the subgrids.
        """
        grid = self.grid.regions
        grid_shape = self.grid_shape
        n = reduce_mul(grid_shape)  # the size of optimization variable vector
        idx_list = [idx for idx, _ in np.ndenumerate(grid)]  # for the perturbed if transpose
        if parallel is False:
            output_list = (self._compute_neighbor_range_and_ratio_for_idx(np.array(idx)) for idx in idx_list)
        else:
            chunk_size = int(np.ceil(n / (num_cpus * num_of_chunks_per_core)))
            with Pool(num_cpus) as pool:
                # neighbor index range
                output_list = pool.starmap(self._compute_neighbor_range_and_ratio_for_idx,
                                           ((np.array(idx),) for idx in idx_list), chunksize=chunk_size)
        return ((idx, output) for idx, output in zip(idx_list, output_list))

    def _compute_neighbor_range_and_ratio_for_idx(self, ref_idx: np.ndarray):
        """
        :param ref_idx: the cell index
        Returns: range_list, overlap_ratio_list
            range_list: the list of ranges of subgrids for the cell
                            (e.g., A range [[ 0, 10], [ 0, 11]] means G(0, 0, 10, 11) in the paper)
            overlap_ratio_list: the list of sigma values for the subgrids.
        """
        ref_idx = np.array(ref_idx)
        mec = self.mec
        ref_cells = self.grid.regions  # if transpose
        tgt_grid = self.grid  # if transpose
        tgt_cells = self.grid.regions  # if transpose
        ref_cell = ref_cells[tuple(ref_idx)]

        strong_neighbor_idx_range = self._compute_strong_neighbor_range(ref_idx)
        analytic_strong_neigh_range = self._compute_analytic_strong_neigh_range(ref_idx)
        if False and not np.all(strong_neighbor_idx_range == analytic_strong_neigh_range):
            raise Exception(
                'strong_neigh_range: {} v.s. {}'.format(strong_neighbor_idx_range, analytic_strong_neigh_range))

        overlap_ratio_list = []

        if np.all(strong_neighbor_idx_range[:, 1] - strong_neighbor_idx_range[:, 0] >= 0):
            range_list = [strong_neighbor_idx_range]
            overlap_ratio_list.append(1.0)
            diff = [-1, 1]
            # For edges
            side_ranges = []
            side_expansions = np.zeros((2, 2, 2), dtype=np.int)
            for dim_i in range(dim):
                for diff_i in range(2):
                    cell_i = np.copy(strong_neighbor_idx_range[:, diff_i])
                    cell_i[dim_i] += diff[diff_i]
                    for layer_i in range(1):
                        if not is_valid_idx(cell_i, tgt_cells.shape):
                            continue
                        start_i = cell_i[dim_i]
                        curr_cell: Range = tgt_cells[tuple(cell_i)]
                        init_overlap_ratio = self._neigh_overlap_ratio_from_center(ref_cell, curr_cell, mec)
                        if init_overlap_ratio == 0.:
                            continue
                        while True:
                            cell_i[dim_i] += diff[diff_i]
                            if not is_valid_idx(cell_i, tgt_cells.shape):
                                break
                            overlap_ratio = self._neigh_overlap_ratio_from_center(ref_cell, tgt_cells[tuple(cell_i)],
                                                                                  mec)
                            if abs(init_overlap_ratio - overlap_ratio) > prec_bnd:
                                break
                        side_expansions[dim_i][diff_i][layer_i] = diff[diff_i] * (cell_i[dim_i] - start_i)
                        side_expansions[dim_i][diff_i][layer_i] = max(1, side_expansions[dim_i][diff_i][layer_i])

            assert np.all(side_expansions >= 0)
            ranges = np.empty((2, 6), dtype=int)
            overlap_ratio_list = []
            range_list = []
            for dim_i in range(2):
                ranges[dim_i] = strong_neighbor_idx_range[dim_i][0] - side_expansions[dim_i][0].sum(), \
                                strong_neighbor_idx_range[dim_i][0] - side_expansions[dim_i][0][0], \
                                strong_neighbor_idx_range[dim_i][0], strong_neighbor_idx_range[dim_i][1] + 1, \
                                strong_neighbor_idx_range[dim_i][1] + 1 + side_expansions[dim_i][1][0], \
                                strong_neighbor_idx_range[dim_i][1] + 1 + side_expansions[dim_i][1].sum()

            for i0, i1 in product(range(5), range(5)):
                range_ = np.empty((2, 2), dtype=int)
                empty = False
                for dim_i, i_ in enumerate([i0, i1]):
                    range_[dim_i] = ranges[dim_i, i_], ranges[dim_i, i_ + 1]
                    if range_[dim_i, 1] == range_[dim_i, 0]:
                        empty = True
                        break
                if empty:
                    continue
                overlap_ratio = self._neigh_overlap_ratio_from_center(ref_cell, tgt_cells[tuple(range_[:, 0])], mec)
                range_to_append = np.copy(range_)
                range_to_append[:, 1] -= 1
                range_list.append(range_to_append)
                overlap_ratio_list.append(overlap_ratio)

        else:
            idx = tgt_grid.truncated_cell_i_from_pts(np.array([ref_cell.center()])).reshape((2,))
            assert idx.shape == (2,), idx.shape
            curr_cell = tgt_grid.regions[tuple(idx)]
            range_list = [np.array([[ref_idx[0], ref_idx[0]], [ref_idx[1], ref_idx[1]]])]
            overlap_ratio_list.append(self._neigh_overlap_ratio_from_center(ref_cell, curr_cell, mec))

        assert len(range_list) == len(overlap_ratio_list)
        return range_list, overlap_ratio_list

    def _compute_strong_neighbor_range(self, ref_idx: np.ndarray):
        """
        :param ref_idx: the cell index
        Returns:
            The array for the range of the 4-neighbor subgrid.
            (e.g., A range [[ 0, 10], [ 0, 11]] means G(0, 0, 10, 11) in the paper)
        """
        strong_neighbor_idx_range = np.empty((dim, 2), np.int)  # (low, high) for each dimension
        for dim_i in range(dim):
            for i, diff in enumerate((-1, 1)):
                idx_diff = np.zeros(dim, np.int)
                idx_diff[dim_i] = diff
                last_idx = self._search_identical_bound_idx_for_a_direction(ref_idx, idx_diff, val=1.0)
                strong_neighbor_idx_range[dim_i, i] = last_idx[dim_i]
        return strong_neighbor_idx_range

    def _compute_analytic_strong_neigh_range(self, ref_idx: np.ndarray):
        """
        :param ref_idx: the cell index
        Returns:
            The array for the range of the 4-neighbor subgrid.
            (e.g., A range [[ 0, 10], [ 0, 11]] means G(0, 0, 10, 11) in the paper)
        """
        out = np.empty((2, 2), dtype=int)
        w = self.mec.b
        w_cell = self.cell_lengths
        grid_shape = self.grid_shape
        ref_idx = ref_idx + 1
        for d in range(dim):
            if ref_idx[d] < w / w_cell[d]:
                out[d, 0] = 1
            else:
                out[d, 0] = ceil(ref_idx[d] + 1 / 2 - w / (2 * w_cell[d]))
            if grid_shape[d] - w / w_cell[d] + 1 < ref_idx[d]:
                out[d, 1] = grid_shape[d]
            else:
                out[d, 1] = floor(ref_idx[d] - 1 / 2 + w / (2 * w_cell[d]))
        out -= 1
        return out

    @classmethod
    def _neigh_overlap_ratio_from_center(cls, ref_cell_region: Range, tgt_cell_region: Range, mec: SquareMechanism):
        """
        Args:
            ref_cell_region: A cell region of the cell (i, j) to compute sigma_(k, l)^(i, j)
            tgt_cell_region: A cell region of the cell (k, l) to compute sigma_(k, l)^(i, j)
            mec: the square mechanism containing the information of the optimal side length of the square region.
        Returns:
            sigma_(k, l)^(i, j)
                where (i, j) is the index of `ref_cell_region` and (k, l) is the index of `tgt_cell_region`.
        """
        ref_cell_region, tgt_cell_region = tgt_cell_region, ref_cell_region
        neigh_region = mec.self_square_region(ref_cell_region.center())
        return overlap(neigh_region, tgt_cell_region).area() / tgt_cell_region.area()

    def _search_identical_bound_idx_for_a_direction(self, ref_idx: np.ndarray, diff: np.ndarray, val: float = None):
        """
        Args:
            ref_idx: a cell index
            diff: a search direction (e.g., [1, 0] means searching by increasing the cell index of the first dimension)
            val: a floating point value
        Returns:
            The last cell index during searching the cells whose sigma is equal to `val`.
        """
        mec = self.mec
        ori_grid = self.grid
        out_grid = self.grid

        ref_cell: Range = out_grid.regions[tuple(ref_idx)]
        idx = ori_grid.truncated_cell_i_from_pts(np.array([ref_cell.center()])).reshape((2,))
        assert idx.shape == (2,), idx.shape
        curr_cell = ori_grid.regions[tuple(idx)]
        while True:
            if not (smoothed_relative_err(
                val, self._neigh_overlap_ratio_from_center(ref_cell, curr_cell, mec)) < prec_bnd):
                break
            idx = idx + diff
            if not (all(idx >= 0) and np.all(idx <= np.array(self.grid_shape) - 1)):
                break
            curr_cell = ori_grid.regions[tuple(idx)]
        return idx - diff

    def _perturb_by_one_cpu(self, arr: np.ndarray, eps):
        pass
