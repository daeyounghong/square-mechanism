__author__ = "Daeyoung Hong, Woohwan Jung"
__date__ = "2022.02.26."

from typing import Union

import numpy as np
from numpy import exp
from numpy.random import binomial, uniform

from classes import Range
from ldp2d.config.config import dim
from ldp2d.utils.others import truncate


def calc_square_center(t: np.ndarray, out_domain: Range, b: Union[float, np.ndarray]):
    """
    :param t: the input data to perturb
    :param out_domain: the output domain.
    :param b: the side length of the square region
    :return: the centers of the square regions
    """
    if isinstance(b, np.ndarray):
        b = b.reshape(-1, 1)
    domain_u_bound = np.array([domain_1d.max() for domain_1d in out_domain]).reshape(1, 2)
    domain_l_bound = np.array([domain_1d.min() for domain_1d in out_domain]).reshape(1, 2)
    center_upper_bound = domain_u_bound - b / 2
    center_lower_bound = domain_l_bound + b / 2
    if not np.all(center_upper_bound - center_lower_bound >= 0):
        print('t = {}'.format(t))
        print('out_domain = {}'.format(out_domain))
        print('side_length = {}'.format(b))
        print('domain_u_bound = {}'.format(domain_u_bound))
        print('domain_l_bound = {}'.format(domain_l_bound))
        print('center_upper_bound = {}'.format(center_upper_bound))
        print('center_lower_bound = {}'.format(center_lower_bound))
        raise Exception('not all(center_upper_bound - center_lower_bound >= 0)')
    center = t
    center = np.minimum(center, center_upper_bound)
    center = np.maximum(center, center_lower_bound)
    return center


def square_region_sampling(b, out_domain: Range, t: np.ndarray):
    """
    :param b: the side length of the square region.
    :param out_domain: the output domain.
    :param t: the input data to perturb.
    :return: the points sampled from the square regions.
    """
    assert len(t.shape) == 2

    # Compute the centers of square regions by Equation (4)
    center = calc_square_center(t, out_domain, b)
    low = -b / 2
    high = b / 2
    if isinstance(b, np.ndarray):
        low = np.repeat(low.reshape(-1, 1), 2, axis=-1)
        high = np.repeat(high.reshape(-1, 1), 2, axis=-1)
    out = uniform(low=low, high=high, size=t.shape) + center
    for d, domain_1d in zip(range(out_domain.dim()), out_domain):
        out[:, d] = np.minimum(domain_1d.max(), out[:, d])
        out[:, d] = np.maximum(domain_1d.min(), out[:, d])
    return out


class SquareMecAlg:

    @classmethod
    def get_domain_lengths(cls, domain: Range):
        """
        :param domain: the domain of the input data.
        :return: the numpy array of the input data, l_x (the length of domain in the x-axis) and
            l_y (the length of domain in the y-axis)
        """
        l_x = domain[0].len()
        l_y = domain[1].len()
        return l_x, l_y

    @classmethod
    def perturb_with_sampled_neigh_points(cls, n: int, dim: int, pts_in_square_region, alpha, out_domain: Range) \
        -> np.ndarray:
        """
        :param n: the number of records in data.
        :param dim: the dimensionality of a record in data.
        :param pts_in_square_region: the sampled points from the square regions.
        :param alpha: the probability density (alpha) outside the square region.
        :param out_domain: the output domain.
        :return: the perturbed points by the square mechanism.
        """

        # truncate neighbors with out_domain
        pts_in_square_region = out_domain.truncate(pts_in_square_region)

        out_domain_volume = out_domain.area()
        prob_background = alpha * out_domain_volume
        domain_arr = np.array([[domain_1d.min(), domain_1d.max()] for domain_1d in out_domain])  # (d, 2)

        is_background = binomial(n=1, p=prob_background, size=n).reshape((n, 1))
        background = uniform(low=domain_arr[:, 0], high=domain_arr[:, 1], size=(n, dim))
        return np.where(is_background, background, pts_in_square_region)

    @classmethod
    def preprocess(cls, domain: Range, t):
        """
        :param domain: the domain of the input data.
        :param t: the input data to perturb.
        :return: the numpy array of the input data, l_x (the length of domain in the x-axis) and
            l_y (the length of domain in the y-axis)
        """
        assert domain.dim() == 2
        l_x, l_y = cls.get_domain_lengths(domain)
        t = np.array(t)
        return t, l_y, l_x

    @classmethod
    def post_process(cls, alpha, b, out_domain: Range, t) -> np.ndarray:
        """
        :param alpha: the probability density (alpha) outside the square region.
        :param b: the side length of the square region
        :param out_domain: the output domain.
        :param t: the input data to perturb
        :return: the perturbed points by the square mechanism.
        """
        assert len(t.shape) == 2

        # Sample perturbed points by Equation (2)
        pts_in_square_region = square_region_sampling(b, out_domain, t)
        perturbed_pts = cls.perturb_with_sampled_neigh_points(len(t), dim, pts_in_square_region, alpha, out_domain)
        return perturbed_pts

    @classmethod
    def low_prob(cls, domain_area, eps, side_length):
        """
        :param domain_area: the area of the domain
        :param eps: the privacy budget epsilon
        :param side_length: the side length of the square region
        :return: the probability density (alpha) outside the square region.
        """
        return 1 / (domain_area + (exp(eps) - 1) * side_length ** 2)

    @classmethod
    def perturb(cls, eps, out_domain: Range, b, t) -> np.ndarray:
        """
        :param eps: the privacy budget epsilon
        :param out_domain: the output domain
        :param b: the side length of the square region
        :param t: the input data to perturb
        :return: the perturbed data
        """
        t, l_y, l_x = cls.preprocess(out_domain, t)
        alpha = cls.low_prob(l_x * l_y, eps, b)
        return cls.post_process(alpha, b, out_domain, t)

    @classmethod
    def optimized_b_for_avg(cls, l_x, l_y, eps):
        """
        :param l_x: the length of domain in the x-axis
        :param l_y: the length of domain in the y-axis
        :param eps: the privacy budget epsilon
        :return: the optimal side length of the square region
        """
        # Select the side of the square region to minimize the expected MSE.
        # Please refer to Section 4.3.
        coeff_list = np.zeros(6)
        l_x_square = l_x ** 2
        l_y_square = l_y ** 2
        eps_term = exp(eps) - 1
        coeff_list[0] = -4 * l_x_square * l_y_square * (l_x_square + l_y_square)
        coeff_list[2] = 8 * l_x_square * l_y_square
        coeff_list[3] = 5 * l_x * l_y * (l_x + l_y)
        coeff_list[4] = 4 * l_x * l_y * eps_term
        coeff_list[5] = 3 * eps_term * (l_x + l_y)
        coeff_list = np.flip(coeff_list)
        roots = np.roots(coeff_list)
        real_roots = np.compress(np.isreal(roots), roots)
        pos_roots = np.compress(real_roots > 0, real_roots)
        assert len(pos_roots) == 1
        return truncate(np.real(pos_roots[0]), 0, min(l_x, l_y))
