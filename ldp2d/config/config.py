__author__ = "Daeyoung Hong, Woohwan Jung"
__date__ = "2022.02.26."

import multiprocessing
import os

from ldp2d.utils.others import load_json

prec_bnd = 1e-12  # the relative error bound for checking whether two `float` values are equal.

re_sanity_bound = 1e-3  # the sanity bound for computing relative errors
num_of_chunks_per_core = 2

dim = 2  # the dimensionality of data

use_full_cpu = False
num_full_cpu = multiprocessing.cpu_count()
if use_full_cpu:
    num_cpus = num_full_cpu
else:
    num_cpus = 4

parallel = False if num_cpus == 1 else True

domain_margin_ratio = 1e-5

print("#(CPUs) = {}".format(num_cpus))


class Config:
    """
    Configuration parsed from JSON file whose path is `config.json`
    Attributes:
        data_dir: the data directory where dataset file and output files are stored.
        eps: the privacy budget
        grid_shape: the grid shape used in the proposed postprocessing for frequency estimation
            based on convex optimization.
    """

    def __init__(self):
        dict_ = load_json('config.json')
        self.data_dir = dict_['data_dir']
        self.eps = dict_['eps']
        self.grid_shape = tuple(dict_['grid_shape'])


config = Config()
data_dir = config.data_dir
local_data_dir = os.path.join(data_dir, 'ldp2d_local')
gowalla_path = os.path.join(data_dir, "loc-gowalla_totalCheckins.txt")  # the path of the original Gowalla file


def processed_data_path():
    """
    :return: the preprocessed data file path.
    """
    return os.path.join(local_data_dir, "processed")
