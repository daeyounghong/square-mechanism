__author__ = "Daeyoung Hong, Woohwan Jung"
__date__ = "2022.02.26."

import os


class DataPaths:
    """
    Args:
        dir_path: the path of the directory that contains the preprocessed data file
    Attributes:
        dir: the path of the directory that contains the preprocessed data file
        data: the path of the preprocessed data file
    """

    def __init__(self, dir_path):
        self.dir = dir_path
        self.data = os.path.join(dir_path, 'data.pickle')
