__author__ = "Daeyoung Hong, Woohwan Jung"
__date__ = "2022.02.26."

from data import GowallaData
from ldp2d.utils.others import set_seed


def main():
    set_seed(0)
    GowallaData().preprocess()


if __name__ == '__main__':
    main()
