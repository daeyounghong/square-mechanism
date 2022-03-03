__author__ = "Daeyoung Hong, Woohwan Jung"
__date__ = "2022.02.26."

from data import GowallaData
from ldp2d.config.config import config
from ldp2d.exp.numeric.tls import GeospatialDataCollectorLDP


def main():
    # Step 1: load the data
    dataset_obj = GowallaData()

    # Step 2: set the configuration
    params = {
        'dataset': dataset_obj, 'ans_grid_shape': (600, 300), 'eps': config.eps, 'seed': 1, 're_sanity_bound': 0.001,
        'grid_shape': config.grid_shape, 'prob': 'cumu', 'l2_coeff': 0.001, 'optimality_tolerance': 2e-16,
        'constraint_tolerance': 1e-05, 'max_iter': 10000, 'step_tol': 5e-13}
    print(params)

    # Step 3: run the algorithm
    with dataset_obj:
        alg_o = GeospatialDataCollectorLDP(**params)
        res = alg_o.run()

    # Step 4: print the results
    mse = res['mse']
    mre = res['freq_est']['re'].mean()
    print("MSE: {}".format(mse))
    print("MRE: {}".format(mre))


if __name__ == '__main__':
    main()
