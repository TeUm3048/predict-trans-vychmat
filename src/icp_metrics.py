import numpy as np
from python import cloud_generator
from python import utils
from python.registrars import ICP_registrar
from python import view_data


def icp(source_pcd, target_pcd):
    registrar = ICP_registrar(source_pcd, target_pcd, threshold=100)
    registrar.register()
    return registrar.get_registration_result()


def main():
    np.random.seed(111)
    NOISE_SCALE = 0.01
    VOXEL_SIZE = NOISE_SCALE * 3

    X = cloud_generator.generate_cloud_matrix(10000, 40)
    Y = utils.add_noise(X, NOISE_SCALE)
    Y = utils.random_permutation(Y)

    df = utils.estimate_metrics(Y, 5, icp)
    print(df)
    view_data.plot_comparison(Y)
    view_data.show_estimation_method_plot(df, title="ICP")


if __name__ == '__main__':
    main()
