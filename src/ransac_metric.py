import numpy as np
from python import cloud_generator
from python import utils
from python.registrars import RANSAC_registrar
from python import view_data
from od3_sample_view import show_voxel_centers


def ransac(source_pcd, target_pcd):
    registrar = RANSAC_registrar(source_pcd, target_pcd, voxel_size=1)
    registrar.register()
    return registrar.get_registration_result()


def main():
    np.random.seed(23)
    NOISE_SCALE = 5
    VOXEL_SIZE = NOISE_SCALE * 3

    X = cloud_generator.generate_cloud_matrix(10000, 40)
    Y = utils.add_noise(X, NOISE_SCALE)
    Y = utils.random_permutation(Y)

    df = utils.estimate_metrics(Y, 5, ransac)
    print(df)
    view_data.plot_comparison(Y)
    # target_pcd = utils.matrix_to_cloud(Y)
    # show_voxel_centers(target_pcd, 1)
    view_data.show_estimation_method_plot(df, title="RANSAC")


if __name__ == '__main__':
    main()
