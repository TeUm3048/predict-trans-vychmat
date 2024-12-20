import numpy as np
from python import cloud_generator
from python import utils
from python.registrars import ICP_registrar
from python import view_data


def icp(source_pcd, target_pcd):
    registrar = ICP_registrar(source_pcd, target_pcd, threshold=100)
    registrar.register()
    return registrar.get_registration_result()


def calc_transformation_distance(L1, t1, transformation):
    L2 = transformation[:3, :3]
    t2 = transformation[:3, 3].reshape(-1, 1)
    return np.linalg.norm(L1 - L2) + np.linalg.norm(t1 - t2)


def main():
    np.random.seed(111)
    NOISE_SCALE = 0.01
    VOXEL_SIZE = NOISE_SCALE * 3

    X = cloud_generator.generate_cloud_matrix(10000)
    N, d = X.shape

    phi = np.pi / 4

    _L = np.array(
        [
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1],
        ]
    )
    _t = np.zeros((d, 1))

    Y = utils.transform(X, _L, _t)

    Y = utils.random_permutation(Y)

    Y = utils.add_noise(Y, sigma=NOISE_SCALE)
    source_pcd = utils.matrix_to_cloud(X)
    target_pcd = utils.matrix_to_cloud(Y)

    result = icp(source_pcd, target_pcd)

    distance = calc_transformation_distance(_L, _t, result.transformation)

    print(result.inlier_rmse, distance)

    print(result.transformation)
    result_pcd = source_pcd.transform(result.transformation)
    Z = np.asarray(result_pcd.points)
    view_data.plot_comparison(X, Y, Z)


if __name__ == '__main__':
    main()
