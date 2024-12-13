import time
import numpy as np
import open3d as o3d
from python.utils import (
    generate_random_permutation,
    generate_random_rigid_transformation,
    center_points_to_origin,
    transform,
    permute,
    add_noise,
)
from python.view_data import plot_comparison
from python.get_stats import get_stats
from python.registrars import RANSAC_registrar, ICP_registrar


def icp_animate_main():
    # np.random.seed(26)
    o3d.utility.random.seed(11)
    NOISE_SCALE = 0.01
    VOXEL_SIZE = NOISE_SCALE * 3  # Размер вокселя для понижения разрешения

    # Загрузка исходного облака точек
    X = np.loadtxt(open("../assets/cat.csv", "rb"), delimiter=",")

    N, d = X.shape

    # Генерация случайного жесткого преобразования и перестановки, добавление шума
    _L = np.array(
        [
            [0.4725979, -0.3009115, 0.8283136],
            [0.8283136, 0.4725979, -0.3009115],
            [-0.3009115, 0.8283136, 0.4725979],
        ]
    )
    print(repr(_L))

    _t = -4 * np.ones((d, 1))
    print(_t)

    _P = generate_random_permutation(N=N)

    Y = transform(X, _L, _t)
    Y = add_noise(Y, sigma=NOISE_SCALE)
    X = add_noise(X, sigma=NOISE_SCALE)

    Y = permute(Y, _P)

    # Центрирование облаков точек
    # X = center_points_to_origin(X)
    # Y = center_points_to_origin(Y)

    # Преобразование numpy массивов в облака точек Open3D
    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(X)
    target_pcd.points = o3d.utility.Vector3dVector(Y)

    # ransac_start_time = time.perf_counter()

    # ransac = RANSAC_registrar(source_pcd, target_pcd, voxel_size=VOXEL_SIZE)
    # ransac.register()

    # ransac_time = time.perf_counter() - ransac_start_time
    # result_ransac = ransac.get_registration_result()

    # Уточнение с помощью ICP
    icp_start_time = time.perf_counter()
    icp = ICP_registrar(source_pcd, target_pcd, threshold=10)
    icp.register()
    icp_time = time.perf_counter() - icp_start_time
    result_icp = icp.get_registration_result()

    # Применение найденного преобразования к исходному облаку точек
    result_pcd = source_pcd.transform(result_icp.transformation)
    Z = np.asarray(result_pcd.points)

    # show_voxel_centers(result_pcd, VOXEL_SIZE)
    stats = get_stats(result_pcd.compute_point_cloud_distance(target_pcd))
    print(f"Stats: {stats}")
    print(f"RMSE: {result_icp.inlier_rmse}, Fitness: {result_icp.fitness}")

    print(f"Transformation matrix:\n{result_icp.transformation}")
    print(f"ICP time: {icp_time}")

    matrix = np.zeros((N, N))
    for i, j in result_icp.correspondence_set:
        matrix[j][i] = 1
    print(np.sum(np.abs(matrix - _P)))
    # Визуализация данных
    plot_comparison(X, Y, Z)


if __name__ == '__main__':
    icp_animate_main()
