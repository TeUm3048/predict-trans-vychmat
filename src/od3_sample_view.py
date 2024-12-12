import numpy as np
import open3d as o3d
from python.utils import (
    generate_random_permutation,
    generate_random_rigid_transformation,
    transform,
    permute,
    add_noise,
)
from python.view_data import plot_comparison
from python.get_stats import get_stats


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30
        )
    )
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=100
        ),
    )
    return pcd_down, pcd_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    result = (
        o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9
                ),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold
                ),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1.0),
        )
    )
    return result


def show_voxel_grid(pcd, voxel_size):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

    # Визуализация воксельной сетки
    o3d.visualization.draw_geometries([voxel_grid])


def show_voxel_centers(pcd, voxel_size):
    # Создание воксельной сетки из облака точек
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

    # Извлечение центральных точек вокселей
    voxel_centers = [
        voxel.grid_index * voxel_size for voxel in voxel_grid.get_voxels()
    ]
    voxel_centers = np.array(voxel_centers, dtype=np.float64)

    # Создание облака точек из центральных точек вокселей
    voxel_pcd = o3d.geometry.PointCloud()
    voxel_pcd.points = o3d.utility.Vector3dVector(voxel_centers)

    # Визуализация облака точек, представляющего центры вокселей
    o3d.visualization.draw_geometries([voxel_pcd])


if __name__ == '__main__':
    np.random.seed(26)
    o3d.utility.random.seed(11)
    NOISE_SCALE = 0.001
    VOXEL_SIZE = NOISE_SCALE * 3  # Размер вокселя для понижения разрешения

    # Загрузка исходного облака точек
    X = np.loadtxt(open("../assets/cat.csv", "rb"), delimiter=",")

    N, d = X.shape

    # Генерация случайного жесткого преобразования и перестановки, добавление шума
    _L, _t = generate_random_rigid_transformation(d=d)
    _P = generate_random_permutation(N=N)

    Y = transform(X, _L, _t)
    Y = add_noise(Y, sigma=NOISE_SCALE)
    X = add_noise(X, sigma=NOISE_SCALE)

    Y = permute(Y, _P)

    # Центрирование облаков точек
    # x_mean = np.mean(X, axis=0)
    # y_mean = np.mean(Y, axis=0)
    # X = X - x_mean
    # Y = Y - y_mean

    # Преобразование numpy массивов в облака точек Open3D
    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(X)
    target_pcd.points = o3d.utility.Vector3dVector(Y)

    # Предварительная обработка облаков точек
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, VOXEL_SIZE)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, VOXEL_SIZE)

    # Глобальная регистрация
    result_ransac = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, VOXEL_SIZE
    )
    source_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=VOXEL_SIZE * 2, max_nn=30
        )
    )
    target_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=VOXEL_SIZE * 2, max_nn=30
        )
    )

    # Уточнение с помощью ICP
    threshold = 0.5  # Максимальное расстояние для поиска соответствий
    result_icp = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        threshold,
        result_ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        # criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
        #     max_iteration=1
        # ),
    )

    # Применение найденного преобразования к исходному облаку точек
    result_pcd = source_pcd.transform(result_icp.transformation)
    Z = np.asarray(result_pcd.points)

    # show_voxel_centers(result_pcd, VOXEL_SIZE)
    stats = get_stats(result_pcd.compute_point_cloud_distance(target_pcd))
    print(f"Stats: {stats}")
    print(f"RMSE: {result_icp.inlier_rmse}, Fitness: {result_icp.fitness}")

    print(f"Transformation matrix:\n{result_icp.transformation}")

    matrix = np.zeros((N, N))
    for i, j in result_icp.correspondence_set:
        matrix[j][i] = 1
    print(np.sum(np.abs(matrix - _P)))
    # Визуализация данных
    plot_comparison(X, Y, Z)
