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

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500),
    )
    return result

if __name__ == '__main__':
    NOISE_SCALE = 0.05
    VOXEL_SIZE = 0.05  # Размер вокселя для понижения разрешения

    # Загрузка исходного облака точек
    X = np.loadtxt(open("../assets/cat.csv", "rb"), delimiter=",")

    N, d = X.shape

    # Генерация случайного жесткого преобразования и перестановки, добавление шума
    _L, _t = generate_random_rigid_transformation(d=d)
    _P = generate_random_permutation(N=N)

    Y = transform(X, _L, _t)
    Y = add_noise(Y, sigma=NOISE_SCALE)
    Y = permute(Y, _P)

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

    # Уточнение с помощью ICP
    threshold = 10  # Максимальное расстояние для поиска соответствий
    result_icp = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        threshold,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    # Применение найденного преобразования к исходному облаку точек
    Z = np.asarray(source_pcd.transform(result_icp.transformation).points)

    # Вычисление метрики
    metric = np.linalg.norm(Y - Z)
    print(f"Metric: {metric}")
    print(f"Transformation matrix:\n{result_icp.transformation}")

    # Визуализация данных
    plot_comparison(X, Y, Z)
