import numpy as np
import open3d as o3d
from predict_transformation.utils import (
    generate_random_permutation,
    generate_random_rigid_transformation,
    transform,
    permute,
    add_noise,
)
from predict_transformation.view_data import plot_comparison
from predict_transformation.get_stats import get_stats
from predict_transformation.registrars import RANSAC_registrar, ICP_registrar


def center_points_to_origin(X: np.matrix):
    mean = np.mean(X, axis=0)
    return X - mean


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
    NOISE_SCALE = 0.01
    VOXEL_SIZE = NOISE_SCALE * 3  # Размер вокселя для понижения разрешения

    # Загрузка исходного облака точек
    X = np.loadtxt(open('../assets/cat.csv', 'rb'), delimiter=',')

    N, d = X.shape

    # Генерация случайного жесткого преобразования и перестановки, добавление шума
    _L, _t = generate_random_rigid_transformation(d=d)
    _P = generate_random_permutation(N=N)

    Y = transform(X, _L, _t)
    Y = add_noise(Y, sigma=NOISE_SCALE)
    X = add_noise(X, sigma=NOISE_SCALE)

    Y = permute(Y, _P)

    # Центрирование облаков точек
    X = center_points_to_origin(X)
    Y = center_points_to_origin(Y)

    # Преобразование numpy массивов в облака точек Open3D
    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(X)
    target_pcd.points = o3d.utility.Vector3dVector(Y)

    ransac = RANSAC_registrar(source_pcd, target_pcd, voxel_size=VOXEL_SIZE)
    ransac.register()
    result_ransac = ransac.get_registration_result()

    icp = ICP_registrar(
        source_pcd,
        target_pcd,
        voxel_size=VOXEL_SIZE,
        init_registration=result_ransac,
    )
    icp.register()
    result_icp = icp.get_registration_result()

    # Применение найденного преобразования к исходному облаку точек
    result_pcd = source_pcd.transform(result_icp.transformation)
    Z = np.asarray(result_pcd.points)

    # show_voxel_centers(result_pcd, VOXEL_SIZE)
    stats = get_stats(result_pcd.compute_point_cloud_distance(target_pcd))
    print(f'Stats: {stats}')
    print(f'RMSE: {result_icp.inlier_rmse}, Fitness: {result_icp.fitness}')

    print(f'Transformation matrix:\n{result_icp.transformation}')

    matrix = np.zeros((N, N))
    for i, j in result_icp.correspondence_set:
        matrix[j][i] = 1
    print(np.sum(np.abs(matrix - _P)))
    # Визуализация данных

    plot_comparison(X, Y, Z)
