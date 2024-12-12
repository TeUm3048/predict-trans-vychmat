import numpy as np
import open3d as o3d
from python.utils import read_pcd, random_transform
from python.get_stats import get_stats
from python.registrars import RANSAC_registrar, ICP_registrar
from python.view_data import plot_comparison


def show_voxel_grid(pcd, voxel_size):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    o3d.visualization.draw_geometries([voxel_grid])


if __name__ == '__main__':
    np.random.seed(26)
    o3d.utility.random.seed(11)
    NOISE_SCALE = 0.01
    VOXEL_SIZE = 1

    source_pcd = read_pcd('../assets/cottage.pcd')
    target_pcd = random_transform(source_pcd)
    show_voxel_grid(source_pcd, VOXEL_SIZE)
    show_voxel_grid(target_pcd, VOXEL_SIZE)

    print('RANSAC поехали')
    ransac = RANSAC_registrar(source_pcd, target_pcd, voxel_size=VOXEL_SIZE)
    ransac.register()
    result_ransac = ransac.get_registration_result()

    print('Итеративно клозируем точки')
    icp = ICP_registrar(
        source_pcd,
        target_pcd,
        init_registration=result_ransac,
    )
    icp.register()
    result_icp = icp.get_registration_result()

    # Применение найденного преобразования к исходному облаку точек
    result_pcd = source_pcd.transform(result_icp.transformation)
    Z = np.asarray(result_pcd.points)

    # show_voxel_centers(result_pcd, VOXEL_SIZE)
    stats = get_stats(result_pcd.compute_point_cloud_distance(target_pcd))
    print(f"Stats: {stats}")
    print(f"RMSE: {result_icp.inlier_rmse}, Fitness: {result_icp.fitness}")
    print(f"Transformation matrix:\n{result_icp.transformation}")

    X = np.asarray(source_pcd.points)
    Y = np.asarray(target_pcd.points)
    plot_comparison(X, Y, Z)
