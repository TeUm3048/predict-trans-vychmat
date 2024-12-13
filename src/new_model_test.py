import numpy as np
import open3d as o3d
from python.utils import read_pcd, random_transform
from python.get_stats import get_stats
from python.registrars import RANSAC_registrar, ICP_registrar
from python.view_data import plot_comparison
from copy import deepcopy


if __name__ == '__main__':
    np.random.seed(2407)
    o3d.utility.random.seed(11)
    NOISE_SCALE = 0.01
    VOXEL_SIZE = 13

    source_pcd = read_pcd('../assets/skeleton.ply')
    source_pcd_copy = deepcopy(source_pcd)
    target_pcd = random_transform(source_pcd)

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

    visual_voxel = 10
    voxelized_source = source_pcd_copy.voxel_down_sample(visual_voxel)
    voxelized_target = target_pcd.voxel_down_sample(visual_voxel)
    voxelized_result = result_pcd.voxel_down_sample(visual_voxel)

    X = np.asarray(voxelized_source.points)
    Y = np.asarray(voxelized_target.points)
    Z = np.asarray(voxelized_result.points)
    plot_comparison(X, Y, Z)
