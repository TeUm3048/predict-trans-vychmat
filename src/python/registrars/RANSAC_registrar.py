import open3d as o3d
from open3d.pipelines import registration


def preprocess_point_cloud(pcd, voxel_size, max_nn_radius=30, max_nn_fpfh=100):
    down_radius = voxel_size * 2
    fpfh_radius = voxel_size * 5

    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=down_radius, max_nn=max_nn_radius
        )
    )
    pcd_fpfh = registration.compute_fpfh_feature(
        pcd_down,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=fpfh_radius, max_nn=max_nn_fpfh
        ),
    )
    return pcd_down, pcd_fpfh


class RANSAC_registrar:
    def __init__(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        source_descriptors=None,
        target_descriopors=None,
        voxel_size=0.03,
        similarity_threshold=0.9,
        distance_threshold=None,
        confidence=0.999,
        iteration=4000000,
        ransac_n=4,
    ):
        self.source_pcd = source_pcd
        self.target_pcd = target_pcd
        # Предварительная обработка облаков точек
        if source_descriptors is None:
            self.source_down, self.source_fpfh = preprocess_point_cloud(
                source_pcd, voxel_size
            )
        else:
            self.source_down = source_pcd
            self.source_fpfh = source_descriptors
        if target_descriopors is None:
            self.target_down, self.target_fpfh = preprocess_point_cloud(
                target_pcd, voxel_size
            )
        else:
            self.target_down = source_pcd
            self.target_fpfh = target_descriopors

        self.voxel_size = voxel_size
        self.similarity_threshold = similarity_threshold
        if distance_threshold is None:
            self.distance_threshold = voxel_size * 1.5
        else:
            self.distance_threshold = distance_threshold

        self.confidence = confidence
        self.iteration = iteration
        self.ransac_n = ransac_n

    def register(self):
        self.result = registration.registration_ransac_based_on_feature_matching(
            self.source_down,
            self.target_down,
            self.source_fpfh,
            self.target_fpfh,
            True,
            self.distance_threshold,
            registration.TransformationEstimationPointToPoint(False),
            self.ransac_n,
            [
                registration.CorrespondenceCheckerBasedOnEdgeLength(
                    self.similarity_threshold
                ),
                registration.CorrespondenceCheckerBasedOnDistance(
                    self.distance_threshold
                ),
            ],
            registration.RANSACConvergenceCriteria(
                self.iteration, self.confidence
            ),
        )

    def get_registration_result(self):
        return self.result
