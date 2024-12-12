import open3d as o3d


def preprocess_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    max_nn_radius: float = 30,
    max_nn_fpfh: int = 100,
):
    """
    Preprocesses a point cloud by performing voxel downsampling, estimating normals, and computing FPFH features.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        voxel_size (float): The size of the voxel for downsampling.
        max_nn_radius (float, optional): The maximum nearest neighbor radius for normal estimation. Defaults to 30.
        max_nn_fpfh (int, optional): The maximum number of nearest neighbors for FPFH feature computation. Defaults to 100.

    Returns:
        Tuple[open3d.geometry.PointCloud, open3d.pipelines.registration.Feature]: A tuple containing the downsampled point cloud and the computed FPFH features.
    """
    down_radius = voxel_size * 2
    fpfh_radius = voxel_size * 5

    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=down_radius, max_nn=max_nn_radius
        )
    )
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
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
        source_descriptors: o3d.pipelines.registration.Feature = None,
        target_descriopors: o3d.pipelines.registration.Feature = None,
        voxel_size=0.03,
        similarity_threshold=0.9,
        distance_threshold: float = None,
        confidence=0.999,
        iteration=4000000,
        ransac_n=4,
    ):
        """
        Initialize the RANSAC_registrar object.

        Args:
            source_pcd (open3d.geometry.PointCloud): The source point cloud.
            target_pcd (open3d.geometry.PointCloud): The target point cloud.
            source_descriptors (open3d.pipelines.registration.Feature, optional): The descriptors of the source point cloud. Defaults to None.
            target_descriopors (open3d.pipelines.registration.Feature, optional): The descriptors of the target point cloud. Defaults to None.
            voxel_size (float): The voxel size used for downsampling the point clouds. Defaults to 0.03.
            similarity_threshold (float): The similarity threshold for feature matching. Defaults to 0.9.
            distance_threshold (float, optional): The distance threshold for RANSAC. If None, it is set to voxel_size * 1.5. Defaults to None.
            confidence (float): The confidence value for RANSAC. Value in interval from 0 to 1.0. Defaults to 0.999.
            iteration (int): The number of iterations for RANSAC. Defaults to 4000000.
            ransac_n (int): The number of points used for RANSAC. Defaults to 4.
        """
        self.source_pcd = source_pcd
        self.target_pcd = target_pcd

        # Preprocess point clouds
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
        self.result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            self.source_down,
            self.target_down,
            self.source_fpfh,
            self.target_fpfh,
            True,
            self.distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            self.ransac_n,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    self.similarity_threshold
                ),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    self.distance_threshold
                ),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(
                self.iteration, self.confidence
            ),
        )

    def get_registration_result(self):
        return self.result
