import open3d as o3d


class ICP_registrar:
    def __init__(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        voxel_size=0.03,
        threshold=0.5,
        init_registration=None,
        iteration=30,
        relative_fitness=1e-06,
        relative_rmse=1e-06,
    ):
        self.source_pcd = source_pcd
        self.target_pcd = target_pcd
        self.voxel_size = voxel_size
        if init_registration is None:
            self.result = o3d.pipelines.registration.RegistrationResult()
        else:
            self.result = init_registration
        self.threshold = threshold
        self.iteration = iteration
        self.relative_fitness = relative_fitness
        self.relative_rmse = relative_rmse

    def register(self):
        # Уточнение с помощью ICP
        self.result = o3d.pipelines.registration.registration_icp(
            self.source_pcd,
            self.target_pcd,
            self.threshold,
            self.result.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.iteration,
                relative_fitness=self.relative_fitness,
                relative_rmse=self.relative_rmse,
            ),
        )

    def get_registration_result(self):
        return self.result
