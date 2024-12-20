import numpy as np
import open3d as o3d
import pandas as pd


def generate_random_permutation(N: int):
    P = np.eye(N, dtype='int8')
    np.random.shuffle(P)
    return P


def generate_random_rigid_transformation(d: int):
    random_matrix = np.random.random((d, d))
    L, _ = np.linalg.qr(random_matrix)
    t = 6 * np.random.random((d, 1))
    return L, t


def transform(matrix, L, t):
    N = matrix.shape[0]
    return (np.dot(L, matrix.T) + np.dot(t, np.ones((1, N)))).T


def permute(matrix, P):
    return np.dot(P, matrix)


def random_permutation(matrix):
    return np.random.permutation(matrix)


def add_noise(matrix, sigma):
    return matrix + sigma * np.random.randn(*matrix.shape)


def read_pcd(filename: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(filename)
    return pcd


def matrix_to_cloud(matrix: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(matrix)
    return pcd


def cloud_to_matrix(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    return np.asarray(pcd.points)


def random_transform(
    input_pcd: o3d.geometry.PointCloud,
    noise_scale=0.01,
) -> o3d.geometry.PointCloud:
    X = np.asarray(input_pcd.points)
    N, d = X.shape

    _L, _t = generate_random_rigid_transformation(d=d)

    Y = transform(X, _L, _t)
    Y = add_noise(Y, sigma=noise_scale)

    Y = random_permutation(Y)
    return matrix_to_cloud(Y)


def generate_uniform_point_on_sphere(radius, offset):

    # Генерация трех случайных значений из нормального распределения
    point = np.random.normal(size=3)

    # Рассчет длины вектора
    magnitude = np.linalg.norm(point)

    # Нормализация вектора, если длина не равна нулю
    while magnitude == 0:
        point = np.random.normal(size=3)

        # Рассчет длины вектора
        magnitude = np.linalg.norm(point)

    point /= magnitude

    # Масштабирование вектора до указанного радиуса
    point *= radius

    # Прибавление смещения к каждой координате
    point += offset

    return point


def calc_transformation_distance(L1, t1, transformation):
    L2 = transformation[:3, :3]
    t2 = transformation[:3, 3].reshape(-1, 1)
    return np.linalg.norm(L1 - L2), np.linalg.norm(t1 - t2)


def estimate_metrics_by_angle(X, phi, method):
    _L, _t, source_pcd, target_pcd = generate_transformation_by_angle(X, phi)

    result = method(source_pcd, target_pcd)

    distance = calc_transformation_distance(_L, _t, result.transformation)

    return result.inlier_rmse, result.fitness, *distance


def generate_transformation_by_angle(X, phi, noise_scale=0, permutation=False):
    N, d = X.shape

    _L = np.array(
        [
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1],
        ]
    )
    _t = np.zeros((d, 1))

    Y = transform(X, _L, _t)

    if permutation:
        Y = random_permutation(Y)

    Y = add_noise(Y, sigma=noise_scale)
    source_pcd = matrix_to_cloud(X)
    target_pcd = matrix_to_cloud(Y)
    return _L, _t, source_pcd, target_pcd


def estimate_metrics(X, n, method):
    df = pd.DataFrame(
        columns=[
            "rmse",
            "inlier_fitness",
            "rotation_distance",
            "translation_distance",
        ]
    )
    step = 2 * np.pi / 8 / (n + 1)
    start = -np.pi / 4 + step
    end = np.pi / 4
    for phi in np.arange(start, end, step):
        estimation = estimate_metrics_by_angle(X, phi, method)
        df.loc[np.rad2deg(phi)] = estimation
    return df
