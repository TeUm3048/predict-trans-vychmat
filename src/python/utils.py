import numpy as np
import open3d as o3d


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
        input_pcd: o3d.geometry.PointCloud, noise_scale=0.01,
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
