import numpy as np


def generate_random_permutation(N: int):
    P = np.eye(N)
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


def add_noise(matrix, sigma):
    return matrix + sigma * np.random.randn(*matrix.shape)
