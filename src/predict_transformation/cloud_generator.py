import numpy as np
from functools import partial

from . import utils


def count_force_coefficient(
        point: np.ndarray, force_application_point: np.ndarray, force_radius
):
    distance: np.float64 = np.linalg.norm(point - force_application_point, axis=1) ** 4
    with np.errstate(divide='ignore', invalid='ignore'):
        coeff = np.where(distance != 0,
                         2 * np.arctan(force_radius ** 4 / (4**4 * distance))/np.pi,
                         1,
                        )
    return coeff


def force(
        point: np.ndarray,
        force_vector: np.ndarray,
        force_application_point: np.ndarray,
        force_radius: float,
):

    n = point.shape[0] if len(point.shape) >= 2 else 1
    force_vector = np.resize(np.repeat(force_vector, n), (3, n)).T
    coeff = count_force_coefficient(point, force_application_point, force_radius)
    coeff = np.resize(np.tile(coeff, 3), (3, n)).T

    applied = coeff * force_vector
    return point + applied


def generate_cloud_matrix(n: int, force_count: int = 40):
    sphere_radius = 100
    sphere_center = np.zeros(3)
    force_radius = 250

    point_generator = partial(
        utils.generate_uniform_point_on_sphere, sphere_radius, sphere_center,
    )
    points = np.vstack([point_generator() for _ in range(n)])

    for _ in range(force_count):
        force_application_point = points[np.random.randint(0, n)]
        sign = np.random.randint(0, 2, 3) * 2 - 1
        force_vector = np.random.uniform(sphere_radius * 0.1, sphere_radius * 0.3, 3) * sign
        # force_application_point = np.array([0, 0, 100])
        # force_vector = np.array([0, 0, 150])

        points = force(points, force_vector, force_application_point, force_radius)
    return points
