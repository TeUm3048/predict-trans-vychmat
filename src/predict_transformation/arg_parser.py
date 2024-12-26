import click
import numpy as np
from predict_transformation import cloud_generator
from predict_transformation import utils
from predict_transformation.registrars import ICP_registrar, RANSAC_registrar
from predict_transformation import view_data
import open3d
import click_aliases
from copy import deepcopy
import os


def load_cloud(path: str, with_headers: bool = False):
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')

    if path.endswith('.csv'):
        cloud = np.loadtxt(
            open(path, 'rb'),
            delimiter=',',
            skiprows=1 if with_headers else 0,
        )
        cloud = utils.matrix_to_cloud(cloud)
    elif path.endswith('.pcd'):
        cloud = utils.load_pcd(path)
    else:
        raise ValueError('Unknown file format')

    return cloud


def save_cloud(cloud, path: str):
    if isinstance(cloud, np.ndarray) and path.endswith('.csv'):
        np.savetxt(path, cloud, delimiter=',')
    elif isinstance(cloud, open3d.geometry.PointCloud) and path.endswith('.pcd'):
        open3d.io.write_point_cloud(path, cloud)
    elif path.endswith('.csv'):
        np.savetxt(path, utils.cloud_to_matrix(cloud), delimiter=',')
    elif path.endswith('.pcd'):
        open3d.io.write_point_cloud(path, cloud)
    else:
        raise ValueError('Unknown file format')


def estimate_voxel_size(cloud):
    return np.median(cloud.compute_nearest_neighbor_distance()) * 4


@click.group(cls=click_aliases.ClickAliasedGroup)
def cli():
    "CLI tool for point cloud operations"
    pass


@cli.command(
    name='create-cloud',
    aliases=['create-potato'],
    context_settings={'show_default': True},
)
@click.option(
    '--output', default='cloud.csv', help='Path to save the created point cloud'
)
@click.option('--n-points', default=10000, help='Number of points in the cloud')
@click.option(
    '--force-count', default=40, help='Number of forces applied to the cloud'
)
@click.option('--sphere-radius', default=100, help='Radius of the sphere')
@click.option(
    '--sphere-center',
    nargs=3,
    type=float,
    default=(0, 0, 0),
    help='Sphere center coordinates',
)
@click.option('--force-radius', default=250, help='Radius of the force')
@click.option(
    '--seed', default=None, type=int, help='Random seed for reproducibility'
)
def create_cloud(
    output, n_points, force_count, sphere_radius, sphere_center, force_radius, seed
):
    "Create a point cloud."
    cloud = cloud_generator.generate_cloud_matrix(
        n=n_points,
        force_count=force_count,
        sphere_radius=sphere_radius,
        sphere_center=sphere_center,
        force_radius=force_radius,
        seed=seed,
    )
    np.savetxt(output, cloud, delimiter=',')
    click.echo(f'Point cloud saved to {output}')


@cli.command(
    name='predict-transformation',
    aliases=['predict'],
    context_settings={'show_default': True},
)
@click.option(
    '--source', default='source.csv', help='Path to the source point cloud'
)
@click.option(
    '--target', default='target.csv', help='Path to the target point cloud'
)
@click.option(
    '--with-headers',
    is_flag=True,
    default=False,
    help='Indicates if the file has headers',
)
@click.option(
    '--ransac', is_flag=True, default=True, help='Use RANSAC for registration'
)
@click.option('--icp', is_flag=True, default=True, help='Use ICP for registration')
@click.option(
    '--output-cloud',
    default='output.csv',
    help='Path to save the registered point cloud',
)
@click.option(
    '--output-rotate',
    default='rotate.csv',
    help='Path to save the rotation matrix',
)
@click.option(
    '--output-translate',
    default='translate.csv',
    help='Path to save the translation vector',
)
@click.option(
    '--output-correspondence',
    default='correspondence_set.csv',
    help='Path to save the correspondence set',
)
@click.option(
    '--view',
    is_flag=True,
    default=True,
    help='View the point clouds before and after registration',
)
def predict(
    source,
    target,
    with_headers,
    ransac,
    icp,
    output_cloud,
    output_rotate,
    output_translate,
    output_correspondence,
    view,
):
    "Predict transformation between two point clouds."
    try:
        source_pcd = load_cloud(source, with_headers=with_headers)
        target_pcd = load_cloud(target, with_headers=with_headers)
    except FileNotFoundError as e:
        click.echo(e)
        return

    registrar_result = None
    if ransac:
        voxel_size = estimate_voxel_size(source_pcd)
        registrar = RANSAC_registrar(source_pcd, target_pcd, voxel_size=voxel_size)
        registrar.register()
        registrar_result = registrar.get_registration_result()

    if icp:
        registrar = ICP_registrar(
            source_pcd,
            target_pcd,
            threshold=100,
            init_registration=registrar_result,
        )
        registrar.register()
        registrar_result = registrar.get_registration_result()

    transformation = registrar_result.transformation
    correspondence_set = registrar_result.correspondence_set
    rotate = transformation[:3, :3]
    translate = transformation[:3, 3]
    result_pcd = deepcopy(source_pcd)
    result_pcd = result_pcd.transform(transformation)
    save_cloud(result_pcd, output_cloud)
    np.savetxt(output_rotate, rotate, delimiter=',')
    np.savetxt(output_translate, translate, delimiter=',')
    np.savetxt(output_correspondence, correspondence_set, delimiter=',', fmt='%d')
    click.echo(
        f'Registration completed. Results saved to {output_cloud}, {output_rotate}, {output_translate}, and {output_correspondence}'
    )
    click.echo(f'Transformation matrix:\n{transformation}')
    click.echo(
        f'RMSE: {registrar_result.inlier_rmse}, Fitness: {registrar_result.fitness}'
    )

    if view:
        source_points = utils.cloud_to_matrix(source_pcd)
        target_points = utils.cloud_to_matrix(target_pcd)
        result_points = utils.cloud_to_matrix(result_pcd)
        view_data.plot_comparison(source_points, target_points, result_points)


@cli.command(
    name='perturb-cloud',
    aliases=['perturb-potato', 'perturb'],
    context_settings={'show_default': True},
)
@click.option(
    '-i', '--input', default='input.csv', help='Path to the input point cloud'
)
@click.option(
    '-o',
    '--output',
    default='output.csv',
    help='Path to save the perturbed point cloud',
)
@click.option('--noise-scale', default=0.01, help='Scale of the noise')
@click.option(
    '--seed', default=None, type=int, help='Random seed for reproducibility'
)
def perturb_cloud(input, output, noise_scale, seed):
    "Perturb a point cloud."
    try:
        cloud = load_cloud(input)
    except FileNotFoundError as e:
        click.echo(e)
        return

    if seed is not None:
        np.random.seed(seed)

    points = utils.cloud_to_matrix(cloud)
    N, d = points.shape

    L, t = utils.generate_random_rigid_transformation(d)
    target_points = utils.transform(points, L, t)
    target_points = utils.add_noise(target_points, noise_scale)
    save_cloud(target_points, output)
    click.echo(f'Perturbed point cloud saved to {output}')


@cli.command(context_settings={'show_default': True})
@click.option('--input', default='input.csv', help='Path to the input point cloud')
@click.option(
    '--output',
    default='output.csv',
    help='Path to save the downsampled point cloud',
)
@click.option('--voxel-size', default=0.01, help='Size of the voxel')
def downsample_cloud(input, output, voxel_size):
    "Downsample a point cloud."
    try:
        cloud = load_cloud(input)
    except FileNotFoundError as e:
        click.echo(e)
        return
    downsampled_cloud = utils.downsample(cloud, voxel_size)
    save_cloud(downsampled_cloud, output)
    click.echo(f'Downsampled point cloud saved to {output}')


if __name__ == '__main__':
    cli()
