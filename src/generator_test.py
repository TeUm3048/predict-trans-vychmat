import plotly.graph_objects as go
from python.cloud_generator import generate_cloud_matrix


if __name__ == '__main__':
    n = 50000
    matrix = generate_cloud_matrix(n)
    fig = go.Figure()
    marker = dict(size=1, opacity=1, color='crimson')
    x, y, z = matrix.T
    scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=marker)
    fig.add_trace(scatter)
    fig.show(rerenderer='html')
