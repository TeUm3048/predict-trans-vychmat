import plotly.graph_objects as go


def plot_comparison(X, Y, Z):

    # Создание интерактивного 3D-графика
    fig = go.Figure()

    # Отображение точек в пространстве
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            name='Initial data',
            mode='markers',
            marker=dict(size=1, opacity=0.2, color='gray'),
        )
    )

    # Отображение точек в пространстве
    fig.add_trace(
        go.Scatter3d(
            x=Y[:, 0],
            y=Y[:, 1],
            z=Y[:, 2],
            name='Noisy data',
            mode='markers',
            marker=dict(size=1, opacity=0.2, color='green'),
        )
    )

    # Отображение точек в пространстве
    fig.add_trace(
        go.Scatter3d(
            x=Z[:, 0],
            y=Z[:, 1],
            z=Z[:, 2],
            name='Solution',
            mode='markers',
            marker=dict(size=1, opacity=0.2, color='crimson'),
        )
    )

    # Настройки графика
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        title='Случайные точки на поверхности сферы',
    )

    # Отображение графика
    fig.show(rerenderer='html')
