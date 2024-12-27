from matplotlib import pyplot as plt
import plotly.graph_objects as go


def plot_comparison(X, Y=None, Z=None, *args):

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
            marker=dict(size=2, opacity=0.2, color='gray'),
        )
    )

    if Y is not None:
        # Отображение точек в пространстве
        fig.add_trace(
            go.Scatter3d(
                x=Y[:, 0],
                y=Y[:, 1],
                z=Y[:, 2],
                name='Noisy data',
                mode='markers',
                marker=dict(size=2, opacity=0.2, color='green'),
            )
        )

    if Z is not None:
        # Отображение точек в пространстве
        fig.add_trace(
            go.Scatter3d(
                x=Z[:, 0],
                y=Z[:, 1],
                z=Z[:, 2],
                name='Solution',
                mode='markers',
                marker=dict(size=2, opacity=0.2, color='crimson'),
            )
        )

    # Настройки графика
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
    )

    # Отображение графика
    fig.show(rerenderer='html')


def show_estimation_method_plot(df, title=""):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(df.index, df["rmse"], label="rmse", color="red", marker=".")

    ax2.plot(
        df.index,
        df["rotation_distance"],
        label="rotation_distance",
        color="green",
        marker=".",
    )
    ax3.plot(
        df.index,
        df["translation_distance"],
        label="translation_distance",
        color="blue",
        marker=".",
    )
    fig.legend()
    fig.suptitle(title)
    plt.savefig(title + '.png')
    plt.show()
