#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
import plotly.graph_objects as go
from PIL import Image
from .utils import get_marker_path
# for matplotlib version
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


#--------------------------------------------------------------------------------
# Plotting functions
#--------------------------------------------------------------------------------
def ContourPlotBee(x,y,Z,bee_colony,title='',optimal_solution=None):
    """
    Create a contour plot with bee markers and an optional optimal solution point.

    Args:
        x (numpy.ndarray)                  : 1D array representing the x-coordinates of the contour grid.
        y (numpy.ndarray)                  : 1D array representing the y-coordinates of the contour grid.
        Z (numpy.ndarray)                  : 2D array representing the values for the contour plot.
        bee_colony (list)                  : List of Bee objects.
        title (str, optional)              : The title of the plot. Defaults to an empty string.
        optimal_solution (tuple, optional) : Optimal solution (x, y) to be marked with a red "X". Defaults to None.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure containing the contour plot with bee markers and, optionally, the optimal solution marker.

    Raises:
        FileNotFoundError : If the "BeeMarker.png" image file is not found.
        ValueError        : If the bee colony is empty.

    Example:
        >>> from beeoptimal.plotting import ContourPlotBee
        >>> from beeoptimal.benchmarks import Sphere2d
        >>> ABC = ArtificialBeeColony(n_bees=100,function=Sphere2d.fun,bounds=Sphere2d.bounds)
        >>> x = np.linspace(function.bounds[0][0], function.bounds[0][1], 100)
        >>> y = np.linspace(function.bounds[1][0], function.bounds[1][1], 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = np.sin(x)[:, None] + np.cos(y)
        >>> points = np.c_[X.ravel(), Y.ravel()]
        >>> Z = np.array([function.evaluate(p) for p in points]).reshape(X.shape)
        >>> bee_colony = ABC.colony_history[0]
        >>> optimal_solution = function.optimal_solution
        >>> ContourPlotBee(x, y, Z, bee_colony)
    """
    bee_marker_path = get_marker_path()
    # ensure colony is not empty
    if not len(bee_colony):
        raise ValueError("Bee colony cannot be empty!")
        
    # Create contour plot
    fig = go.Figure(data=go.Contour(
        z=Z,
        x=x,
        y=y,
        colorscale='Oranges',
        opacity=0.6,
        contours=dict(
            showlabels=False,
            labelfont=dict(
                size=12,
                color='white'
            )
        )
    ))
    fig.update_layout(
        title=title,
        xaxis_title='x1',
        yaxis_title='x2',
        width=600,
        height=600
    )
    
    bee_marker = Image.open(bee_marker_path)
    bee_marker_size = min(x.max()-x.min(), y.max()-y.min()) * 0.05
    
    for bee in bee_colony:
        fig.add_layout_image(
                    dict(
                        source=bee_marker,
                        xref="x",
                        yref="y",
                        xanchor="center",
                        yanchor="middle",
                        x=bee.position[0],
                        y=bee.position[1],
                        sizex= bee_marker_size,
                        sizey=bee_marker_size,
                        sizing="contain",
                        opacity=1,
                        #layer="above"
                    )
                )
    if optimal_solution is not None:
        fig.add_trace(go.Scatter(x=[optimal_solution[0]], y=[optimal_solution[1]], mode='markers', marker=dict(size=10, color='red',symbol='x')))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def ContourPlotBee_matplotlib(x, y, Z, bee_colony, title='', optimal_solution=None):
    """
    Create a contour plot with bee markers and an optional optimal solution point.

    Args:
        x (numpy.ndarray)                  : 1D array representing the x-coordinates of the contour grid.
        y (numpy.ndarray)                  : 1D array representing the y-coordinates of the contour grid.
        Z (numpy.ndarray)                  : 2D array representing the values for the contour plot.
        bee_colony (list)                  : List of Bee objects with a `position` attribute.
        title (str, optional)              : The title of the plot. Defaults to an empty string.
        optimal_solution (tuple, optional) : Optimal solution (x, y) to be marked with a red "X". Defaults to None.

    Returns:
        matplotlib.figure.Figure: A matplotlib figure containing the contour plot with bee markers and, optionally, the optimal solution marker.
        
    Raises:
        ValueError        : If the bee colony is empty.
        
    Example:
        >>> from beeoptimal.plotting import ContourPlotBee_matplotlib
        >>> from beeoptimal.benchmarks import Sphere2d
        >>> ABC = ArtificialBeeColony(n_bees=100,function=Sphere2d.fun,bounds=Sphere2d.bounds)
        >>> x = np.linspace(function.bounds[0][0], function.bounds[0][1], 100)
        >>> y = np.linspace(function.bounds[1][0], function.bounds[1][1], 100)
        >>> X, Y = np.meshgrid(x, y)
        >>> Z = np.sin(x)[:, None] + np.cos(y)
        >>> points = np.c_[X.ravel(), Y.ravel()]
        >>> Z = np.array([function.evaluate(p) for p in points]).reshape(X.shape)
        >>> bee_colony = ABC.colony_history[0]
        >>> optimal_solution = function.optimal_solution
        >>> ContourPlotBee_matplotlib(x, y, Z, bee_colony)
    """
    # Ensure colony is not empty
    if not len(bee_colony):
        raise ValueError("Bee colony cannot be empty!")

    # Load bee marker image
    bee_marker_path = get_marker_path()
    bee_marker = Image.open(bee_marker_path)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create contour plot
    contour = ax.contourf(x, y, Z, levels=10, cmap='Oranges', alpha=0.6)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Function Value', fontsize=12)
    
     # Add bee markers
    for bee in bee_colony:
        bee_x, bee_y = bee.position
        image_box = OffsetImage(bee_marker, zoom=0.04)
        ab = AnnotationBbox(image_box, (bee_x, bee_y), frameon=False,zorder=1)
        ax.add_artist(ab)
    
    # Add optimal solution marker, if provided
    if optimal_solution is not None:
        ax.plot(optimal_solution[0], optimal_solution[1], 'X', markersize=12, label='Optimal Solution',zorder=2,markerfacecolor='#F24E41',markeredgecolor='black')
        ax.legend(loc='upper left')
    
    # Set plot labels and title
    ax.set_title(title)
    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)

    plt.tight_layout()    
    plt.close(fig)
    return fig