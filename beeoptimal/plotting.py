#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from .utils import get_marker_path
# for matplotlib version
import matplotlib.pyplot as plt
#from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from .benchmarks import BenchmarkFunction


#--------------------------------------------------------------------------------
# Plotting functions
#--------------------------------------------------------------------------------

def contourplot(function, title=None, extended_bounds=None,figsize=(600,600)):
    """
    Plots a 2D benchmark function as a contour plot.
    
    Args:
        function (BenchmarkFunction)            : A benchmark function object.
        title (str)                             : Title of the plot. Defaults to empty string.
        extended_bounds (float or numpy.ndarray): Specifies the extended bounds for the plot.
            
                                                    - If float, bounds are extended by that percentage of the original range.
                                                    - If NumPy array, it directly specifies the new bounds.
                                                    - If None, no extension is applied. Defaults to None.
            
    Returns:
        plotly.graph_objects.Figure: A Plotly figure containing the contour plot of the function.
        
    Raises:
        ValueError: If extended_bounds is not a float or a NumPy array of shape (2, 2).
    """
    assert isinstance(function, BenchmarkFunction), "function must be a BenchmarkFunction object."
    assert len(function.bounds) == 2, "Function must be 2D"
    
    # Determine the bounds
    x_bounds = function.bounds[0,:]
    y_bounds = function.bounds[1,:]
    
    if extended_bounds is not None:
        if isinstance(extended_bounds, float):
            # Extend by a percentage
            x_range = x_bounds[1] - x_bounds[0]
            y_range = y_bounds[1] - y_bounds[0]
            x_bounds = (x_bounds[0] - extended_bounds * x_range, x_bounds[1] + extended_bounds * x_range)
            y_bounds = (y_bounds[0] - extended_bounds * y_range, y_bounds[1] + extended_bounds * y_range)
        elif isinstance(extended_bounds, np.ndarray):
            assert extended_bounds.shape == (2, 2), "extended_bounds must have shape (2, 2)"
            x_bounds = (extended_bounds[0][0], extended_bounds[0][1])
            y_bounds = (extended_bounds[1][0], extended_bounds[1][1])
        else:
            raise ValueError("extend_bounds must be a float or a NumPy array of shape (2, 2)")
    
    # Generate grid
    x = np.linspace(x_bounds[0], x_bounds[1], 100)
    y = np.linspace(y_bounds[0], y_bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    points = np.c_[X.ravel(), Y.ravel()]  
    Z = np.array([function.evaluate(p) for p in points]).reshape(X.shape)
    
    # Create contour plot
    fig = go.Figure(data=go.Contour(
        x=x, y=y, z=Z,
        colorscale='Oranges', opacity=0.6,
        contours=dict(
            showlabels=False,
            labelfont=dict(
                size=12,
                color='white')
        )
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='x1',
        yaxis_title='x2',
        width=figsize[0],
        height=figsize[1]
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    
    return fig



def plot_surface(function,title='',figsize=(600,600)):
    """
    Plots the surface of a 2D benchmark function.
    
    Args:
        function (BenchmarkFunction)            : A benchmark function object.
        title (str)                             : Title of the plot. Defaults to empty string.
        figsize (tuple)                         : Size of the figure. Defaults to (600, 600).
    
    Returns:
        plotly.graph_objects.Figure: A Plotly figure containing the surface plot of the function.
    """
    
    assert len(function.bounds) == 2, "Function must be 2D"
    
    x = np.linspace(function.bounds[0][0], function.bounds[0][1], 100)
    y = np.linspace(function.bounds[1][0], function.bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    points = np.c_[X.ravel(), Y.ravel()]  
    Z = np.array([function.evaluate(p) for p in points]).reshape(X.shape)
    
    fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y, colorscale='Oranges')])
    
    fig.update_layout(
        title=title,
        scene = dict(
            xaxis_title='x1',
            yaxis_title='x2',
            zaxis_title='f(x1,x2)',
        ),
        width=figsize[0],
        height=figsize[1]
    )
    
    return fig
    
    
def contourplot_bees(function,bee_colony,optimal_solution=None,title='',extended_bounds=None,bee_marker_size=None,figsize=(600,600)):
    """
    Create a contour plot with bee markers and an optional optimal solution point.

    Args:
        function (BenchmarkFunction)            : A benchmark function object.
        bee_colony (list)                       : List of Bee objects.
        optimal_solution (numpy.ndarray)        : The optimal solution point. Defaults to None.
        title (str)                             : Title of the plot. Defaults to empty string.
        extended_bounds (float or numpy.ndarray): Specifies the extended bounds for the plot.
        
                                                    - If float, bounds are extended by that percentage of the original range.
                                                    - If NumPy array, it directly specifies the new bounds.
                                                    - If None, no extension is applied. Defaults to None.
            
        bee_marker_size (int or float, optional): Size of the bee markers. Defaults to None.
        figsize (tuple)                         : Size of the figure. Defaults to (600, 600).
        
    Returns:
        plotly.graph_objects.Figure: A Plotly figure containing the contour plot with bee markers and, optionally, the optimal solution marker.

    Raises:
        ValueError        : If the bee colony is empty.
    """
    
    assert isinstance(function, BenchmarkFunction), "function must be a BenchmarkFunction object."
    assert len(function.bounds) == 2, "Function must be 2D"
    if not len(bee_colony):
        raise ValueError("Bee colony cannot be empty!")
        
    fig = contourplot(function,title,extended_bounds=extended_bounds,figsize=figsize)
    
    bee_marker_path = get_marker_path()
    bee_marker = Image.open(bee_marker_path)
    
    if bee_marker_size is None:
        x_range = np.abs(function.bounds[0,1]-function.bounds[0,0])
        y_range = np.abs(function.bounds[1,1]-function.bounds[1,0])
        bee_marker_size = min(x_range,y_range) * 0.05
    
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
                        opacity=1
                    )
                )
    if optimal_solution is not None:
        assert len(optimal_solution) == 2, "optimal_solution must be a 2D point."
        fig.add_trace(go.Scatter(
            x=[optimal_solution[0]],
            y=[optimal_solution[1]],
            mode='markers',
            marker=dict(size=12, color='red', symbol='x'),
            name='Optimal Solution'
        ))
    
    return fig


# def ContourPlotBee(x,y,Z,bee_colony,title='',optimal_solution=None):
#     """
#     Create a contour plot with bee markers and an optional optimal solution point.

#     Args:
#         x (numpy.ndarray)                  : 1D array representing the x-coordinates of the contour grid.
#         y (numpy.ndarray)                  : 1D array representing the y-coordinates of the contour grid.
#         Z (numpy.ndarray)                  : 2D array representing the values for the contour plot.
#         bee_colony (list)                  : List of Bee objects.
#         title (str, optional)              : The title of the plot. Defaults to an empty string.
#         optimal_solution (tuple, optional) : Optimal solution (x, y) to be marked with a red "X". Defaults to None.

#     Returns:
#         plotly.graph_objects.Figure: A Plotly figure containing the contour plot with bee markers and, optionally, the optimal solution marker.

#     Raises:
#         ValueError        : If the bee colony is empty.

#     Example:
#         >>> from beeoptimal.plotting import ContourPlotBee
#         >>> from beeoptimal.benchmarks import Sphere2d
#         >>> ABC = ArtificialBeeColony(n_bees=100,function=Sphere2d.fun,bounds=Sphere2d.bounds)
#         >>> x = np.linspace(function.bounds[0][0], function.bounds[0][1], 100)
#         >>> y = np.linspace(function.bounds[1][0], function.bounds[1][1], 100)
#         >>> X, Y = np.meshgrid(x, y)
#         >>> Z = np.sin(x)[:, None] + np.cos(y)
#         >>> points = np.c_[X.ravel(), Y.ravel()]
#         >>> Z = np.array([function.evaluate(p) for p in points]).reshape(X.shape)
#         >>> bee_colony = ABC.colony_history[0]
#         >>> optimal_solution = function.optimal_solution
#         >>> ContourPlotBee(x, y, Z, bee_colony)
#     """
#     bee_marker_path = get_marker_path()
#     # ensure colony is not empty
#     if not len(bee_colony):
#         raise ValueError("Bee colony cannot be empty!")
        
#     # Create contour plot
#     fig = go.Figure(data=go.Contour(
#         z=Z,
#         x=x,
#         y=y,
#         colorscale='Oranges',
#         opacity=0.6,
#         contours=dict(
#             showlabels=False,
#             labelfont=dict(
#                 size=12,
#                 color='white'
#             )
#         )
#     ))
#     fig.update_layout(
#         title=title,
#         xaxis_title='x1',
#         yaxis_title='x2',
#         width=600,
#         height=600
#     )
    
#     bee_marker = Image.open(bee_marker_path)
#     bee_marker_size = min(x.max()-x.min(), y.max()-y.min()) * 0.05
    
#     for bee in bee_colony:
#         fig.add_layout_image(
#                     dict(
#                         source=bee_marker,
#                         xref="x",
#                         yref="y",
#                         xanchor="center",
#                         yanchor="middle",
#                         x=bee.position[0],
#                         y=bee.position[1],
#                         sizex= bee_marker_size,
#                         sizey=bee_marker_size,
#                         sizing="contain",
#                         opacity=1,
#                         #layer="above"
#                     )
#                 )
#     if optimal_solution is not None:
#         fig.add_trace(go.Scatter(x=[optimal_solution[0]], y=[optimal_solution[1]], mode='markers', marker=dict(size=10, color='red',symbol='x')))
#     fig.update_xaxes(showgrid=False)
#     fig.update_yaxes(showgrid=False)
#     return fig



# def ContourPlotBee_matplotlib(x, y, Z, bee_colony, title='', optimal_solution=None):
#     """
#     Create a contour plot with bee markers and an optional optimal solution point.

#     Args:
#         x (numpy.ndarray)                  : 1D array representing the x-coordinates of the contour grid.
#         y (numpy.ndarray)                  : 1D array representing the y-coordinates of the contour grid.
#         Z (numpy.ndarray)                  : 2D array representing the values for the contour plot.
#         bee_colony (list)                  : List of Bee objects with a `position` attribute.
#         title (str, optional)              : The title of the plot. Defaults to an empty string.
#         optimal_solution (tuple, optional) : Optimal solution (x, y) to be marked with a red "X". Defaults to None.

#     Returns:
#         matplotlib.figure.Figure: A matplotlib figure containing the contour plot with bee markers and, optionally, the optimal solution marker.
        
#     Raises:
#         ValueError        : If the bee colony is empty.
        
#     Example:
#         >>> from beeoptimal.plotting import ContourPlotBee_matplotlib
#         >>> from beeoptimal.benchmarks import Sphere2d
#         >>> ABC = ArtificialBeeColony(n_bees=100,function=Sphere2d.fun,bounds=Sphere2d.bounds)
#         >>> x = np.linspace(function.bounds[0][0], function.bounds[0][1], 100)
#         >>> y = np.linspace(function.bounds[1][0], function.bounds[1][1], 100)
#         >>> X, Y = np.meshgrid(x, y)
#         >>> Z = np.sin(x)[:, None] + np.cos(y)
#         >>> points = np.c_[X.ravel(), Y.ravel()]
#         >>> Z = np.array([function.evaluate(p) for p in points]).reshape(X.shape)
#         >>> bee_colony = ABC.colony_history[0]
#         >>> optimal_solution = function.optimal_solution
#         >>> ContourPlotBee_matplotlib(x, y, Z, bee_colony)
#     """
#     # Ensure colony is not empty
#     if not len(bee_colony):
#         raise ValueError("Bee colony cannot be empty!")

#     # Load bee marker image
#     bee_marker_path = get_marker_path()
#     bee_marker = Image.open(bee_marker_path)
    
#     fig, ax = plt.subplots(figsize=(12, 8))
    
#     # Create contour plot
#     contour = ax.contourf(x, y, Z, levels=10, cmap='Oranges', alpha=0.6)
#     cbar = fig.colorbar(contour, ax=ax)
#     cbar.set_label('Function Value', fontsize=12)
    
#      # Add bee markers
#     for bee in bee_colony:
#         bee_x, bee_y = bee.position
#         image_box = OffsetImage(bee_marker, zoom=0.04)
#         ab = AnnotationBbox(image_box, (bee_x, bee_y), frameon=False,zorder=1)
#         ax.add_artist(ab)
    
#     # Add optimal solution marker, if provided
#     if optimal_solution is not None:
#         ax.plot(optimal_solution[0], optimal_solution[1], 'X', markersize=12, label='Optimal Solution',zorder=2,markerfacecolor='#F24E41',markeredgecolor='black')
#         ax.legend(loc='upper left')
    
#     # Set plot labels and title
#     ax.set_title(title)
#     ax.set_xlabel('x1', fontsize=12)
#     ax.set_ylabel('x2', fontsize=12)

#     plt.tight_layout()    
#     plt.close(fig)
#     return fig