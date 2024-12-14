#++++++++++++++++++++++++++++++++++++
# Libraries and modules
#++++++++++++++++++++++++++++++++++++

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from .utils import get_marker_path
from .benchmarks import BenchmarkFunction
from .bee import Bee

#++++++++++++++++++++++++++++++++++++
# Plotting functions
#++++++++++++++++++++++++++++++++++++

def contourplot(function, title=None, bounds=None,zoom=1.0,figsize=(600,600)):
    """
    Plots a 2D benchmark function as a contour plot.
    
    Args:
        function (BenchmarkFunction)     : A benchmark function object.
        title (str,optional)             : Title of the plot. Defaults to empty string.
        bounds (numpy.ndarray, optional) : Custom bounds for the plot (different from the default ones). Defaults to None (uses the default bounds).
        zoom (float, optional)           : Zoom factor for the plot. Defaults to 1.0 (no zoom).
        figsize (tuple, optional)        : Size of the figure. Defaults to (600,600).
            
    Returns:
        plotly.graph_objects.Figure: A Plotly figure containing the contour plot of the function.
        
    Raises:
        TypeError     : If `function` is not a `BenchmarkFunction` object.
        TypeError     : If `bounds` is not a NumPy array (when `bounds` is provided).
        ValueError    : If `bounds` does not have shape (2, 2) (when `bounds` is provided).
        ValueError    : If `zoom` is not greater than zero.
    """

    if not isinstance(function, BenchmarkFunction):
        raise TypeError("`function` must be a `BenchmarkFunction` object.")

    if bounds is not None:
        if not isinstance(bounds, np.ndarray):
            raise TypeError("`bounds` must be provided as a NumPy array.")
        if bounds.shape != (2, 2):
            raise ValueError(f"`bounds` must have shape (2, 2), but got {bounds.shape}")
    
    if not (isinstance(zoom, (int, float)) and zoom > 0):
        raise ValueError(f"`zoom` must be greater than zero, but got {zoom}")
    
    # Determine the bounds (use predefined ones if no custom bounds are provided)
    if bounds is not None:
        x_bounds = (bounds[0, 0], bounds[0, 1])
        y_bounds = (bounds[1, 0], bounds[1, 1])
    else:
        x_bounds = (function.bounds[0, 0], function.bounds[0, 1])
        y_bounds = (function.bounds[1, 0], function.bounds[1, 1])

    # Apply zoom factor
    x_range = x_bounds[1] - x_bounds[0]
    y_range = y_bounds[1] - y_bounds[0]
    x_center = (x_bounds[0] + x_bounds[1]) / 2
    y_center = (y_bounds[0] + y_bounds[1]) / 2

    x_bounds = (
        x_center - (x_range / 2) * zoom,
        x_center + (x_range / 2) * zoom
    )
    y_bounds = (
        y_center - (y_range / 2) * zoom,
        y_center + (y_range / 2) * zoom
    )
    
    # Generate grid and make contourplot
    x = np.linspace(x_bounds[0], x_bounds[1], 100)
    y = np.linspace(y_bounds[0], y_bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    points = np.c_[X.ravel(), Y.ravel()]  
    Z = np.array([function.evaluate(p) for p in points]).reshape(X.shape)
    
    fig = go.Figure(
        data=go.Contour(
            x=x, y=y, z=Z,
            colorscale='Oranges', opacity=0.6,
            contours=dict(
                showlabels=False,
                labelfont=dict(size=12,color='white')
                )
            )
        )
    
    fig.update_layout(
        title= str(title),
        xaxis_title='x1',
        yaxis_title='x2',
        width=figsize[0],
        height=figsize[1]
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    
    return fig



def surfaceplot(function,title='',bounds=None,zoom=1.0,figsize=(600,600)):
    """
    Plots the surface of a 2D benchmark function.
    
    Args:
        function (BenchmarkFunction)     : A benchmark function object.
        title (str,optional)             : Title of the plot. Defaults to empty string.
        bounds (numpy.ndarray, optional) : Custom bounds for the plot (different from the default ones). Defaults to None (uses the default bounds).
        zoom (float, optional)           : Zoom factor for the plot. Defaults to 1.0 (no zoom).
        figsize (tuple,optional)         : Size of the figure. Defaults to (600, 600).
    
    Returns:
        plotly.graph_objects.Figure: A Plotly figure containing the surface plot of the function.
        
    Raises:
        TypeError     : If `function` is not a `BenchmarkFunction` object.
        TypeError     : If `bounds` is not a NumPy array (when bounds is provided).
        ValueError    : If `bounds` does not have shape (2, 2) (when `bounds` is provided).
        ValueError    : If `zoom` is not greater than zero.
    """
    # Input checks
    if not isinstance(function, BenchmarkFunction):
        raise TypeError("`function` must be a `BenchmarkFunction` object.")

    if bounds is not None:
        if not isinstance(bounds, np.ndarray):
            raise TypeError("`bounds` must be provided as a NumPy array.")
        if bounds.shape != (2, 2):
            raise ValueError("`bounds` must have shape (2, 2).")
    
    if not (isinstance(zoom, (int, float)) and zoom > 0):
        raise ValueError(f"`zoom` must be greater than zero, but got {zoom}")
    
    # Determine the bounds (use predefined ones if no custom bounds are provided)
    if bounds is not None:
        x_bounds = (bounds[0, 0], bounds[0, 1])
        y_bounds = (bounds[1, 0], bounds[1, 1])
    else:
        x_bounds = (function.bounds[0, 0], function.bounds[0, 1])
        y_bounds = (function.bounds[1, 0], function.bounds[1, 1])

    # Apply zoom factor
    x_range = x_bounds[1] - x_bounds[0]
    y_range = y_bounds[1] - y_bounds[0]
    x_center = (x_bounds[0] + x_bounds[1]) / 2
    y_center = (y_bounds[0] + y_bounds[1]) / 2

    x_bounds = (
        x_center - (x_range / 2) * zoom,
        x_center + (x_range / 2) * zoom
    )
    y_bounds = (
        y_center - (y_range / 2) * zoom,
        y_center + (y_range / 2) * zoom
    )
    
    # Generate grid and make surfaceplot 
    x = np.linspace(x_bounds[0], x_bounds[1], 100)
    y = np.linspace(y_bounds[0], y_bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    points = np.c_[X.ravel(), Y.ravel()]  
    Z = np.array([function.evaluate(p) for p in points]).reshape(X.shape)
    
    fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y, colorscale='Oranges')])
    
    fig.update_layout(
        title = str(title),
        scene = dict(
            xaxis_title='x1',
            yaxis_title='x2',
            zaxis_title='f(x1,x2)',
        ),
        width=figsize[0],
        height=figsize[1]
    )
    
    return fig
    
    
def contourplot_bees(function,bee_colony,optimal_solution=None,title='',bounds=None,zoom=1.0,bee_marker_size=None,figsize=(600,600)):
    """
    Create a contour plot with bee markers and an optional optimal solution point.

    Args:
        function (BenchmarkFunction)            : A benchmark function object.
        bee_colony (list of Bee)                : List of Bee objects.
        optimal_solution (numpy.ndarray)        : The optimal solution point. Defaults to None.
        title (str)                             : Title of the plot. Defaults to empty string.
        bounds (numpy.ndarray, optional)        : Custom bounds for the plot (different from the default ones). Defaults to None (uses the default bounds).
        zoom (float, optional)                  : Zoom factor for the plot. Defaults to 1.0 (no zoom).
        bee_marker_size (int or float, optional): Size of the bee markers. Defaults to None.
        figsize (tuple)                         : Size of the figure. Defaults to (600, 600).
        
    Returns:
        plotly.graph_objects.Figure: A Plotly figure containing the contour plot with bee markers and, optionally, the optimal solution marker.

    Raises:
        TypeError     : If `bee_colony` is not a list
        ValueError    : If `bee_colony` is empty
        TypeError     : If elements of `bee_colony` are not `Bee` objects.
        TypeError     : If `optimal_solution` is not a NumPy array (when `optimal_solution` is provided).
        ValueError    : If `optimal_solution` is not a 2D point (when `optimal_solution` is provided).
    """
    # Input checks
    if not isinstance(bee_colony, list):
        raise TypeError("`bee_colony` must be a list.")
    if not len(bee_colony):
        raise ValueError("Bee colony cannot be empty!")
    if not isinstance(bee_colony[0], Bee):
        raise TypeError("Elements of `bee_colony` must be Bee objects.")
    
    fig = contourplot(function,title,bounds=bounds,zoom=zoom,figsize=figsize)
    
    # Bee marker settings
    bee_marker_path = get_marker_path()
    bee_marker = Image.open(bee_marker_path)
    if bee_marker_size is None:
        x_range = np.abs(function.bounds[0,1]-function.bounds[0,0])
        y_range = np.abs(function.bounds[1,1]-function.bounds[1,0])
        bee_marker_size = min(x_range,y_range) * 0.05
    
    # Add bees
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
    # Add optimal solution (if specified)
    if optimal_solution is not None:
        if not isinstance(optimal_solution,np.ndarray):
            raise TypeError("optimal_solution must be a NumPy array.")
        optimal_solution = optimal_solution.reshape(1,-1)
        if optimal_solution.shape != (1,2):
            raise ValueError("optimal_solution must be a 2D point.")
        fig.add_trace(go.Scatter(
            x=[optimal_solution[0,0]],
            y=[optimal_solution[0,1]],
            mode='markers',
            marker=dict(size=12, color='red', symbol='x'),
            name='Optimal Solution')
        )
    
    return fig