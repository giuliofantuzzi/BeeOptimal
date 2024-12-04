import plotly.graph_objects as go
from PIL import Image
import os

def get_marker_path():
    # Get the directory of the current script (plots.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the full path to marker.png
    marker_path = os.path.join(base_dir, "package_assets", "BeeMarker.png")
    if not os.path.exists(marker_path):
        raise FileNotFoundError(f"Marker file not found at: {marker_path}")
    return marker_path

def ContourPlotBee(x,y,Z,bee_colony,title='',optimal_solution=None):
    marker_path = get_marker_path()
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
    
    bee_marker = Image.open(marker_path)
    bee_marker_size = min(x.max()-x.min(), y.max()-y.min()) * 0.05
    
    #for bee_x,bee_y in bee_colony:
    for bee in bee_colony:
        fig.add_layout_image(
                    dict(
                        source=bee_marker,
                        xref="x",
                        yref="y",
                        xanchor="center",
                        yanchor="middle",
                        x=bee.position[0],#bee_x,
                        y=bee.position[1],#bee_y,
                        sizex= bee_marker_size,
                        sizey=bee_marker_size,
                        sizing="contain",
                        opacity=1,
                        #layer="above"
                    )
                )
    if optimal_solution:
        fig.add_trace(go.Scatter(x=[optimal_solution[0]], y=[optimal_solution[1]], mode='markers', marker=dict(size=10, color='red',symbol='x')))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig