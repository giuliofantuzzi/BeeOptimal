import plotly.graph_objects as go
from PIL import Image

def ContourPlotBee(x,y,Z,bee_colony,title='',marker_path=None,optimal_solution=None):
    if not marker_path:
        raise ValueError('Please provide a path to the marker image')
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
        width=700,
        height=700,
    )
    
    bee_marker = Image.open(marker_path)
    
    for bee_x,bee_y in bee_colony:
        fig.add_layout_image(
                    dict(
                        source=bee_marker,
                        xref="x",
                        yref="y",
                        xanchor="center",
                        yanchor="middle",
                        x=bee_x,
                        y=bee_y,
                        sizex=min(x.max()-x.min(), y.max()-y.min()) * 0.05,#0.5,
                        sizey=min(x.max()-x.min(), y.max()-y.min()) * 0.05,#0.5,
                        sizing="contain",
                        opacity=1,
                        #layer="above"
                    )
                )
    if optimal_solution:
        fig.add_trace(go.Scatter(x=[optimal_solution[0]], y=[optimal_solution[1]], mode='markers', marker=dict(size=10, color='red'), name='Optimal solution'))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig