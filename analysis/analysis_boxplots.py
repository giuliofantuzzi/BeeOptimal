#++++++++++++++++++++++++++++++++++++
# Libraries and modules
#++++++++++++++++++++++++++++++++++++

import pandas as pd 
from beeoptimal.benchmarks import *
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

#++++++++++++++++++++++++++++++++++++
# Global variables and functions
#++++++++++++++++++++++++++++++++++++

BENCHMARK_FUNCTIONS =  [Sphere2d,Rosenbrock2d,Ackley2d,Rastrigin2d,Weierstrass2d,Griewank2d,Schwefel2d,Sumsquares2d,
                        Sphere10d,Rosenbrock10d,Ackley10d,Rastrigin10d,Weierstrass10d,Griewank10d,Schwefel10d,Sumsquares10d,
                        Sphere30d,Rosenbrock30d,Ackley30d,Rastrigin30d,Weierstrass30d,Griewank30d,Schwefel30d,Sumsquares30d]

CSV_PATH            = 'simulations/opt_report_full.csv'
BOXPLOTS_PATH       = 'images/opt_boxplots/'

def optimization_boxplot(function,simulations_df,scale='log',title=''):
    """
    Boxplot of the simulation results for the ABC algorithm with different configurations (for a specific function).

    Args:
        function (BenchmarkFunction)      : Benchmark function.
        simulations_df (pandas.DataFrame) : Optimization results of the ABC algorithm.
        scale (str, optional)             : Scale of the values ('log' or 'linear'). Defaults to 'log'.
        title (str, optional)             : Title of the plot. Defaults to ''.

    Returns:
        plotly.graph_objects.Figure : Boxplot of the optimization results.
    """
    plot_df = simulations_df[(simulations_df['Function'] == function.name)].copy()
    #plot_df = simulations_df[(simulations_df['Function'] == function.name) & (simulations_df['Mutation'].isin(['StandardABC','ModifiedABC']))].copy()
    plot_df["Configuration"] = plot_df["Mutation"] + " (" + plot_df["Initialization"] + ")"
    fig = go.Figure()
    for config in plot_df["Configuration"].unique():
        values = plot_df[plot_df["Configuration"] == config]["OptValue"].values
        fig.add_trace(go.Box(x=values, name=config))

    # Update the layout of the plot
    fig.update_layout(
        title=title,
        title_font=dict(size=16, weight='bold'),
        title_x=0.5,
        xaxis_title="Optimum (log scale)",
        xaxis=dict(type=scale,tickformat='.2e',title_font=dict(size=12, weight='normal'),tickfont=dict(size=10,weight='normal')),
        yaxis=dict(tickangle=0,title_font=dict(size=12, weight='normal'),tickfont=dict(size=10,weight='normal')), 
        showlegend=False,           
        width=1500,
        height=600
    )
    color_palette = px.colors.qualitative.Prism

    fig.update_layout(colorway=color_palette,template='ggplot2')
    return fig


if __name__ == '__main__':
    # Load the optimization results
    simulations_df = pd.read_csv(CSV_PATH)
    # Clip the values for a better visualization
    simulations_df['OptValue'] = np.clip(simulations_df['OptValue'],10e-15,None)
    # Create and store the boxplots
    for function in BENCHMARK_FUNCTIONS:
        boxplot = optimization_boxplot(function,simulations_df,scale='linear',title=f'Simulation results for {function.name}')
        boxplot.write_image(BOXPLOTS_PATH+f'{function.name}_boxplot.png',scale=2)
        #boxplot.write_image(BOXPLOTS_PATH+f'{function.name}_Standard_vs_Modified_boxplot.png',scale=2)
        