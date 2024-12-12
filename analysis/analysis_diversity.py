#++++++++++++++++++++++++++++++++++++
# Libraries and modules
#++++++++++++++++++++++++++++++++++++
import pandas as pd
import plotly.express as px

#++++++++++++++++++++++++++++++++++++
# Global variables and settings
#++++++++++++++++++++++++++++++++++++

BENCHMARK_FUNCTIONS_NAMES = ['Sphere','Rosenbrock','Ackley','Rastrigin','Weierstrass','Griewank','Schwefel','Sumsquares']
IMG_PATH                  = 'images/analysis_diversity/'
DIMS                      = [2,10,30]

#++++++++++++++++++++++++++++++++++++
# Main
##++++++++++++++++++++++++++++++++++++


if __name__=="__main__":
    
    diversity_df = pd.read_csv('simulations/diversity_report.csv')
    
    for function_name in BENCHMARK_FUNCTIONS_NAMES:
        current_functions = [f"{function_name}-{dim}d" for dim in DIMS]
        current_df = diversity_df[diversity_df['Function'].isin(current_functions)]
        
        fig = px.box(current_df, x="Function", y="Diversity",color='Initialization')
        fig.update_layout(
            title='Diversity of the initial population for different dimensions',
            width=800,height=500
            )
        fig.write_image(f'{IMG_PATH}{function_name}.png',scale=3)
        