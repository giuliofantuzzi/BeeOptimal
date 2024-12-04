#++++++++++++++++++++++++++++++++++++
# Libraries and modules
#++++++++++++++++++++++++++++++++++++

from beeoptimal import ArtificialBeeColony
import numpy as np
from beeoptimal.benchmarks import *
from tqdm import trange
import pandas as pd


#++++++++++++++++++++++++++++++++++++
# Global variables and settings
#++++++++++++++++++++++++++++++++++++
N_BEES              = 100
LIMIT               = 'default'
MAX_ITERS           = 500
BENCHMARK_FUNCTIONS = [Sphere2d,Rosenbrock2d,Ackley2d,Rastrigin2d,Griewank2d,Schwefel2d,Sumsquares2d,Eggholder,
                       Sphere30d,Rosenbrock30d,Ackley30d,Rastrigin30d,Griewank30d,Schwefel30d,Sumsquares30d]
SELECTION           = 'RouletteWheel'
MUTATIONS           = ['ModifiedABC','ABC/best/1','ABC/best/2']
INITIALIZATIONS     = ['random','cahotic']
STAGNATION_TOL      = 1e-6
random_seed         = 1234
N_SIMULATIONS       = 10
CSV_PATH            = 'opt_report/report.csv'

#++++++++++++++++++++++++++++++++++++
# Main
#++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':
    
    np.random.seed(random_seed) # Seed for reproducibility
    
    benchmark_df = pd.DataFrame(columns=['Function','Initialization','Mutation','Mean','Median','Std','Best','Worst'])
    for function in BENCHMARK_FUNCTIONS:
        for mutation in MUTATIONS:
            for initialization in INITIALIZATIONS:
                print('\n')
                print('-'*100)
                print(f"Evaluating {function.name.upper()} with {initialization} initialization and {mutation} mutation")
                print('-'*100)
                
                optimum_series = np.full(N_SIMULATIONS,np.nan)
                
                for s in trange(N_SIMULATIONS,desc='Simulations'):
                    
                    ABC = ArtificialBeeColony(n_bees = N_BEES,
                                            bounds   = function.bounds,
                                            function = function.fun)
                    ABC.optimize(max_iters     = MAX_ITERS,
                                selection      = SELECTION,
                                mutation       = mutation,
                                initialization = initialization,
                                stagnation_tol = STAGNATION_TOL,
                                random_seed    = None)

                    optimum_series[s] = ABC.optimal_bee.value
                
                mean_optimum   = np.mean(optimum_series)
                median_optimum = np.median(optimum_series)
                std_optimum    = np.std(optimum_series)
                best_optimum   = np.min(optimum_series)
                worst_optimum  = np.max(optimum_series)
                
                benchmark_df.loc[len(benchmark_df)] = [function.name,initialization,mutation,mean_optimum,median_optimum,std_optimum,best_optimum,worst_optimum]

    benchmark_df.to_csv(CSV_PATH,index=False)
            