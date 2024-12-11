#++++++++++++++++++++++++++++++++++++
# Libraries and modules
#++++++++++++++++++++++++++++++++++++

from beeoptimal.abc import ArtificialBeeColony
import numpy as np
from beeoptimal.benchmarks import *
from tqdm import trange
import pandas as pd


#++++++++++++++++++++++++++++++++++++
# Global variables and settings
#++++++++++++++++++++++++++++++++++++
N_BEES              = 100
LIMIT               = 'default'
MAX_ITERS           = 1000
BENCHMARK_FUNCTIONS = [Sphere2d,Rosenbrock2d,Ackley2d,Rastrigin2d,Weierstrass2d,Griewank2d,Schwefel2d,Sumsquares2d,
                       Sphere10d,Rosenbrock10d,Ackley10d,Rastrigin10d,Weierstrass10d,Griewank10d,Schwefel10d,Sumsquares10d,
                       Sphere30d,Rosenbrock30d,Ackley30d,Rastrigin30d,Weierstrass30d,Griewank30d,Schwefel30d,Sumsquares30d]
SELECTION           = 'RouletteWheel'
MUTATIONS           = ['ModifiedABC','DirectedABC']#['StandardABC','ModifiedABC','ABC/best/1','ABC/best/2']
INITIALIZATIONS     = ['random','cahotic']                      
MR                  = 0.7
STAGNATION_TOL      = 1e-6#np.NINF#1e-6
RANDOM_SEED         = 12345
N_SIMULATIONS       = 15
FULL_CSV_PATH       = 'simulations/ModDir_full_stagnation.csv'      
STATS_CSV_PATH      = 'simulations/ModDir_stats_stagnation.csv'

#++++++++++++++++++++++++++++++++++++
# Main
#++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':
    
    # Seed for reproducibility
    np.random.seed(RANDOM_SEED) 
    
    opt_report_full_df  = pd.DataFrame(columns=['Function','Initialization','Mutation','OptValue'])
    opt_report_stats_df = pd.DataFrame(columns=['Function','Initialization','Mutation','Mean','Std','Median','Best','Worst'])
    
    for function in BENCHMARK_FUNCTIONS:
        for mutation in MUTATIONS:
            for initialization in INITIALIZATIONS:
                print('\n')
                print('-'*100)
                print(f"Evaluating {function.name.upper()} with {initialization} initialization and {mutation} mutation")
                print('-'*100)
                
                optimum_series = np.full(N_SIMULATIONS,np.nan)
                
                for s in trange(N_SIMULATIONS,desc='Running simulations'):
                    
                    ABC = ArtificialBeeColony(n_bees   = N_BEES,
                                              bounds   = function.bounds,
                                              function = function.fun)
                    ABC.optimize(max_iters      = MAX_ITERS,
                                 selection      = SELECTION,
                                 mutation       = mutation,
                                 initialization = initialization,
                                 mr             = MR,   
                                 stagnation_tol = STAGNATION_TOL,
                                 verbose        = False,
                                 random_seed    = None)
                    
                    optimum_series[s] = ABC.optimal_bee.value
                    
                    opt_report_full_df.loc[len(opt_report_full_df)] = [function.name,initialization,mutation,ABC.optimal_bee.value]
                
                mean   = np.mean(optimum_series)
                std    = np.std(optimum_series)
                median = np.median(optimum_series)
                best   = np.min(optimum_series)
                worst  = np.max(optimum_series)
                opt_report_stats_df.loc[len(opt_report_stats_df)] = [function.name,initialization,mutation,mean,std,median,best,worst]

    opt_report_full_df.to_csv(FULL_CSV_PATH,index=False)
    opt_report_stats_df.to_csv(STATS_CSV_PATH,index=False)
            