#++++++++++++++++++++++++++++++++++++
# Libraries and modules
#++++++++++++++++++++++++++++++++++++

import numpy as np
import pandas as pd
from tqdm import trange
from beeoptimal import ArtificialBeeColony
from beeoptimal.benchmarks import *


#++++++++++++++++++++++++++++++++++++
# Global variables and settings
#++++++++++++++++++++++++++++++++++++

N_BEES              = 100
LIMIT               = 'default'
MAX_ITERS           = 2         # Note: we are interested in the diversity at initialization
BENCHMARK_FUNCTIONS = [Sphere2d,Rosenbrock2d,Ackley2d,Rastrigin2d,Weierstrass2d,Griewank2d,Schwefel2d,Sumsquares2d,
                       Sphere10d,Rosenbrock10d,Ackley10d,Rastrigin10d,Weierstrass10d,Griewank10d,Schwefel10d,Sumsquares10d,
                       Sphere30d,Rosenbrock30d,Ackley30d,Rastrigin30d,Weierstrass30d,Griewank30d,Schwefel30d,Sumsquares30d]
SELECTION           = 'RouletteWheel'
MUTATION            = 'StandardABC'
INITIALIZATIONS     = ['random','cahotic']
RANDOM_SEED         = 1234
N_SIMULATIONS       = 15
CSV_PATH            = 'simulations/diversity_report.csv'

#++++++++++++++++++++++++++++++++++++
# Main
#++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':
    
    # Fix seed for reproducibility
    np.random.seed(RANDOM_SEED)   
    
    diversity_df  = pd.DataFrame(columns=['Function','Initialization','Diversity'])
    
    for function in BENCHMARK_FUNCTIONS:
        
        print('\n')
        print('-'*100)
        print(f"Function: {function.name.upper()}")
        print('-'*100)
        
        for i,initialization in enumerate(INITIALIZATIONS):
            # Simulations
            for s in trange(N_SIMULATIONS,desc=f'Simulations (Initialization={initialization})'):

                ABC = ArtificialBeeColony(
                    n_bees   = N_BEES,
                    bounds   = function.bounds,
                    function = function.fun
                    )
                
                ABC.optimize(
                    max_iters     = MAX_ITERS,
                    limit          = LIMIT,
                    selection      = SELECTION,
                    mutation       = MUTATION,
                    initialization = initialization,
                    random_seed    = None
                    )
                
                initial_positions = np.array([bee.position for bee in ABC.colony_history[0]]) # n_employs x n_dimensions
                diversity = np.mean(np.sqrt(np.mean((initial_positions - initial_positions.mean(axis=0))**2,axis=1)))
                
                diversity_df.loc[len(diversity_df)] = [function.name,initialization,diversity]
                
                
    diversity_df.to_csv(CSV_PATH,index=False)
                
        
                
                