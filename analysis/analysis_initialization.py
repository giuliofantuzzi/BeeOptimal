#++++++++++++++++++++++++++++++++++++
# Libraries and modules
#++++++++++++++++++++++++++++++++++++

from beeoptimal import ArtificialBeeColony
import numpy as np
from beeoptimal.benchmarks import *
import matplotlib.pyplot as plt
from tqdm import trange

#++++++++++++++++++++++++++++++++++++
# Global variables and settings
#++++++++++++++++++++++++++++++++++++

N_BEES              = 100
LIMIT               = 'default'
MAX_ITERS           = 500
BENCHMARK_FUNCTIONS = [Sphere2d,Rosenbrock2d,Ackley2d,Rastrigin2d,Weierstrass2d,Griewank2d,Schwefel2d,Sumsquares2d,Eggholder,
                       Sphere10d,Rosenbrock10d,Ackley10d,Rastrigin10d,Weierstrass10d,Griewank10d,Schwefel10d,Sumsquares10d,
                       Sphere30d,Rosenbrock30d,Ackley30d,Rastrigin30d,Weierstrass30d,Griewank30d,Schwefel30d,Sumsquares30d]
SELECTION           = 'RouletteWheel'
MUTATION            = 'StandardABC'
INITIALIZATIONS     = ['random','cahotic']
PLOT_COLORS         = ['#2E86C1','#E74C3C']
RANDOM_SEED         = 1234
STAGNATION_TOL      = np.NINF # No stagnation, we want to see the full optimization process
N_SIMULATIONS       = 15
IMG_PATH            = 'images/analysis_initialization/'

#++++++++++++++++++++++++++++++++++++
# Main
#++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':
    
    # Fix seed for reproducibility
    np.random.seed(RANDOM_SEED)   
    
    for function in BENCHMARK_FUNCTIONS:
        
        print('\n')
        print('-'*100)
        print(f"Function: {function.name.upper()}")
        print('-'*100)
        
        cost_history    = np.full((len(INITIALIZATIONS),N_SIMULATIONS,MAX_ITERS+1),np.nan)
        for i,initialization in enumerate(INITIALIZATIONS):
            # Simulations
            for s in trange(N_SIMULATIONS,desc=f'Simulations (Initialization={initialization})'):

                ABC = ArtificialBeeColony(n_bees   = N_BEES,
                                          bounds   = function.bounds,
                                          function = function.fun)
                
                ABC.optimize(max_iters     = MAX_ITERS,
                            limit          = LIMIT,
                            selection      = SELECTION,
                            mutation       = MUTATION,
                            initialization = initialization,
                            stagnation_tol = STAGNATION_TOL, 
                            random_seed    = None)

                cost_history[i,s,:]    = [best_bee.value for best_bee in ABC.optimal_bee_history]
        
        # Compute statistics
        cost_medians = np.median(cost_history,axis=1)
        cost_median = np.clip(cost_medians,a_min=10e-20,a_max=None)
        
        # Comparison plots
        plt.figure(figsize=(10, 6))
        x = np.arange(MAX_ITERS+1)
        for i,initialization in enumerate(INITIALIZATIONS):
            plt.plot(x,cost_medians[i,:],label=f"{initialization.upper()} initialization",color=PLOT_COLORS[i])
        plt.title(f"{function.name.upper()}: log(cost) function over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("log(Cost function)")
        plt.legend(loc='best')
        plt.grid(True)
        plt.yscale("log")
        plt.savefig(f"{IMG_PATH}{function.name}_initialization_analysis.png",dpi=200)
            