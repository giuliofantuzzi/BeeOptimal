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

N_BEES              = [25,50,100,250]
LIMIT               = 'default'
MAX_ITERS           = 1000
BENCHMARK_FUNCTIONS = [Sphere2d,Rosenbrock2d,Ackley2d,Rastrigin2d,Weierstrass2d,Griewank2d,Schwefel2d,Sumsquares2d,
                       Sphere10d,Rosenbrock10d,Ackley10d,Rastrigin10d,Weierstrass10d,Griewank10d,Schwefel10d,Sumsquares10d,
                       Sphere30d,Rosenbrock30d,Ackley30d,Rastrigin30d,Weierstrass30d,Griewank30d,Schwefel30d,Sumsquares30d]
SELECTION           = 'RouletteWheel'
MUTATION            = 'StandardABC'
INITIALIZATION      = 'random'
STAGNATION_TOL      = np.NINF 
RANDOM_SEED         = 1234
IMG_PATH            = 'images/analysis_nbees/'
N_SIMULATIONS       = 15
PLOT_COLORS         = ['#2E86C1','#E74C3C','#F3C40F','#68C73C']


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

        cost_history    = np.full((len(N_BEES),N_SIMULATIONS,MAX_ITERS+1),np.nan)

        for i,n_bees in enumerate(N_BEES):
            # Simulations
            for s in trange(N_SIMULATIONS,desc=f'Simulations (N_BEES={n_bees})'):

                ABC = ArtificialBeeColony(
                    n_bees   = n_bees,
                    bounds   = function.bounds,
                    function = function.fun
                    )
                
                ABC.optimize(
                    max_iters     = MAX_ITERS,
                    limit          = LIMIT,
                    selection      = SELECTION,
                    mutation       = MUTATION,
                    initialization = INITIALIZATION,
                    stagnation_tol = STAGNATION_TOL,
                    random_seed    = None
                    )

                cost_history[i,s,:]    = [best_bee.value for best_bee in ABC.optimal_bee_history]
        
        # Compute statistics
        cost_medians = np.median(cost_history,axis=1)
        cost_medians = np.clip(cost_medians,a_min=10e-30,a_max=None)
        # cost_means = np.mean(cost_history,axis=1)
        # cost_stds  = np.std(cost_history,axis=1)
        # cost_lower = cost_means - 1.96*cost_stds/np.sqrt(N_SIMULATIONS)
        # cost_upper = cost_means + 1.96*cost_stds/np.sqrt(N_SIMULATIONS)
        
        # Comparison plots
        plt.figure(figsize=(10, 6))
        x = np.arange(MAX_ITERS+1)
        for i,n_bees in enumerate(N_BEES):
            plt.plot(x,cost_medians[i,:],label=f"N_BEES={n_bees}",color=PLOT_COLORS[i])
        plt.title(f"{function.name.upper()}: log(cost) function over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("log(Cost function)")
        plt.legend(loc='best')
        plt.grid(True)
        plt.yscale("log")
        plt.savefig(f"{IMG_PATH}{function.name}_nbees_analysis.png",dpi=200)