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
BENCHMARK_FUNCTIONS = [Sphere30d,Rosenbrock30d,Ackley30d,Rastrigin30d,Griewank30d,Schwefel30d,Sumsquares30d]
#BENCHMARK_FUNCTIONS = [Sphere2d,Rosenbrock2d,Ackley2d,Rastrigin2d,Griewank2d,Schwefel2d,Sumsquares2d]
SELECTION           = 'RouletteWheel'
random_seed         = 1234
IMG_PATH            = 'images/AnalysisInitialization/'
N_SIMULATIONS       = 10


#++++++++++++++++++++++++++++++++++++
# Main
#++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':
    
    for function in BENCHMARK_FUNCTIONS:
        
        print('\n')
        print('-'*100)
        print(f"Function: {function.name.upper()}")
        print('-'*100)
        
        # Simulations with Standard Initialization
        fitness_history_StandardInit = np.full((N_SIMULATIONS,MAX_ITERS+1),np.nan)
        np.random.seed(random_seed)
        for s in trange(N_SIMULATIONS,desc='Simulations (Standard Initialization)'):
            
            ABC = ArtificialBeeColony(n_bees   = N_BEES,
                                      bounds   = function.bounds,
                                      function = function.fun)
            ABC.optimize(max_iters      = MAX_ITERS,
                         selection      = SELECTION,
                         mutation       = 'StandardABC',
                         initialization = 'random',
                         random_seed    = None)

            fitness_history_StandardInit[s,:] = [best_bee.value for best_bee in ABC.optimal_bee_history]
            
        means_StandardInit   = np.mean(fitness_history_StandardInit,axis=0)
        
        # Simulations with Cahotic Initialization
        fitness_history_CahoticInit = np.full((N_SIMULATIONS,MAX_ITERS+1),np.nan)
        np.random.seed(random_seed)
        for s in trange(N_SIMULATIONS,desc='Simulations (Cahotic Initialization)'):
            
            ABC = ArtificialBeeColony(n_bees   = N_BEES,
                                      bounds   = function.bounds,
                                      function = function.fun)
            ABC.optimize(max_iters      = MAX_ITERS,
                         selection      = SELECTION,
                         mutation       = 'StandardABC',
                         initialization = 'cahotic',
                         random_seed    = None)

            fitness_history_CahoticInit[s,:] = [best_bee.value for best_bee in ABC.optimal_bee_history]
            
        means_CahoticInit   = np.mean(fitness_history_CahoticInit,axis=0)
        
    
        
        # Plotting the probabilistic series
        plt.figure(figsize=(10, 6))
        x = np.arange(MAX_ITERS+1)
        plt.plot(x, np.clip(means_StandardInit,a_min=10e-16,a_max=None), label="Standard Initialization", color="#2E86C1")
        plt.plot(x, np.clip(means_CahoticInit,a_min=10e-16,a_max=None) , label="Cahotic Initialization", color="#E74C3C")
        plt.title(f"{function.name.upper()}: log(cost) function over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("log(Cost function)")
        plt.legend()
        plt.grid(True)
        plt.yscale("log")
        plt.savefig(f"{IMG_PATH}{function.name}_initializations_comparison.png",dpi=200)