#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
from ABC import ArtificialBeeColony
import numpy as np
from benchmark import *
import matplotlib.pyplot as plt
from tqdm import trange

#--------------------------------------------------------------------------------
# Global variables and settings
#--------------------------------------------------------------------------------
N_BEES              = 100
LIMIT               = 'default'
MAX_ITERS           = 500
BENCHMARK_FUNCTIONS = [Sphere30d,Rosenbrock30d,Ackley30d,Rastrigin30d,Griewank30d,Schwefel30d,Sumsquares30d]
SELECTION           = 'RouletteWheel'
random_seed         = 1234
IMG_PATH            = 'images/AnalysisMutation/'
N_SIMULATIONS       = 10


#--------------------------------------------------------------------------------
# Main
#--------------------------------------------------------------------------------

if __name__ == '__main__':
    
    for function in BENCHMARK_FUNCTIONS:
        
        print('\n')
        print('-'*100)
        print(f"Function: {function.name.upper()}")
        print('-'*100)
        
        # Simulations with Standard ABC
        fitness_history_StandardABC = np.full((N_SIMULATIONS,MAX_ITERS+1),np.nan)
        np.random.seed(1234)
        for s in trange(N_SIMULATIONS,desc='Simulations (Standard ABC):'):
            
            ABC = ArtificialBeeColony(n_bees = N_BEES,
                                    bounds   = function.bounds,
                                    function = function.fun)
            ABC.optimize(max_iters=MAX_ITERS,selection='RouletteWheel',mutation='StandardABC',random_seed=None)

            fitness_history_StandardABC[s,:] = [best_bee.value for best_bee in ABC.optimal_bee_history]
            
        means_StandardABC   = np.mean(fitness_history_StandardABC,axis=0)
        # stds_StandardABC    = np.std(fitness_history_StandardABC,axis=0)
        # lower_StandardABC   = means_StandardABC - 1.96* stds_StandardABC/np.sqrt(N_SIMULATIONS)
        # upper_StandardABC   = means_StandardABC + 1.96* stds_StandardABC/np.sqrt(N_SIMULATIONS)
        
        # Simulations with Modified ABC
        fitness_history_ModifiedABC = np.full((N_SIMULATIONS,MAX_ITERS+1),np.nan)
        np.random.seed(1234)

        for s in trange(N_SIMULATIONS,desc='Simulations (Modified ABC)'):
            
            ABC = ArtificialBeeColony(n_bees = N_BEES,
                                    bounds   = function.bounds,
                                    function = function.fun)
            ABC.optimize(max_iters=MAX_ITERS,selection='RouletteWheel',mutation='ModifiedABC',random_seed=None)

            fitness_history_ModifiedABC[s,:] = [best_bee.value for best_bee in ABC.optimal_bee_history]
            
        means_ModifiedABC  = np.mean(fitness_history_ModifiedABC,axis=0)
        # stds_ModifiedABC   = np.std(fitness_history_ModifiedABC,axis=0)
        # lower_ModifiedABC  = means_ModifiedABC - 1.96* stds_ModifiedABC/np.sqrt(N_SIMULATIONS)
        # upper_ModifiedABC  = means_ModifiedABC + 1.96* stds_ModifiedABC/np.sqrt(N_SIMULATIONS)
        
        # Simulations with ABC/best/1
        fitness_history_ABCbest1 = np.full((N_SIMULATIONS,MAX_ITERS+1),np.nan)
        np.random.seed(1234)
        for s in trange(N_SIMULATIONS,desc='Simulations (ABC with DE/best/1):'):
            
            ABC = ArtificialBeeColony(n_bees = N_BEES,
                                    bounds   = function.bounds,
                                    function = function.fun)
            ABC.optimize(max_iters=MAX_ITERS,selection='RouletteWheel',mutation='ABC/best/1',random_seed=None)

            fitness_history_ABCbest1[s,:] = [best_bee.value for best_bee in ABC.optimal_bee_history]
            
        means_ABCbest1   = np.mean(fitness_history_ABCbest1,axis=0)
        # stds_ABCbest1    = np.std(fitness_history_ABCbest1,axis=0)
        # lower_ABCbest1   = means_ABCbest1 - 1.96* stds_ABCbest1/np.sqrt(N_SIMULATIONS)
        # upper_ABCbest1   = means_ABCbest1 + 1.96* stds_ABCbest1/np.sqrt(N_SIMULATIONS)
        
        # Simulations with ABC/best/2
        fitness_history_ABCbest2 = np.full((N_SIMULATIONS,MAX_ITERS+1),np.nan)
        np.random.seed(1234)

        for s in trange(N_SIMULATIONS,desc='Simulations (ABC with DE/best/2)'):
            
            ABC = ArtificialBeeColony(n_bees = N_BEES,
                                    bounds   = function.bounds,
                                    function = function.fun)
            ABC.optimize(max_iters=MAX_ITERS,selection='RouletteWheel',mutation='ABC/best/2',random_seed=None)

            fitness_history_ABCbest2[s,:] = [best_bee.value for best_bee in ABC.optimal_bee_history]
            
        means_ABCbest2   = np.mean(fitness_history_ABCbest2,axis=0)
        # stds_ABCbest2    = np.std(fitness_history_ABCbest2,axis=0)
        # lower_ABCbest2   = means_ABCbest2 - 1.96* stds_ABCbest2/np.sqrt(N_SIMULATIONS)
        # upper_ABCbest2   = means_ABCbest2 + 1.96* stds_ABCbest2/np.sqrt(N_SIMULATIONS)
    
        
        # Plotting the probabilistic series
        plt.figure(figsize=(10, 6))
        x = np.arange(MAX_ITERS+1)
        plt.plot(x, np.clip(means_StandardABC,a_min=10e-16,a_max=None), label="Standard ABC", color="#2E86C1")
        plt.plot(x, np.clip(means_ModifiedABC,a_min=10e-16,a_max=None), label="Modified ABC", color="#8E44AD")
        plt.plot(x, np.clip(means_ABCbest1,a_min=10e-16,a_max=None), label="ABC/best/1", color="#E74C3C")
        plt.plot(x, np.clip(means_ABCbest2,a_min=10e-16,a_max=None), label="ABC/best/2", color="#F1C40F")
        plt.title(f"{function.name.upper()}: log(cost) function over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("log(Cost function)")
        plt.legend()
        plt.grid(True)
        plt.yscale("log")
        
        plt.savefig(f"{IMG_PATH}{function.name}_mutations_comparison.png",dpi=200)