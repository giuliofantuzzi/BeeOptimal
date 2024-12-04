#++++++++++++++++++++++++++++++++++++
# Libraries and modules
#++++++++++++++++++++++++++++++++++++

from ABC import ArtificialBeeColony
import numpy as np
from benchmark import *
import matplotlib.pyplot as plt
from tqdm import trange

#++++++++++++++++++++++++++++++++++++
# Global variables and settings
#++++++++++++++++++++++++++++++++++++

N_BEES              = [25,50,100,250,500]
LIMIT               = 'default'
MAX_ITERS           = 500
BENCHMARK_FUNCTIONS = [Sphere30d,Rosenbrock30d,Ackley30d,Rastrigin30d,Griewank30d,Schwefel30d,Sumsquares30d]
SELECTION           = 'RouletteWheel'
MUTATION            = 'StandardABC'
INITIALIZATION      = 'random'
random_seed         = 1234
IMG_PATH            = 'images/AnalysisNbees/'
N_SIMULATIONS       = 10


#++++++++++++++++++++++++++++++++++++
# Main
#++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':
    
    # Fix seed for reproducibility
    np.random.seed(random_seed)     
    
    for function in BENCHMARK_FUNCTIONS:
        print('\n')
        print('-'*100)
        print(f"Function: {function.name.upper()}")
        print('-'*100)

        cost_history    = np.full((len(N_BEES),N_SIMULATIONS,MAX_ITERS+1),np.nan)
        #fitness_history = np.full((len(N_BEES),N_SIMULATIONS,MAX_ITERS+1),np.nan)

        for i,n_bees in enumerate(N_BEES):
            # Simulations
            for s in trange(N_SIMULATIONS,desc=f'Simulations (N_BEES={n_bees})'):

                ABC = ArtificialBeeColony(n_bees   = n_bees,
                                          bounds   = function.bounds,
                                          function = function.fun)
                
                ABC.optimize(max_iters     = MAX_ITERS,
                            limit          = LIMIT,
                            selection      = SELECTION,
                            mutation       = MUTATION,
                            initialization = INITIALIZATION,
                            random_seed    = None)

                cost_history[i,s,:]    = [best_bee.value for best_bee in ABC.optimal_bee_history]
                #fitness_history[i,s,:] = [best_bee.fitness for best_bee in ABC.optimal_bee_history]
        
        # Compute statistics
        cost_means = np.mean(cost_history,axis=1)
        cost_stds  = np.std(cost_history,axis=1)
        cost_lower = cost_means - 1.96*cost_stds/np.sqrt(N_SIMULATIONS)
        cost_upper = cost_means + 1.96*cost_stds/np.sqrt(N_SIMULATIONS)
        
        # fitness_means = np.mean(fitness_history,axis=1)
        # fitness_stds  = np.std(fitness_history,axis=1)
        # fitness_lower = fitness_means - 1.96*fitness_stds/np.sqrt(N_SIMULATIONS)
        # fitness_upper = fitness_means + 1.96*fitness_stds/np.sqrt(N_SIMULATIONS)
        
        # Comparison plots
        plt.figure(figsize=(10, 6))
        x = np.arange(MAX_ITERS+1)
        plt.plot(x, cost_means[0,:], label=f"{N_BEES[0]} BEES", color="#2E86C1")
        plt.plot(x, cost_means[1,:], label=f"{N_BEES[1]} BEES", color="#8E44AD")
        plt.plot(x, cost_means[2,:], label=f"{N_BEES[2]} BEES", color="#E74C3C")
        plt.plot(x, cost_means[3,:], label=f"{N_BEES[3]} BEES", color="#F1C40F")
        plt.plot(x, cost_means[4,:], label=f"{N_BEES[4]} BEES", color="#17A589")
        plt.title(f"{function.name.upper()}: log(cost) function over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("log(Cost function)")
        plt.legend(loc='best')
        plt.grid(True)
        plt.yscale("log")
        plt.savefig(f"{IMG_PATH}{function.name}_Nbees_comparison_cost.png",dpi=200)
        
        # plt.figure(figsize=(10, 6))
        # x = np.arange(MAX_ITERS+1)
        # plt.plot(x, fitness_means[0,:], label=f"{N_BEES[0]} BEES", color="#2E86C1")
        # plt.fill_between(x, fitness_lower[0,:], fitness_upper[0,:], color="#2E86C1", alpha=0.3)
        # plt.plot(x, fitness_means[1,:], label=f"{N_BEES[1]} BEES", color="#8E44AD")
        # plt.fill_between(x, fitness_lower[1,:], fitness_upper[1,:], color="#8E44AD", alpha=0.3)
        # plt.plot(x, fitness_means[2,:], label=f"{N_BEES[2]} BEES", color="#E74C3C")
        # plt.fill_between(x, fitness_lower[2,:], fitness_upper[2,:], color="#E74C3C", alpha=0.3)
        # plt.plot(x, fitness_means[3,:], label=f"{N_BEES[3]} BEES", color="#F1C40F")
        # plt.fill_between(x, fitness_lower[3,:], fitness_upper[3,:], color="#F1C40F", alpha=0.3)
        # plt.plot(x, fitness_means[4,:], label=f"{N_BEES[4]} BEES", color="#17A589")
        # plt.fill_between(x, fitness_lower[4,:], fitness_upper[4,:], color="#17A589", alpha=0.3)
        # plt.title(f"{function.name.upper()}: Fitness over iterations")
        # plt.xlabel("Iteration")
        # plt.ylabel("Fitness")
        # plt.legend(loc='best')
        # plt.grid(True)
        # plt.savefig(f"{IMG_PATH}{function.name}_Nbees_comparison_fitness.png",dpi=200)