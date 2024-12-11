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
MAX_ITERS           = 1000
BENCHMARK_FUNCTIONS =  [Sphere2d,Rosenbrock2d,Ackley2d,Rastrigin2d,Weierstrass2d,Griewank2d,Schwefel2d,Sumsquares2d,
                       Sphere10d,Rosenbrock10d,Ackley10d,Rastrigin10d,Weierstrass10d,Griewank10d,Schwefel10d,Sumsquares10d,
                       Sphere30d,Rosenbrock30d,Ackley30d,Rastrigin30d,Weierstrass30d,Griewank30d,Schwefel30d,Sumsquares30d]
SELECTION           = 'RouletteWheel'
MUTATIONS           = ['StandardABC','ModifiedABC','ABC/best/1','ABC/best/2','DirectedABC']
INITIALIZATIONS     = ['random','cahotic']                      
MR                  = 0.7
STAGNATION_TOL      = 1e-7
RANDOM_SEED         = 12345
N_SIMULATIONS       = 10
PLOT_COLORS         = ['#2E86C1','#55DDF9','#14521A','#68C73C','#BE291D','#FA9B94','#C28D00','#FADA48','#590C6A','#E896E8'] 
IMG_PATH            = 'images/analysis_mutation/'

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
        
        cost_history    = np.full((len(MUTATIONS),len(INITIALIZATIONS),N_SIMULATIONS,MAX_ITERS+1),np.nan)
        
        for m,mutation in enumerate(MUTATIONS):
            for i,initialization in enumerate(INITIALIZATIONS):
                # Simulations
                for s in trange(N_SIMULATIONS,desc=f'Simulations (Mutation={mutation}, Initialization={initialization})'):

                    ABC = ArtificialBeeColony(n_bees   = N_BEES,
                                              bounds   = function.bounds,
                                              function = function.fun)

                    ABC.optimize(max_iters     = MAX_ITERS,
                                limit          = LIMIT,
                                selection      = SELECTION,
                                mutation       = mutation,
                                initialization = initialization,
                                stagnation_tol = STAGNATION_TOL,
                                mr             = MR,
                                verbose=False)

                    optimal_history = [best_bee.value for best_bee in ABC.optimal_bee_history]
                    # Padding in case of stagnation
                    optimal_history += [optimal_history[-1]] * (MAX_ITERS + 1 - len(optimal_history))
                    cost_history[m,i,s,:] = optimal_history
                    
        cost_medians = np.median(cost_history,axis=2)
        cost_medians = np.clip(cost_medians,a_min=10e-30,a_max=None)
        
        plt.figure(figsize=(10, 7))
        x = np.arange(MAX_ITERS+1)
        for m,mutation in enumerate(MUTATIONS):
            plt.plot(x,cost_medians[m,0,:],'-',label=f"{mutation} (random)",color=PLOT_COLORS[m*len(INITIALIZATIONS)])
            plt.plot(x,cost_medians[m,1,:],'--',label=f"{mutation} (cahotic)",color=PLOT_COLORS[m*len(INITIALIZATIONS)+1])
            
        plt.title(f"{function.name.upper()}: log(cost) function over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("log(Cost function)")
        plt.legend(loc='best')
        plt.grid(True)
        plt.yscale("log")
        plt.savefig(f"{IMG_PATH}{function.name}_mutation_analysis.png",dpi=200)