#++++++++++++++++++++++++++++++++++++
# Libraries and modules
#++++++++++++++++++++++++++++++++++++

from beeoptimal import ArtificialBeeColony
import numpy as np
from beeoptimal.benchmarks import *
from beeoptimal.plotting import contourplot_bees
import tempfile
from PIL import Image

#++++++++++++++++++++++++++++++++++++
# Global variables and settings
#++++++++++++++++++++++++++++++++++++

COLONY_SIZE         = 100
LIMIT               = 'default'
MAX_ITERS           = 1000
STAGNATION_TOL      = 1e-6
BENCHMARK_FUNCTIONS = [Sphere2d,Rosenbrock2d,Ackley2d,Rastrigin2d,Weierstrass2d,Griewank2d,Schwefel2d,Sumsquares2d,Eggholder]
SELECTION           = 'RouletteWheel'
MUTATIONS           = ['StandardABC','ModifiedABC','ABC/best/1','ABC/best/2','DirectedABC']
INITIALIZATIONS     = ['random','cahotic']
MR                  = 0.7
SF                  = 1.0
SELF_ADAPTIVE_SF    = False
VERBOSE             = True
RANDOM_SEED         = 12345
MUTATION_NAMES      = ['StandardABC','ModifiedABC','ABCbest1','ABCbest2','DirectedABC']
GIF_PATH            = 'images/optimization_gifs/'

#++++++++++++++++++++++++++++++++++++
# Tests
#++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':
    
    print("/////////////////////////////////////////////////////")
    print("          TESTING ABC ON BENCMARK FUNCTIONS          ")
    print("/////////////////////////////////////////////////////")
    
    for function in BENCHMARK_FUNCTIONS:
        for mutation,mutation_name in zip(MUTATIONS,MUTATION_NAMES):
            for initialization in INITIALIZATIONS:
        
                # OPTIMIZATION  
                print('\n')
                print('-'*100)
                print(f"Evaluating {function.name.upper()} with {initialization} initialization and {mutation} mutation")
                print('-'*100)
                
                ABC = ArtificialBeeColony(
                    colony_size     = COLONY_SIZE,
                    bounds          = np.array(function.bounds),
                    function        = function.fun,
                    n_employed_bees = None
                    )

                ABC.optimize(
                    max_iters        = MAX_ITERS,
                    limit            = LIMIT,
                    selection        = SELECTION,
                    mutation         = mutation,
                    initialization   = initialization,
                    sf               = SF,
                    self_adaptive_sf = SELF_ADAPTIVE_SF,
                    mr               = MR,
                    stagnation_tol   = STAGNATION_TOL,
                    verbose          = VERBOSE,
                    random_seed      = RANDOM_SEED
                    )
                
                print(f"Optimal Solution:")
                print(f"\tExpected : {function.optimal_solution}")
                print(f"\tFound    : {ABC.optimal_bee.position}")
                print(f"Optimal Value:")
                print(f"\tExpected : {function.optimal_value}")
                print(f"\tFound    : {ABC.optimal_bee.value}")
                print('-'*100)
                
                # GIF
                plots = []
                # Adaptive step in order to have gifs with same number of frames
                step = max(1, (ABC.actual_iters+1) // 50) #Note: actual_iters +1 to include initial population
                for iteration in range(0,(ABC.actual_iters+1),step): 
                    plots.append(contourplot_bees(
                        function=function,
                        bee_colony=ABC.colony_history[iteration],
                        title=f"{function.name.upper()} optimization [Iteration {iteration} / {ABC.actual_iters}]",
                        optimal_solution=function.optimal_solution,
                        bounds=None,
                        zoom = 1.0,
                        figsize=(600,600)
                        ))

                with tempfile.TemporaryDirectory() as tmpdirname:
                    image_files = []
                    # Save each figure as a separate image file
                    for i, fig in enumerate(plots):
                        # Define the file path
                        file_path = f"{tmpdirname}/frame_{i}.png"
                        fig.write_image(file_path, format="png", scale=3)   # Plotly version
                        #fig.savefig(file_path, format="png",dpi=300)       # Matplotlib version
                        image_files.append(file_path)

                    # Open images and save as GIF
                    images = [Image.open(file) for file in image_files]
                    gif_path = GIF_PATH  + f"{mutation_name}/" + f"{function.name}_{mutation_name}_{initialization}.gif"
                    images[0].save(gif_path, save_all=True, append_images=images[1:], 
                                duration=200, loop=0)
                    print(f"Animated GIF saved in {gif_path}")
        
#--------------------------------------------------------------------------------