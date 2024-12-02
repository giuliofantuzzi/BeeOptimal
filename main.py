#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
from ABC import ArtificialBeeColony
import numpy as np
from utils.benchmark import Sphere,Rastrigin,Ackley,Eggholder
from utils.plotting_utils import ContourPlotBee
import tempfile
from PIL import Image

#--------------------------------------------------------------------------------
# Global variables and settings
#--------------------------------------------------------------------------------
N_BEES              = 100
LIMIT               = 'default'
MAX_ITERS           = 50
BENCHMARK_FUNCTIONS = [Sphere,Rastrigin,Ackley,Eggholder]
SELECTION           = 'RouletteWheel'
MUTATION            = 'ABC/best/1'
MR                  = 0.8
SF                  = 1.0
SelfAdaptiveSF      = True
verbose             = False
random_seed         = 1234
GIF_PATH            = 'images/ABCbest2/'
#--------------------------------------------------------------------------------
# Tests
#--------------------------------------------------------------------------------
if __name__ == '__main__':
    
    print("/////////////////////////////////////////////////////")
    print("          TESTING ABC ON BENCMARK FUNCTIONS          ")
    print("/////////////////////////////////////////////////////")
    
    for function_test in BENCHMARK_FUNCTIONS:
        
        # OPTIMIZATION  
        print('\n')
        print('-'*100)
        print(f"Function: {function_test.name.upper()}")
        print('-'*100)
        
        ABC = ArtificialBeeColony(n_bees   = N_BEES,
                                  bounds   = function_test.bounds,
                                  function = function_test.fun)

        ABC.optimize(max_iters      = MAX_ITERS,
                     limit          = LIMIT,
                     selection      = SELECTION,
                     mutation       = MUTATION,
                     SF             = SF,
                     SelfAdaptiveSF = SelfAdaptiveSF,
                     MR             = MR,
                     verbose        = False,
                     random_seed    = random_seed)
        
        print(f"Optimal Solution:")
        print(f"\tExpected : {function_test.optimal_solution}")
        print(f"\tFound    : {ABC.optimal_bee.position}")
        print(f"Optimal Value:")
        print(f"\tExpected : {function_test.optimal_value}")
        print(f"\tFound    : {ABC.optimal_bee.value}")
        print('-'*100)
        
        # GIF
        x = np.linspace(function_test.bounds[0][0]-0.05*np.abs(function_test.bounds[0][0]),
                        function_test.bounds[0][1]+0.05*np.abs(function_test.bounds[0][1]), 
                        100)
        y = np.linspace(function_test.bounds[1][0]-0.05*np.abs(function_test.bounds[1][0]),
                        function_test.bounds[1][1]+0.05*np.abs(function_test.bounds[1][1]),
                        100)
        X, Y = np.meshgrid(x, y)
        points = np.c_[X.ravel(), Y.ravel()]  
        Z = np.array([function_test.evaluate(p) for p in points]).reshape(X.shape)
        
        plots = []
        for iteration,bee_colony in enumerate(ABC.colony_history):
            plots.append(ContourPlotBee(x=x,y=y,Z=Z,bee_colony=bee_colony,
                                        title=f"Iteration {iteration+1} / {ABC.max_iters}",
                                        marker_path='assets/BeeMarker.png',
                                        optimal_solution=function_test.optimal_solution))
            
        with tempfile.TemporaryDirectory() as tmpdirname:
            image_files = []
            # Save each figure as a separate image file
            for i, fig in enumerate(plots):
                # Define the file path
                file_path = f"{tmpdirname}/frame_{i}.png"
                fig.write_image(file_path, format="png", scale=3)
                image_files.append(file_path)

            # Open images and save as GIF
            images = [Image.open(file) for file in image_files]
            gif_path = GIF_PATH + f"{function_test.name}_animated_opt.gif"
            images[0].save(gif_path, save_all=True, append_images=images[1:], 
                           duration=300, loop=0)
            print(f"Animated GIF saved in {gif_path}")
            
#--------------------------------------------------------------------------------