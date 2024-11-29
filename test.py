#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
from ABC.abc import ArtificialBeeColony
import numpy as np
from utils.benchmark import Sphere,Rastrigin,Ackley,Eggholder
from utils.plotting_utils import ContourPlotBee
import tempfile
from PIL import Image

#--------------------------------------------------------------------------------
# Global variables and settings
#--------------------------------------------------------------------------------
N_BEES      = 50
LIMIT      = (N_BEES // 2) * 2
MAX_ITERS   = 50
BENCHMARK_FUNCTIONS = [Sphere,Rastrigin,Ackley,Eggholder]

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
        
        ABC = ArtificialBeeColony(n_bees      = N_BEES,
                                  limit       = LIMIT,
                                  max_iters   = MAX_ITERS,
                                  bounds      = function_test.bounds,
                                  function    = function_test.fun
                                )
        ABC.optimize()
        
        print(f"Optimal Solution:")
        print(f"\tExpected : {function_test.optimal_solution}")
        print(f"\tFound    : {ABC.optimal_source[0]}")
        print(f"Optimal Value:")
        print(f"\tExpected : {function_test.optimal_value}")
        print(f"\tFound    : {ABC.optimal_source[1]}")
        print('-'*100)
        
        # GIF
        x = np.linspace(function_test.bounds[0][0]-0.05*np.abs(function_test.bounds[0][0]),
                        function_test.bounds[0][1]+0.05*np.abs(function_test.bounds[0][1]), 
                        100)
        y = np.linspace(function_test.bounds[1][0]-0.05*np.abs(function_test.bounds[1][0]),
                        function_test.bounds[1][1]+0.05*np.abs(function_test.bounds[1][1]),
                        100)
        X, Y = np.meshgrid(x, y)
        Z = function_test.evaluate((X, Y))
        
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
                fig.write_image(file_path, format="png", scale=2)
                image_files.append(file_path)

            # Open images and save as GIF
            images = [Image.open(file) for file in image_files]
            gif_path = f"images/{function_test.name}_animated_opt.gif"
            images[0].save(gif_path, save_all=True, append_images=images[1:], 
                           duration=300, loop=0)
            print(f"Animated GIF saved in {gif_path}")