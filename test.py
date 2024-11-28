#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
from ABC.abc import ArtificialBeeColony
import numpy as np
from utils.test_functions import *
from utils.plotting_utils import ContourPlotBee
import tempfile
from PIL import Image

#--------------------------------------------------------------------------------
# Global variables and settings
#--------------------------------------------------------------------------------
N_BEES      = 50
LIMIT      = (N_BEES // 2) * 2
MAX_ITERS   = 50

# functions_info = {'sphere' : {'function' : sphere,
#                               'lower_bound': [-100,-100], 
#                               'upper_bound': [100,100],
#                               'optimal_solution': [0,0],
#                               'optimal_value': 0
#                               },
#                   'rastrigin' : {'function' : rastrigin,
#                                  'lower_bound': [-5.12,-5.12],
#                                  'upper_bound': [5.12,5.12],
#                                  'optimal_solution': [0,0],
#                                  'optimal_value': 0
#                                  },
#                   'ackley' : {'function' : ackley,
#                               'lower_bound': [-5,-5], 
#                               'upper_bound': [5,5],
#                               'optimal_solution': [0,0],
#                               'optimal_value': 0
#                               },
functions_info = {
                  'eggholder' : {'function' : eggholder,
                                 'lower_bound': [-512,-512],
                                 'upper_bound': [512,512],
                                 'optimal_solution': [512,404.2319],
                                 'optimal_value': -959.6407
                                }
                 }


#--------------------------------------------------------------------------------
# Tests
#--------------------------------------------------------------------------------
if __name__ == '__main__':
    
    for name,function in functions_info.items():
        # OPTIMIZATION
        print('\n')
        print('-'*100)
        print(f"Function: {name.upper()}")
        
        ABC = ArtificialBeeColony(n_bees      = N_BEES,
                                  limit       = LIMIT,
                                  max_iters   = MAX_ITERS,
                                  lower_bound = function['lower_bound'],
                                  upper_bound = function['upper_bound'],
                                  function    = function['function']
                                )
        ABC.optimize()
        
        print(f">> Optimal solution -> Expected: {function['optimal_solution']} , Found: {ABC.optimal_source[0]}")
        print(f">> Optimal value    -> Expected: {function['optimal_value']} , Found: {ABC.optimal_source[1]}")
        print('-'*100)
        # PLOT
        x = np.linspace(function['lower_bound'][0]-0.05*np.abs(function['lower_bound'][0]), function['upper_bound'][0]+0.05*np.abs(function['upper_bound'][0]), 100)
        y = np.linspace(function['lower_bound'][1]-0.05*np.abs(function['lower_bound'][1]), function['upper_bound'][1]+0.05*np.abs(function['upper_bound'][1]), 100)
        
        X, Y = np.meshgrid(x, y)
        Z = function['function']((X, Y))
        plots = []
        for iteration,bee_colony in enumerate(ABC.colony_history):
            plots.append(ContourPlotBee(x,y,Z,bee_colony,title=f"Iteration {iteration+1} / {ABC.max_iters}",marker_path='assets/BeeMarker.png',optimal_solution=function['optimal_solution']))
            
        with tempfile.TemporaryDirectory() as tmpdirname:
            image_files = []
            # Save each figure as a separate image file
            for i, fig in enumerate(plots):
                # Define the file path
                file_path = f"{tmpdirname}/frame_{i}.png"
                fig.write_image(file_path, format="png", scale=2)  # Adjust 'scale' for image resolution
                image_files.append(file_path)

            # Open images and save as GIF
            images = [Image.open(file) for file in image_files]
            gif_path = f"images/{name}_animated_opt.gif"
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=400, loop=0)
    print('\n')