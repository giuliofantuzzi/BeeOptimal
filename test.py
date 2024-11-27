from ABC.abc import ArtificialBeeColony
import numpy as np

def sphere(point):
    x, y = point
    res = x**2 + y**2
    return res

def rastrigin(point):
    x, y = point
    res = 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))
    return res

def ackley(point):
    x, y = point
    res = -20 * np.e**(-0.2*np.sqrt(0.5*(x**2 + y**2))) -np.e**(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.e + 20
    return res

def eggholder(point):
    x, y = point
    res = -(y + 47)*np.sin(np.sqrt(np.abs(x/2 + y + 47))) - x*np.sin(np.sqrt(np.abs(x-(y+47))))
    return res

n_bees      = 50
limit       = (n_bees // 2) * 2
max_iters   = 100000

functions_dict = {'sphere' : {'n_bees' : n_bees,
                              'limit' : limit,
                              'max_iters' : max_iters,
                              'lower_bound': [-100,-100], 
                              'upper_bound': [100,100],
                              'function' : sphere
                              },
                  'rastrigin' : {'n_bees' : n_bees,
                                 'limit' : limit,
                                 'max_iters' : max_iters,
                                 'lower_bound': [-5.12,-5.12], 
                                 'upper_bound': [5.12,5.12],
                                 'function' : rastrigin
                                 },
                    'ackley' : {'n_bees' : n_bees,
                                'limit' : limit,
                                'max_iters' : max_iters,
                                'lower_bound': [-5,-5], 
                                'upper_bound': [5,5],
                                'function' : ackley
                                },
                    'eggholder' : {'n_bees' : n_bees,
                                      'limit' : limit,
                                      'max_iters' : max_iters,
                                      'lower_bound': [-512,-512], 
                                      'upper_bound': [512,512],
                                      'function' : eggholder
                                      }
                    }

for name,function in functions_dict.items():
    print(f"Function: {name}")
    ABC = ArtificialBeeColony(n_bees = function['n_bees'],
                              limit = function['limit'],
                              max_iters = function['max_iters'],
                              lower_bound = function['lower_bound'],
                              upper_bound = function['upper_bound'],
                              function = function['function']
                              )
    ABC.optimize()
    print('-'*40)
    print(f"Optimal solution {ABC.optimal_source[0]}")
    print(f"Optimal value {ABC.optimal_source[1]}")
    print('-'*40)