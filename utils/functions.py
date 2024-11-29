import numpy as np

def sphere(point):
    x, y = point
    return x**2 + y**2

def rastrigin(point):
    x, y = point
    return 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))

def ackley(point):
    x, y = point
    return -20 * np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.e + 20

def eggholder(point):
    x, y = point
    return -(y + 47)*np.sin(np.sqrt(np.abs(x/2 + y + 47))) - x*np.sin(np.sqrt(np.abs(x-(y+47))))