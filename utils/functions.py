import numpy as np

def sphere(point):
    point = np.array(point)
    return np.sum(point**2)

def rosenbrock(point):
    point = np.array(point)
    return np.sum(100*(point[1:] - point[:-1]**2)**2 + (point[:-1]-1)**2)
        
def ackley(point):
    return -20 * np.exp(-0.2*np.sqrt(0.5*(point[0]**2 + point[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*point[0]) + np.cos(2*np.pi*point[1]))) + np.e + 20

def rastrigin(point):
    point = np.array(point)
    return (10*len(point) + np.sum((point**2 - 10*np.cos(2*np.pi*point))))

def griewank(point):
    point = np.array(point)
    return 1 + np.sum(point**2)/4000 - np.prod(np.cos(point/np.sqrt(np.arange(1, len(point)+1))))

def schwefel(point):
    point = np.array(point)
    return 418.9829*len(point) - np.sum(point*np.sin(np.sqrt(np.abs(point))))

def sumsquares(point):
    point = np.array(point)
    return np.sum(np.arange(1, len(point)+1)* (point**2) )

def eggholder(point):
    return -(point[1] + 47)*np.sin(np.sqrt(np.abs(point[0]/2 + point[1] + 47))) - point[0]*np.sin(np.sqrt(np.abs(point[0]-(point[1]+47))))
