import numpy as np

def sphere(point):
    point = np.array(point)
    return np.sum(point**2)
        
def rastrigin(point):
    point = np.array(point)
    return (10*len(point) + np.sum((point**2 - 10*np.cos(2*np.pi*point))))

def ackley(point):
    return -20 * np.exp(-0.2*np.sqrt(0.5*(point[0]**2 + point[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*point[0]) + np.cos(2*np.pi*point[1]))) + np.e + 20

def eggholder(point):
    return -(point[1] + 47)*np.sin(np.sqrt(np.abs(point[0]/2 + point[1] + 47))) - point[0]*np.sin(np.sqrt(np.abs(point[0]-(point[1]+47))))