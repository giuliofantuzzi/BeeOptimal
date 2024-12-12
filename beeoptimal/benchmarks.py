#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
import numpy as np

#--------------------------------------------------------------------------------
# Benchmark function class
#--------------------------------------------------------------------------------
class BenchmarkFunction:
    """
        Class to define a D-dimensional benchmark function for optimization.
        
        Args:
            name (str)                    : The name of the function. Defaults to 'Unspecified Name'.
            fun (callable)                : The function to evaluate. Defaults to None.
            bounds (array-like)           : The bounds of the function, provided as a numpy array of shape `(D,2)`. Defaults to None.
            optimal_solution (array-like) : The known solution achieving the optimal value, provided as numpy array of shape `(D,)`. Defaults to None. 
            
        Examples:
            >>> from beeoptimal.benchmarks import BenchmarkFunction
            >>> import numpy as np
            >>> sphere = lambda point: np.sum(np.array(point)**2)
            >>> Sphere2d = BenchmarkFunction(
                    name             = "Sphere-2d",
                    fun              = sphere,
                    bounds           = np.array([(-5.12, 5.12)]*2),
                    optimal_solution = np.zeros(2)
                )
    """
    
    def __init__(self, name = 'Unspecified Name', fun=None, bounds=None, optimal_solution=None):
        """
        Initialize a benchmark function.
        
        Args:
            name (str)                    : The name of the function. Defaults to 'Unspecified Name'.
            fun (callable)                : The function to evaluate. Defaults to None.
            bounds (array-like)           : The bounds of the function, provided as a numpy array of shape `(D,2)`. Defaults to None.
            optimal_solution (array-like) : The known solution achieving the optimal value, provided as numpy array of shape `(D,)`. Defaults to None.
        """
        self.name = name
        self.fun = fun
        self.bounds = bounds
        self.optimal_solution = optimal_solution

    def evaluate(self, point):
        """
        Evaluate the function at the given point.
        
        Args:
            point (array-like) : The point at which to evaluate the function, provided as numpy array of shape `(D,)`.
        
        Returns:
            float: The value of the function computed at the given point.
            
        Raises:
            ValueError: If the function is not defined.
        """
        
        if self.fun is None:
            raise ValueError("The function is not defined.")
        return self.fun(point)
    
    @property
    def optimal_value(self):
        """Return the optimal value of the function."""
        return self.evaluate(self.optimal_solution)
    
#--------------------------------------------------------------------------------
#  Callable benchmark functions
#--------------------------------------------------------------------------------
def sphere(point):
    return np.sum(point**2)

def rosenbrock(point):
    return np.sum(100*(point[1:] - point[:-1]**2)**2 + (point[:-1]-1)**2)
        
def ackley(point):
    D = len(point)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(point**2) / D)) - np.exp(np.sum(np.cos(2 * np.pi * point)) / D) + 20 + np.e

def rastrigin(point):
    return (10*len(point) + np.sum((point**2 - 10*np.cos(2*np.pi*point))))

def weierstrass(point,a=0.5,b=3,k_max=20):
    k = np.arange(k_max + 1)
    term1 = np.sum(a**k[:, None] * np.cos(2 * np.pi * b**k[:, None] * (point + 0.5)), axis=0)
    term2 = np.sum(a**k * np.cos(2 * np.pi * b**k * 0.5))
    return np.sum(term1) - len(point) * term2

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

    
    
#--------------------------------------------------------------------------------
# Benchmark functions for testing
#--------------------------------------------------------------------------------

# 2d functions

Sphere2d = BenchmarkFunction(
    name             = "Sphere-2d",
    fun              = sphere,
    bounds           = np.array([(-5.12, 5.12)]*2),
    optimal_solution = np.zeros(2)
)

Rosenbrock2d = BenchmarkFunction(
    name             = "Rosenbrock-2d",
    fun              = rosenbrock,
    bounds           = np.array([(-2.048, 2.048)]*2),
    optimal_solution = np.ones(2)
)

Ackley2d = BenchmarkFunction(
    name             = "Ackley-2d",
    fun              = ackley,
    bounds           = np.array([(-5, 5)]*2),
    optimal_solution = np.zeros(2)
)

Rastrigin2d = BenchmarkFunction(
    name             = "Rastrigin-2d",
    fun              = rastrigin,
    bounds           = np.array([(-5.12, 5.12)]*2),
    optimal_solution = np.zeros(2)
)

Weierstrass2d = BenchmarkFunction(
    name             = "Weierstrass-2d",
    fun              = weierstrass,
    bounds           = np.array([(-0.5, 0.5)]*2),
    optimal_solution = np.zeros(2)
)

Griewank2d = BenchmarkFunction(
    name             = "Griewank-2d",
    fun              = griewank,
    bounds           = np.array([(-600, 600)]*2),
    optimal_solution = np.zeros(2)
)

Schwefel2d = BenchmarkFunction(
    name             = "Schwefel-2d",
    fun              = schwefel,
    bounds           = np.array([(-500, 500)]*2),
    optimal_solution = np.full(2,420.9687)
)

Sumsquares2d = BenchmarkFunction(
    name             = "Sumsquares-2d",
    fun              = sumsquares,
    bounds           = np.array([(-10, 10)]*2),
    optimal_solution = np.zeros(2)
)    

Eggholder = BenchmarkFunction(
    name             = "Eggholder",
    fun              = eggholder,
    bounds           = np.array([(-512, 512)]*2),
    optimal_solution = np.array([512,404.2319])
)

# 10d functions

Sphere10d = BenchmarkFunction(
    name             = "Sphere-10d",
    fun              = sphere,
    bounds           = np.array([(-5.12, 5.12)]*10),
    optimal_solution = np.zeros(10)
)

Rosenbrock10d = BenchmarkFunction(
    name             = "Rosenbrock-10d",
    fun              = rosenbrock,
    bounds           = np.array([(-2.048, 2.048)]*10),
    optimal_solution = np.ones(10)
)

Ackley10d = BenchmarkFunction(
    name             = "Ackley-10d",
    fun              = ackley,
    bounds           = np.array([(-5, 5)]*10),
    optimal_solution = np.zeros(2)
)

Rastrigin10d = BenchmarkFunction(
    name             = "Rastrigin-10d",
    fun              = rastrigin,
    bounds           = np.array([(-5.12, 5.12)]*10),
    optimal_solution = np.zeros(10)
)

Weierstrass10d = BenchmarkFunction(
    name             = "Weierstrass-10d",
    fun              = weierstrass,
    bounds           = np.array([(-0.5, 0.5)]*10),
    optimal_solution = np.zeros(10)
)

Griewank10d = BenchmarkFunction(
    name             = "Griewank-10d",
    fun              = griewank,
    bounds           = np.array([(-600, 600)]*10),
    optimal_solution = np.zeros(10)
)

Schwefel10d = BenchmarkFunction(
    name             = "Schwefel-10d",
    fun              = schwefel,
    bounds           = np.array([(-500, 500)]*10),
    optimal_solution = np.full(10,420.9687)
)

Sumsquares10d = BenchmarkFunction(
    name             = "Sumsquares-10d",
    fun              = sumsquares,
    bounds           = np.array([(-10, 10)]*10),
    optimal_solution = np.zeros(10)
)    

# 30d functions

Sphere30d = BenchmarkFunction(
    name             = "Sphere-30d",
    fun              = sphere,
    bounds           = np.array([(-5.12, 5.12)]*30),
    optimal_solution = np.zeros(30)
)

Rosenbrock30d = BenchmarkFunction(
    name             = "Rosenbrock-30d",
    fun              = rosenbrock,
    bounds           = np.array([(-2.048, 2.048)]*30),
    optimal_solution = np.ones(30)
)

Ackley30d = BenchmarkFunction(
    name             = "Ackley-30d",
    fun              = ackley,
    bounds           = np.array([(-5, 5)]*30),
    optimal_solution = np.zeros(2)
)

Rastrigin30d = BenchmarkFunction(
    name             = "Rastrigin-30d",
    fun              = rastrigin,
    bounds           = np.array([(-5.12, 5.12)]*30),
    optimal_solution = np.zeros(30)
)

Weierstrass30d = BenchmarkFunction(
    name             = "Weierstrass-30d",
    fun              = weierstrass,
    bounds           = np.array([(-0.5, 0.5)]*30),
    optimal_solution = np.zeros(30)
)

Griewank30d = BenchmarkFunction(
    name             = "Griewank-30d",
    fun              = griewank,
    bounds           = np.array([(-600, 600)]*30),
    optimal_solution = np.zeros(30)
)

Schwefel30d = BenchmarkFunction(
    name             = "Schwefel-30d",
    fun              = schwefel,
    bounds           = np.array([(-500, 500)]*30),
    optimal_solution = np.full(30,420.9687)
)

Sumsquares30d = BenchmarkFunction(
    name             = "Sumsquares-30d",
    fun              = sumsquares,
    bounds           = np.array([(-10, 10)]*30),
    optimal_solution = np.zeros(30)
)    
#--------------------------------------------------------------------------------