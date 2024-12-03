#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
import numpy as np

#--------------------------------------------------------------------------------
# Benchmark functions
#--------------------------------------------------------------------------------
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


#--------------------------------------------------------------------------------
# Benchmark function class
#--------------------------------------------------------------------------------
class BenchmarkFunction:
    def __init__(self, name = 'Unspecified Name', fun=None, bounds=None, optimal_solution=None):
        """
        Initialize a benchmark function.
        
        Args:
            name (str): The name of the function.
            func (callable): The function to evaluate.
            bounds (list of tuples): Bounds for the variables as [(x_min, x_max), (y_min, y_max)].
            true_optimal_value (float, optional): The known optimal value of the function.
            true_solution (list, optional): The known solution(s) achieving the optimal value.
        """
        self.name = name
        self.fun = fun
        self.bounds = bounds
        self.optimal_solution = optimal_solution

    @property
    def optimal_value(self):
        """Return the optimal value of the function."""
        return self.evaluate(self.optimal_solution)
    
    def evaluate(self, point):
        """Evaluate the function at the given point."""
        return self.fun(point)
    
#--------------------------------------------------------------------------------
# Benchmark functions for testing
#--------------------------------------------------------------------------------

# 2d functions

Sphere2d = BenchmarkFunction(
    name="Sphere-2d",
    fun=sphere,
    bounds=[(-5.12, 5.12)]*2,
    optimal_solution=[0]*2
)

Rosenbrock2d = BenchmarkFunction(
    name="Rosenbrock-2d",
    fun=rosenbrock,
    bounds=[(-2.048, 2.048)]*2,
    optimal_solution=[1]*2
)

Ackley2d = BenchmarkFunction(
    name="Ackley-2d",
    fun=ackley,
    bounds=[(-5, 5)]*2,
    optimal_solution=[0]*2
)

Rastrigin2d = BenchmarkFunction(
    name="Rastrigin-2d",
    fun=rastrigin,
    bounds=[(-5.12, 5.12)]*2,
    optimal_solution=[0]*2
)

Griewank2d = BenchmarkFunction(
    name="Griewank-2d",
    fun=griewank,
    bounds=[(-600, 600)]*2,
    optimal_solution=[0]*2
)

Schwefel2d = BenchmarkFunction(
    name="Schwefel-2d",
    fun=schwefel,
    bounds=[(-500, 500)]*2,
    optimal_solution=[420.9687]*2
)

Sumsquares2d = BenchmarkFunction(
    name="Sumsquares-2d",
    fun=sumsquares,
    bounds=[(-10, 10)]*2,
    optimal_solution=[0]*2
)    

Eggholder = BenchmarkFunction(
    name="Eggholder",
    fun=eggholder,
    bounds=[(-512, 512), (-512, 512)],
    optimal_solution=[512,404.2319]
)

# 30d functions
Sphere30d = BenchmarkFunction(
    name="Sphere-30d",
    fun=sphere,
    bounds=[(-5.12, 5.12)]*30,
    optimal_solution=[0]*2
)

Rosenbrock30d = BenchmarkFunction(
    name="Rosenbrock-30d",
    fun=rosenbrock,
    bounds=[(-2.048, 2.048)]*30,
    optimal_solution=[1]*30
)

Ackley30d = BenchmarkFunction(
    name="Ackley-30d",
    fun=ackley,
    bounds=[(-5, 5)]*30,
    optimal_solution=[0]*30
)

Rastrigin30d = BenchmarkFunction(
    name="Rastrigin-30d",
    fun=rastrigin,
    bounds=[(-5.12, 5.12)]*30,
    optimal_solution=[0]*30
)

Griewank30d = BenchmarkFunction(
    name="Griewank-30d",
    fun=griewank,
    bounds=[(-600, 600)]*30,
    optimal_solution=[0]*30
)

Schwefel30d = BenchmarkFunction(
    name="Schwefel-30d",
    fun=schwefel,
    bounds=[(-500, 500)]*30,
    optimal_solution=[420.9687]*30
)

Sumsquares30d = BenchmarkFunction(
    name="Sumsquares-30d",
    fun=sumsquares,
    bounds=[(-10, 10)]*30,
    optimal_solution=[0]*30
)  

# 60d functions
Sphere60d = BenchmarkFunction(
    name="Sphere-60d",
    fun=sphere,
    bounds=[(-5.12, 5.12)]*60,
    optimal_solution=[0]*60
)

Rosenbrock60d = BenchmarkFunction(
    name="Rosenbrock-60d",
    fun=rosenbrock,
    bounds=[(-2.048, 2.048)]*60,
    optimal_solution=[1]*60
)

Ackley60d = BenchmarkFunction(
    name="Ackley-60d",
    fun=ackley,
    bounds=[(-5, 5)]*60,
    optimal_solution=[0]*60
)

Rastrigin60d = BenchmarkFunction(
    name="Rastrigin-60d",
    fun=rastrigin,
    bounds=[(-5.12, 5.12)]*60,
    optimal_solution=[0]*60
)

Griewank60d = BenchmarkFunction(
    name="Griewank-60d",
    fun=griewank,
    bounds=[(-600, 600)]*60,
    optimal_solution=[0]*60
)

Schwefel60d = BenchmarkFunction(
    name="Schwefel-60d",
    fun=schwefel,
    bounds=[(-500, 500)]*60,
    optimal_solution=[420.9687]*60
)

Sumsquares60d = BenchmarkFunction(
    name="Sumsquares-60d",
    fun=sumsquares,
    bounds=[(-10, 10)]*60,
    optimal_solution=[0]*60
)  
#--------------------------------------------------------------------------------