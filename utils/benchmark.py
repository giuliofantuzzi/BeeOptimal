#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
import numpy as np
from .functions import sphere,rastrigin,ackley,eggholder

#--------------------------------------------------------------------------------
# Benchmark Function class
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
Sphere = BenchmarkFunction(
    name="Sphere",
    fun=sphere,
    bounds=[(-5.12, 5.12), (-5.12, 5.12)],
    optimal_solution=[0, 0]
)

Rastrigin = BenchmarkFunction(
    name="Rastrigin",
    fun=rastrigin,
    bounds=[(-5.12, 5.12), (-5.12, 5.12)],
    optimal_solution=[0, 0]
)

Ackley = BenchmarkFunction(
    name="Ackley",
    fun=ackley,
    bounds=[(-5, 5), (-5, 5)],
    optimal_solution=[0, 0]
)

Eggholder = BenchmarkFunction(
    name="Eggholder",
    fun=eggholder,
    bounds=[(-512, 512), (-512, 512)],
    optimal_solution=[512,404.2319]
)

Sphere30d = BenchmarkFunction(
    name="Sphere30d",
    fun=sphere,
    bounds=[(-5.12, 5.12)]*30,
    optimal_solution=[0]*30
)

Rastrigin30d = BenchmarkFunction(
    name="Rastrigin30d",
    fun=rastrigin,
    bounds=[(-5.12, 5.12)]*30,
    optimal_solution=[0]*30
)