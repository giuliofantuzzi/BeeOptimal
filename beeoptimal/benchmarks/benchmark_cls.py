#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
import numpy as np

#--------------------------------------------------------------------------------
# Benchmark function class
#--------------------------------------------------------------------------------
class BenchmarkFunction:
    """
        Class to define a benchmark function for optimization.
        
        Attributes:
            - name (str)                    : The name of the function. Defaults to 'Unspecified Name'.
            - fun (callable)                : The function to evaluate. Defaults to None.
            - bounds (array-like)           : The bounds of the function, provided as a 2D array. Defaults to None.
            - optimal_solution (array-like) : The known solution achieving the optimal value. Defaults to None.

        Methods:
            - evaluate(float)      : Evaluate the function at the given point.
            - optimal_value(float) : Return the optimal value of the function.
    """
    
    def __init__(self, name = 'Unspecified Name', fun=None, bounds=None, optimal_solution=None):
        """
        Initialize a benchmark function.
        
        Args:
            - name (str)                    : The name of the function. Defaults to 'Unspecified Name'.
            - fun (callable)                : The function to evaluate. Defaults to None.
            - bounds (array-like)           : The bounds of the function, provided as a 2D array. Defaults to None.
            - optimal_solution (array-like) : The known solution achieving the optimal value. Defaults to None.
        """
        self.name = name
        self.fun = fun
        self.bounds = bounds
        self.optimal_solution = optimal_solution

    def evaluate(self, point):
        """
        Evaluate the function at the given point.
        
        Args:
            - point (array-like) : The point at which to evaluate the function.
        
        Returns:
            - float: The value of the function computed at the given point.
            
        Raises:
            - ValueError: If the function is not defined.
        """
        
        if self.fun is None:
            raise ValueError("The function is not defined.")
        return self.fun(point)
    
    @property
    def optimal_value(self):
        """Return the optimal value of the function."""
        return self.evaluate(self.optimal_solution)