#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
import numpy as np

#--------------------------------------------------------------------------------
# Bee class
#--------------------------------------------------------------------------------
class Bee():
    """
    Instantiates a bee object for the artificial bee colony algorithm.

    Args:
        position (array-like) : The current position of the bee in the search space.
        function (callable)   : The objective function to evaluate the position.
        bounds (array-like)   : Bounds for each dimension of the search space.
        trial (int)           : Counter to track the number of trials or unsuccessful updates for the bee.
    
    .. warning::
        AssertionError: If the length of the position and bounds arrays are not equal.
    """
    
    def __init__(self,position,function,bounds):
        """
        Initializes a Bee instance with a position, an objective function, and search space bounds.

        Args:
            position (array-like) : The initial position of the bee in the search space
            function (callable)   : The objective function to evaluate the position
            bounds (array-like)   : Bounds for each dimension of the search space.
            trial (int)           : Counter to track the number of trials or unsuccessful updates for the bee.
        
        Raises:
            AssertionError: If the length of the position and bounds arrays are not equal.
        """
    
        assert(len(position) == len(bounds)) , 'Position and bounds must have the same length'
        self.position = position
        self.function = function
        self.bounds   = bounds
        self.trial    = 0
    
    @property
    def value(self):
        """
        Computes the value of the objective function at the bee's current position.

        Returns:
            float: The objective function value for the current position.
        """
        return self.function(self.position) 
    
    @property
    def fitness(self):
        """
        Computes the fitness of the bee based on the value of the objective function

        Returns:
            float: The fitness value at the bee's current position.
        """
        if self.value >= 0:
            return 1/(1+self.value)
        else:
            return 1 + np.abs(self.value)
        
#-------------------------------------------------------------------------------- 