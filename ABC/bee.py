#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
import numpy as np

#--------------------------------------------------------------------------------
# Bee class
#--------------------------------------------------------------------------------
class Bee():
    def __init__(self, position,function,lower_bound,upper_bound):
        assert(len(lower_bound) == len(upper_bound))
        
        if position:
            assert(len(position) == len(lower_bound))
            self.position = position
        else:
            self.position = [np.random.uniform(low=lower_bound[i], high=upper_bound[i]) for i in range(len(lower_bound))]
        
        self.function    = function
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.trial       = 0
    
    @property
    def value(self):
        return self.function(self.position) 
    
    @property
    def fitness(self):
        if self.value >= 0:
            return 1/(1+self.value)
        else:
            return 1 + abs(self.value)
        
#-------------------------------------------------------------------------------- 