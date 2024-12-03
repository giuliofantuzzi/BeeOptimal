#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
import numpy as np

#--------------------------------------------------------------------------------
# Bee class
#--------------------------------------------------------------------------------
class Bee():
    def __init__(self,function,bounds,position='random'):
        if position=='random':
            self.position = [np.random.uniform(low=bounds[i][0], high=bounds[i][1]) for i in range(len(bounds))]
        elif position=='cahotic':
            raise NotImplementedError
        else:
            assert(len(position) == len(bounds))
            self.position = position
        
        self.function = function
        self.bounds   = bounds
        self.trial    = 0
    
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