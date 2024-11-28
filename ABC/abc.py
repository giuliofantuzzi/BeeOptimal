#--------------------------------------------------------------------------------
# Libraries and modules
#--------------------------------------------------------------------------------
import numpy as np
import copy
from .bee import Bee
from tqdm import trange

#--------------------------------------------------------------------------------
# Artificial Bee Colony (ABC) class
#--------------------------------------------------------------------------------
class ArtificialBeeColony():
    
    def __init__(self,n_bees,limit,max_iters,function,lower_bound,upper_bound):
        assert (len(lower_bound) == len(upper_bound))
        assert ( n_bees > 1)
        self.dim             = len(lower_bound)
        self.n_bees          = n_bees
        self.limit           = limit if limit is not None else (self.n_bees //2)* self.dim
        self.max_iters       = max_iters
        self.function        = function
        self.lower_bound     = lower_bound
        self.upper_bound     = upper_bound
        self.n_employed_bees = self.n_bees // 2
        self.n_onlooker_bees = self.n_bees - self.n_employed_bees
        self.employed_bees   = None
        self.onlooker_bees   = None
        self.optimal_source  = None
        self.optimal_source_history = []
        self.colony_history = []
     
    def optimize(self):
        
        # Initialization
        self.employed_bees = [Bee(position    = None,
                                  function    = self.function,
                                  lower_bound = self.lower_bound,
                                  upper_bound = self.upper_bound) for _ in range(self.n_employed_bees) ]
        self.optimal_source_history.append(max(self.employed_bees,key=lambda bee: bee.fitness).position)
        self.colony_history.append([bee.position for bee in self.employed_bees])
        # Loop
        for _ in trange(self.max_iters,desc='Running Optimization'):
            self.send_employees_()
            self.send_onlookers_()
            self.send_scouts_()
            self.optimal_source_history.append(max(self.employed_bees,key=lambda bee: bee.fitness).position)
            self.colony_history.append([bee.position for bee in self.employed_bees])
        
        # Store best Solution
        self.optimal_source = (max(self.employed_bees,key=lambda bee: bee.fitness).position, 
                               max(self.employed_bees,key=lambda bee: bee.fitness).value)
        
    def send_employees_(self):
        for bee_idx, bee in enumerate(self.employed_bees):
            # Choose Donor Bee
            while True:
                k = np.random.randint(0,len(self.employed_bees))
                if k != bee_idx:
                    break
            DonorBee = copy.deepcopy(self.employed_bees[k])
            # Candidate Bee
            j = np.random.randint(0,len(bee.position))
            phi = np.random.uniform(-1,1)
            CandidateBee = copy.deepcopy(bee)
            CandidateBee.position[j] = bee.position[j] + phi*(bee.position[j] - DonorBee.position[j])
            if CandidateBee.position[j] < self.lower_bound[j]:
                CandidateBee.position[j] = self.lower_bound[j]
            if CandidateBee.position[j] > self.upper_bound[j]:
                CandidateBee.position[j] = self.upper_bound[j]
            # Greedy Selection
            if CandidateBee.fitness >= bee.fitness:
                self.employed_bees[bee_idx] = CandidateBee
            else:
                bee.trial += 1
                
    def waggle_dance_(self,selection='RouletteWheel'):
        fitness_values = [bee.fitness for bee in self.employed_bees]
        if selection == 'RouletteWheel':
            selection_probabilities  = [fitness/sum(fitness_values) for fitness in fitness_values]
            dance_winners = np.random.choice(range(self.n_employed_bees),size=self.n_onlooker_bees,p=selection_probabilities)
            return dance_winners
        
    def send_onlookers_(self):
        dance_winners = self.waggle_dance_()
        self.onlooker_bees = [Bee(position    = self.employed_bees[winner_idx].position,
                                  function    = self.function,
                                  lower_bound = self.lower_bound,
                                  upper_bound = self.upper_bound) for winner_idx in dance_winners
                              ]
        
        for bee_idx, bee in enumerate(self.onlooker_bees):
            # Choose Donor Bee
            while True:
                k = np.random.randint(0,len(self.onlooker_bees))
                if k != bee_idx:
                    break
            DonorBee = copy.deepcopy(self.onlooker_bees[k])
            # Candidate Bee
            j = np.random.randint(0,len(bee.position))
            phi = np.random.uniform(-1,1)
            CandidateBee = copy.deepcopy(bee)
            CandidateBee.position[j] = bee.position[j] + phi*(bee.position[j] - DonorBee.position[j])
            if CandidateBee.position[j] < self.lower_bound[j]:
                CandidateBee.position[j] = self.lower_bound[j]
            if CandidateBee.position[j] > self.upper_bound[j]:
                CandidateBee.position[j] = self.upper_bound[j]
            
            # Greedy Selection
            if CandidateBee.fitness >= bee.fitness:
                self.onlooker_bees[bee_idx] = CandidateBee
        
        # Update Employed Bees
        for bee_idx, bee in zip(dance_winners,self.onlooker_bees):
            if bee.fitness > self.employed_bees[bee_idx].fitness:
                self.employed_bees[bee_idx] = bee
    
    def send_scouts_(self):
        for bee_idx, bee in enumerate(self.employed_bees):
            if bee.trial > self.limit:
                self.employed_bees[bee_idx] = Bee(position    = None,
                                                  function    = self.function,
                                                  lower_bound = self.lower_bound,
                                                  upper_bound = self.upper_bound )

#--------------------------------------------------------------------------------