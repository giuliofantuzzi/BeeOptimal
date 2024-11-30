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
    
    def __init__(self,n_bees,limit,function,bounds):
        assert ( n_bees > 2)
        self.dim                    = len(bounds)
        self.n_bees                 = n_bees
        self.limit                  = limit if limit is not None else (self.n_bees //2)* self.dim
        self.function               = function
        self.bounds                 = bounds
        self.n_employed_bees        = self.n_bees // 2
        self.n_onlooker_bees        = self.n_bees - self.n_employed_bees
        self.max_iters              = None
        self.selection              = None
        self.employed_bees          = []
        self.onlooker_bees          = []
        self.colony_history         = []
        self.optimal_source         = ([np.nan for _ in range(self.dim)],np.nan)
        self.optimal_source_history = []
     
    def optimize(self,max_iters=100,selection='RouletteWheel'):
        self.max_iters = max_iters
        self.selection = selection
        # Initialization
        self.employed_bees = [Bee(position = None,
                                  function = self.function,
                                  bounds   = self.bounds) for _ in range(self.n_employed_bees) ]
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
            donor_bee = copy.deepcopy(self.employed_bees[k])
            # Candidate Bee
            j = np.random.randint(0,len(bee.position))
            phi = np.random.uniform(-1,1)
            candidate_bee = copy.deepcopy(bee)
            candidate_bee.position[j] = bee.position[j] + phi*(bee.position[j] - donor_bee.position[j])
            candidate_bee.position[j] = np.clip(candidate_bee.position[j],self.bounds[j][0],self.bounds[j][1])
            # Greedy Selection
            if candidate_bee.fitness >= bee.fitness:
                self.employed_bees[bee_idx] = candidate_bee
            else:
                bee.trial += 1
                
    def waggle_dance_(self):
        fitness_values = [bee.fitness for bee in self.employed_bees]
        if self.selection == 'RouletteWheel':
            selection_probabilities  = [fitness/sum(fitness_values) for fitness in fitness_values]
            dance_winners = np.random.choice(range(self.n_employed_bees),size=self.n_onlooker_bees,p=selection_probabilities)
            return dance_winners
        
    def send_onlookers_(self):
        dance_winners = self.waggle_dance_()
        self.onlooker_bees = [Bee(position = self.employed_bees[winner_idx].position,
                                  function = self.function,
                                  bounds   = self.bounds) for winner_idx in dance_winners
                              ]
        for bee_idx, bee in enumerate(self.onlooker_bees):
            # Choose Donor Bee
            while True:
                k = np.random.randint(0,len(self.onlooker_bees))
                if k != bee_idx:
                    break
            donor_bee = copy.deepcopy(self.onlooker_bees[k])
            # Candidate Bee
            j = np.random.randint(0,len(bee.position))
            phi = np.random.uniform(-1,1)
            candidate_bee = copy.deepcopy(bee)
            candidate_bee.position[j] = bee.position[j] + phi*(bee.position[j] - donor_bee.position[j])
            candidate_bee.position[j] = np.clip(candidate_bee.position[j],self.bounds[j][0],self.bounds[j][1])
            
            # Greedy Selection
            if candidate_bee.fitness >= bee.fitness:
                self.onlooker_bees[bee_idx] = candidate_bee
        
        # Update Employed Bees
        for bee_idx, bee in zip(dance_winners,self.onlooker_bees):
            if bee.fitness > self.employed_bees[bee_idx].fitness:
                self.employed_bees[bee_idx] = bee
    
    def send_scouts_(self):
        for bee_idx, bee in enumerate(self.employed_bees):
            if bee.trial > self.limit:
                self.employed_bees[bee_idx] = Bee(position = None,
                                                  function = self.function,
                                                  bounds   = self.bounds)

#--------------------------------------------------------------------------------