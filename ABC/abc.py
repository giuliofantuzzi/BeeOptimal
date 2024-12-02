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
        self.dim                 = len(bounds)
        self.n_bees              = n_bees
        self.limit               = limit if limit is not None else (self.n_bees //2)* self.dim
        self.function            = function
        self.bounds              = bounds
        self.n_employed_bees     = self.n_bees // 2
        self.n_onlooker_bees     = self.n_bees - self.n_employed_bees
        self.max_iters           = None
        self.selection           = None
        self.mutation            = None
        self.employed_bees       = []
        self.onlooker_bees       = []
        self.colony_history      = []
        self.optimal_bee         = []
        self.optimal_bee_history = []
     
    def optimize(self,max_iters=100,selection='RouletteWheel',mutation='StandardABC',verbose=False,random_seed=None):
        
        self.max_iters = max_iters
        self.selection = selection
        self.mutation  = mutation
        
        # Initialization
        if random_seed:
            np.random.seed(random_seed)
        self.employed_bees = [Bee(position = None,
                                  function = self.function,
                                  bounds   = self.bounds) for _ in range(self.n_employed_bees) ]
        self.colony_history.append(copy.deepcopy(self.employed_bees))
        self.optimal_bee = copy.deepcopy(max(self.employed_bees,key=lambda bee: bee.fitness))
        self.optimal_bee_history.append(copy.deepcopy(self.optimal_bee))
        
        # Loop
        for _ in trange(self.max_iters,desc='Running Optimization',disable= not verbose):
            self.send_employees_()
            self.send_onlookers_()
            self.send_scouts_()
            self.colony_history.append(copy.deepcopy(self.employed_bees))
            self.optimal_bee = copy.deepcopy(max(self.employed_bees,key=lambda bee: bee.fitness))
            self.optimal_bee_history.append(copy.deepcopy(self.optimal_bee))
    
    def send_employees_(self):
        for bee_idx, bee in enumerate(self.employed_bees):
            candidate_bee = self.get_candidate_neighbor_(bee=bee,bee_idx=bee_idx,population=self.employed_bees)
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
            # Get Candidate Neighbor
            candidate_bee = self.get_candidate_neighbor_(bee=bee,bee_idx=bee_idx,population=self.onlooker_bees)
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
                
    def get_candidate_neighbor_(self,bee,bee_idx,population):
        
        if self.mutation == 'StandardABC':
            # Donor Bee
            donor_bee = self.get_donor_bees_(n_donors=1,bee_idx=bee_idx,population=population)[0]
            # Candidate Bee
            j = np.random.randint(0,len(bee.position))
            phi = np.random.uniform(-1,1)
            candidate_bee = copy.deepcopy(bee)
            candidate_bee.position[j] = bee.position[j] + phi*(bee.position[j] - donor_bee.position[j])
            candidate_bee.position[j] = np.clip(candidate_bee.position[j],self.bounds[j][0],self.bounds[j][1])
            
        # Probabilistic mutation still to do (with MR parameter)
            
        if self.mutation == 'ABC/best/1':
            # 2 Donor Bees
            donor1,donor2 = self.get_donor_bees_(n_donors=2,bee_idx=bee_idx,population=population)
            # Candidate Bee
            j = np.random.randint(0,len(bee.position))
            phi = np.random.uniform(-1,1)
            candidate_bee = copy.deepcopy(bee)
            candidate_bee.position[j] = self.optimal_bee.position[j] + phi*(donor1.position[j] - donor2.position[j])
            candidate_bee.position[j] = np.clip(candidate_bee.position[j],self.bounds[j][0],self.bounds[j][1])
            
        if self.mutation == 'ABC/best/2':
            # 4 Donor Bees
            donor1,donor2,donor3,donor4 = self.get_donor_bees_(n_donors=4,bee_idx=bee_idx,population=population)
            # Candidate Bee
            j = np.random.randint(0,len(bee.position))
            phi = np.random.uniform(-1,1)
            candidate_bee = copy.deepcopy(bee)
            candidate_bee.position[j] = self.optimal_bee.position[j] + phi*(donor1.position[j] - donor2.position[j]) + phi*(donor3.position[j] - donor4.position[j])
            candidate_bee.position[j] = np.clip(candidate_bee.position[j],self.bounds[j][0],self.bounds[j][1])
        
        return candidate_bee

    def get_donor_bees_(self,n_donors=1,bee_idx=None,population=None):
        available_indices = np.delete(np.arange(len(population)), bee_idx)
        k_list = np.random.choice(available_indices,size=n_donors,replace=False)
        return [copy.deepcopy(population[k]) for k in k_list]
#--------------------------------------------------------------------------------