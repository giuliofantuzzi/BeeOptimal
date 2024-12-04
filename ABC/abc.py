#++++++++++++++++++++++++++++++++++++
# Libraries and modules
#++++++++++++++++++++++++++++++++++++

import numpy as np
import copy
from .bee import Bee
from tqdm import trange

#++++++++++++++++++++++++++++++++++++
# Artificial Bee Colony (ABC) class
#++++++++++++++++++++++++++++++++++++

class ArtificialBeeColony():
    
    #------------------------------------------------------------------------------------------------------------------
    def __init__(self,n_bees,function,bounds,n_employed_bees=None):
        assert ( n_bees > 2)
        self.dim                 = len(bounds)
        self.n_bees              = n_bees
        self.n_employed_bees     = n_employed_bees if n_employed_bees is not None else (self.n_bees // 2)
        self.n_onlooker_bees     = self.n_bees - self.n_employed_bees
        self.function            = function
        self.bounds              = bounds
        self.lower_bounds        = np.array([bound[0] for bound in bounds])
        self.upper_bounds        = np.array([bound[1] for bound in bounds])
        self.employed_bees       = []
        self.onlooker_bees       = []
        self.colony_history      = []
        self.optimal_bee         = []
        self.optimal_bee_history = []
    
    #------------------------------------------------------------------------------------------------------------------
    def optimize(self,
                 max_iters      = 100,
                 limit          = 'default',
                 selection      = 'RouletteWheel',
                 mutation       = 'StandardABC',
                 initialization = 'random',
                 SF             = 1.0,
                 SelfAdaptiveSF = False,
                 MR             = 0.8,
                 verbose        = False,
                 random_seed    = None):
        
        # Define optimization attributes
        self.max_iters      = max_iters
        self.limit          = limit if limit != 'default' else (0.6 * self.n_employed_bees * self.dim)  
        self.selection      = selection
        self.mutation       = mutation
        self.initialization = initialization
        self.MR             = MR
        self.SF             = SF 
        self.SelfAdaptiveSF = SelfAdaptiveSF 
        
        # Initialization
        if random_seed:
            np.random.seed(random_seed)
            
        if self.initialization == 'random':
            self.employed_bees = [Bee(position = [np.random.uniform(bound[0],bound[1]) for bound in self.bounds],
                                      function = self.function,
                                      bounds   = self.bounds) for _ in range(self.n_employed_bees) ]
        elif self.initialization == 'cahotic':
            # Define cahotic map and iterate over it
            cahotic_map = np.random.rand(self.n_employed_bees,self.dim)
            for _ in range(300):
                cahotic_map = np.sin(cahotic_map * np.pi)    
                
            cahotic_pop  = [Bee(position = self.lower_bounds + (self.upper_bounds - self.lower_bounds) * cahotic_map[i,:],
                                function = self.function,
                                bounds   = self.bounds) for i in range(self.n_employed_bees) ]
            opposite_pop = [Bee(position = self.lower_bounds + self.upper_bounds - cahotic_pop[i].position,
                                function = self.function,
                                bounds   = self.bounds) for i in range(self.n_employed_bees) ]
            
            self.employed_bees = sorted(cahotic_pop+opposite_pop, key=lambda bee: bee.fitness, reverse=True)[:self.n_employed_bees]
            
            
        
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
    
    #------------------------------------------------------------------------------------------------------------------
    def send_employees_(self):
        succesful_mutations = 0
        for bee_idx, bee in enumerate(self.employed_bees):
            candidate_bee = self.get_candidate_neighbor_(bee=bee,bee_idx=bee_idx,population=self.employed_bees)
            # Greedy Selection
            if candidate_bee.fitness >= bee.fitness:
                self.employed_bees[bee_idx] = candidate_bee
                succesful_mutations += 1
            else:
                bee.trial += 1
        if self.SelfAdaptiveSF:
            self.update_SF_(succesful_mutations_ratio= (succesful_mutations / self.n_employed_bees) )
    #------------------------------------------------------------------------------------------------------------------            
    def waggle_dance_(self):
        fitness_values = [bee.fitness for bee in self.employed_bees]
        if self.selection == 'RouletteWheel':
            selection_probabilities  = [fitness/sum(fitness_values) for fitness in fitness_values]
            dance_winners = np.random.choice(range(self.n_employed_bees),size=self.n_onlooker_bees,p=selection_probabilities)
            return dance_winners
    #------------------------------------------------------------------------------------------------------------------    
    def send_onlookers_(self):
        dance_winners = self.waggle_dance_()
        self.onlooker_bees = [Bee(position = self.employed_bees[winner_idx].position,
                                  function = self.function,
                                  bounds   = self.bounds) for winner_idx in dance_winners
                              ]
        succesful_mutations = 0
        for bee_idx, bee in enumerate(self.onlooker_bees):
            # Get Candidate Neighbor
            candidate_bee = self.get_candidate_neighbor_(bee=bee,bee_idx=bee_idx,population=self.onlooker_bees)
            # Greedy Selection
            if candidate_bee.fitness >= bee.fitness:
                self.onlooker_bees[bee_idx] = candidate_bee
                succesful_mutations += 1
        if self.SelfAdaptiveSF:
            self.update_SF_(succesful_mutations_ratio= (succesful_mutations / self.n_onlooker_bees) )
        
        # Update Employed Bees
        for bee_idx, bee in zip(dance_winners,self.onlooker_bees):
            if bee.fitness > self.employed_bees[bee_idx].fitness:
                self.employed_bees[bee_idx] = bee
    #------------------------------------------------------------------------------------------------------------------
    def send_scouts_(self):
        for bee_idx, bee in enumerate(self.employed_bees):
            if bee.trial > self.limit:
                if self.initialization == 'random':
                    self.employed_bees[bee_idx] = Bee(position = [np.random.uniform(bound[0],bound[1]) for bound in self.bounds],
                                                      function = self.function,
                                                      bounds   = self.bounds)
                elif self.initialization == 'cahotic':
                    cahotic_map = np.random.rand(self.n_employed_bees,self.dim)
                    for _ in range(300):
                        cahotic_map = np.sin(cahotic_map * np.pi)    
                    cahotic_pop  = [Bee(position = self.lower_bounds + (self.upper_bounds - self.lower_bounds) * cahotic_map[i,:],
                                        function = self.function,
                                        bounds   = self.bounds) for i in range(self.n_employed_bees) ]
                    opposite_pop = [Bee(position = self.lower_bounds + self.upper_bounds - cahotic_pop[i].position,
                                        function = self.function,
                                        bounds   = self.bounds) for i in range(self.n_employed_bees) ]
            
                    self.employed_bees = sorted(cahotic_pop+opposite_pop, 
                                                key=lambda bee: bee.fitness, reverse=True)[:self.n_employed_bees]
    #------------------------------------------------------------------------------------------------------------------            
    def get_candidate_neighbor_(self,bee,bee_idx,population):
        
        phi = np.random.uniform(-self.SF,self.SF)
        
        if self.mutation == 'StandardABC':
            donor_bee = self.get_donor_bees_(n_donors=1,bee_idx=bee_idx,population=population)[0]
            candidate_bee = copy.deepcopy(bee)
            j = np.random.randint(0,self.dim)
            candidate_bee.position[j] = bee.position[j] + phi*(bee.position[j] - donor_bee.position[j])
            candidate_bee.position[j] = np.clip(candidate_bee.position[j],self.bounds[j][0],self.bounds[j][1])
            
        if self.mutation == 'ModifiedABC':
            donor_bee = self.get_donor_bees_(n_donors=1,bee_idx=bee_idx,population=population)[0]
            candidate_bee = copy.deepcopy(bee)
            for j in range(self.dim):
                if np.random.uniform() <= self.MR:
                    candidate_bee.position[j] = bee.position[j] + phi*(bee.position[j] - donor_bee.position[j])
                    candidate_bee.position[j] = np.clip(candidate_bee.position[j],self.bounds[j][0],self.bounds[j][1])
            
        if self.mutation == 'ABC/best/1':
            donor1,donor2 = self.get_donor_bees_(n_donors=2,bee_idx=bee_idx,population=population)
            candidate_bee = copy.deepcopy(bee)
            j = np.random.randint(0,self.dim)
            candidate_bee.position[j] = self.optimal_bee.position[j] + phi*(donor1.position[j] - donor2.position[j])
            candidate_bee.position[j] = np.clip(candidate_bee.position[j],self.bounds[j][0],self.bounds[j][1])
            
        if self.mutation == 'ABC/best/2':
            donor1,donor2,donor3,donor4 = self.get_donor_bees_(n_donors=4,bee_idx=bee_idx,population=population)
            candidate_bee = copy.deepcopy(bee)
            j = np.random.randint(0,self.dim)
            candidate_bee.position[j] = self.optimal_bee.position[j] + phi*(donor1.position[j] - donor2.position[j]) \
                                        + phi*(donor3.position[j] - donor4.position[j])
            candidate_bee.position[j] = np.clip(candidate_bee.position[j],self.bounds[j][0],self.bounds[j][1])
        
        return candidate_bee
    #------------------------------------------------------------------------------------------------------------------
    def get_donor_bees_(self,n_donors=1,bee_idx=None,population=None):
        available_indices = np.delete(np.arange(len(population)), bee_idx)
        k_list = np.random.choice(available_indices,size=n_donors,replace=False)
        return [copy.deepcopy(population[k]) for k in k_list]
    #------------------------------------------------------------------------------------------------------------------
    def update_SF_(self,succesful_mutations_ratio):
        if succesful_mutations_ratio > 1/5:
            self.SF = self.SF / 0.85
        elif succesful_mutations_ratio < 1/5:
            self.SF = self.SF * 0.85
    #------------------------------------------------------------------------------------------------------------------
        
