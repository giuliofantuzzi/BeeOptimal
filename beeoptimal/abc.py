#++++++++++++++++++++++++++++++++++++
# Libraries and modules
#++++++++++++++++++++++++++++++++++++

import numpy as np
import copy
from .bee import Bee
from tqdm import trange,tqdm

#++++++++++++++++++++++++++++++++++++
# Artificial Bee Colony (ABC) class
#++++++++++++++++++++++++++++++++++++

class ArtificialBeeColony():
    
    """
    Artificial Bee Colony (ABC) class
    
    Args:
        dim (int)                        : The dimensionality of the search space.
        n_bees (int)                     : The total number of bees in the colony.
        n_employed_bees (int)            : The number of employed bees.
        n_onlooker_bees (int)            : The number of onlooker bees.
        max_scouts (int)                 : The maximum number of scout bees per iteration. Defaults to None (will be set to n_employed_bees).
        function (callable)              : The objective function to optimize.
        bounds (array-like)              : The bounds for each dimension of the search space, provided as a 2D array [(lower1, upper1), ..., (lowerD, upperD)].
        employed_bees (list[Bee])        : The employed bees in the colony.
        onlooker_bees (list[Bee])        : The onlooker bees in the colony.
        colony_history (list[list[Bee]]) : The history of the employed bees at each iteration.
        optimal_bee (Bee)                : The optimal bee in the colony.
        optimal_bee_history (list[Bee])  : The history of the optimal bee at each iteration.
        max_iters (int)                  : The maximum number of iterations. Defaults to 1000.
        actual_iters (int)               : The actual number of iterations.
        limit (int)                      : The trial limit for scout bees. If 'default', it is set to 0.6 * n_employed_bees * dimensionality. Defaults to 'default'.
        selection (str, optional)        : The selection strategy for onlooker bees. Must be one among 'RouletteWheel' and 'Tournament'. Defaults to 'RouletteWheel'.
        mutation (str)                   : The mutation strategy. Must be one among 'StandardABC', 'ModifiedABC', 'ABC/best/1', 'ABC/best/2' and 'DirectedABC'. Defaults to 'StandardABC'.
        initialization (str)             : The initialization strategy for the bee population. Must be one among 'random' and 'cahotic'. Defaults to 'random'.
        stagnation_tol (float)           : The tolerance for stagnation in fitness values to trigger early termination. Defaults to np.NINF (i.e. stagnation disabled).
        sf (float)                       : The scaling factor for mutations. Defaults to 1.0.
        initial_sf (float)               : The initial scaling factor. Defaults to 1.0.
        self_adaptive_sf (bool)          : Whether to use a self-adaptive scaling factor. Defaults to False.
        mr (float)                       : The mutation rate for 'ModifiedABC' strategy. Defaults to 0.7.
    
    .. note::
            To ensure compatibility with all the mutation types, the bee colony must have at least 5 employed bees and at least 5 onlokeer bees.
    """
    
    #------------------------------------------------------------------------------------------------------------------
    
    def __init__(self,n_bees,function,bounds,n_employed_bees=None,max_scouts=None):
        """
        Initializes the ABC
        
        Args:
            n_bees (int)                    : The total number of bees in the colony.
            function (callable)             : The objective function to optimize.
            bounds (array-like)             : The bounds for each dimension of the search space, provided as a 2D array [(lower1, upper1), ..., (lowerD, upperD)].
            n_employed_bees (int, optional) : The number of employed bees. Defaults to half the total number of bees.
            max_scouts (int, optional)      : The maximum number of scout bees per iteration. Defaults to None (will be set to n_employed_bees).

        Raises:
            AssertionError: If the number of bees is less than or equal to 10 (for compatibility with ALL mutation types).
            AssertionError: If the number of employed bees is less than 5 (for compatibility with ALL mutation types).
            AssertionError: If the number of onlokeer bees is less than 5 (for compatibility with ALL mutation types).
            AssertionError: If the number of employed bees is greater than or equal to the total number of bees.
        
        .. note::
            - To ensure compatibility with all the mutation types, the bee colony must have at least 5 employed bees and at least 5 onlokeer bees.
        """
        
        assert ( n_bees >= 10) , 'The number of bees must be at least 10 for compatibility with all mutation types'
        if n_employed_bees:
            assert (n_employed_bees >= 5)     , 'The number of employed bees must be at least 5 for compatibility with all mutation types'
            assert (n_employed_bees < n_bees) , 'The number of employed bees must be lower than the number of bees'
    
        self.dim                 = len(bounds)
        self.n_bees              = int(n_bees)
        self.n_employed_bees     = int(n_employed_bees) if n_employed_bees is not None else self.n_bees // 2
        self.n_onlooker_bees     = self.n_bees - self.n_employed_bees
        self.max_scouts          = int(max_scouts) if max_scouts is not None else self.n_employed_bees
        self.function            = function
        self.bounds              = bounds if bounds is not None else np.array([(-1e30,1e30)]*self.dim)
        self.employed_bees       = []
        self.onlooker_bees       = []
        self.colony_history      = []
        self.optimal_bee         = None
        self.optimal_bee_history = []
        
        assert (self.n_onlooker_bees >= 5)               , 'The number of onlooker bees must be at least 5. Please change your configuration'
        assert (self.max_scouts <= self.n_employed_bees) , 'The maximum number of scouts must be lower (or equal) than the number of employed bees'
    
    #------------------------------------------------------------------------------------------------------------------
    
    def optimize(self,
                 max_iters        = 1000,
                 limit            = 'default',
                 selection        = 'RouletteWheel',
                 mutation         = 'StandardABC',
                 initialization   = 'random',
                 tournament_size  = None,
                 stagnation_tol   = np.NINF,
                 sf               = 1.0,
                 self_adaptive_sf = False,
                 mr               = 1.0,
                 verbose          = False,
                 random_seed      = None):
        """
        Runs the optimization process.

        Args:
            max_iters (int, optional)        : The maximum number of iterations. Defaults to 1000.
            limit (int or str, optional)     : The trial limit for scout bees. If 'default', it is set to 0.6 * n_employed_bees * dimensionality. Defaults to 'default'.
            selection (str, optional)        : The selection strategy for onlooker bees. Must be one among 'RouletteWheel' and 'Tournament'. Defaults to 'RouletteWheel'.
            mutation (str, optional)         : The mutation strategy. Must be one among 'StandardABC', 'ModifiedABC', 'ABC/best/1' and 'ABC/best/2'. Defaults to 'StandardABC'.
            initialization (str, optional)   : The initialization strategy for the bee population. Must be one among 'random' and 'cahotic'. Defaults to 'random'.
            tournament_size (int, optional)  : The size of the tournament for the 'Tournament' selection strategy. Defaults to None.
            stagnation_tol (float, optional) : The tolerance for stagnation in fitness values to trigger early termination. Defaults to np.NINF (i.e. stagnation disabled).
            sf (float, optional)             : The scaling factor for mutations. Defaults to 1.0.
            self_adaptive_sf (bool, optional): Whether to use a self-adaptive scaling factor. Defaults to False.
            mr (float, optional)             : The mutation rate for 'ModifiedABC' strategy. Defaults to 1.0.
            verbose (bool, optional)         : Whether to display optimization progress. Defaults to False.
            random_seed (int, optional)      : The seed for random number generation. Defaults to None.
        
        Raise:
            AssertionError: If the number of iterations is less than 2.
            AssertionError: If the trial limit is less than 1.
            AssertionError: If the selection strategy is invalid.
            AssertionError: If the initialization strategy is invalid.
            AssertionError: If the tournament size is invalid.
            
        """
        
        assert (max_iters > 0)                                                                                 , 'The number of iterations must be greater than 0'
        
        assert (mutation in ['StandardABC', 'ModifiedABC', 'ABC/best/1','ABC/best/2','DirectedABC'])           , 'Invalid mutation strategy. Please choose one among StandardABC, ModifiedABC, ABC/best/1, ABC/best/2 and DirectedABC'
        assert (initialization in ['random','cahotic'])                                                        , 'Invalid initialization strategy. Please choose one among random and cahotic'
        assert (selection in ['RouletteWheel','Tournament'])                                                   , 'Invalid selection strategy. Please choose one among RouletteWheel and Tournament'
        assert (mr>=0 and mr<=1)                                                                               , 'Invalid mutation rate. Please choose a value between 0 and 1'
        if selection == 'Tournament':
            assert (tournament_size > 0 and tournament_size <= self.n_employed_bees)  , 'Tournament size must be between 1 and the number of employed bees'
        else:
            assert (tournament_size is None)                                          , 'Tournament size must be None for the selected selection strategy'
        
        self.max_iters        = max_iters
        self.actual_iters     = 0
        self.limit            = limit if limit != 'default' else (0.6 * self.n_employed_bees * self.dim)  
        self.selection        = selection
        self.tournament_size  = tournament_size
        self.mutation         = mutation
        self.initialization   = initialization
        self.mr               = mr
        self.sf               = sf 
        self.initial_sf       = sf
        self.self_adaptive_sf = self_adaptive_sf 
        self.directions       = np.zeros((self.n_employed_bees,self.dim))
        self.stagnation_tol   = stagnation_tol
        
        assert (self.limit > 0) , "The trial limit must be greater than 1. If this error occurs when limit='default', please set it manually to 1"
                
        # Initialization
        if random_seed:
            np.random.seed(random_seed)
            
        if self.initialization == 'random':
            self.employed_bees = [Bee(position = np.random.uniform(self.bounds[:,0],self.bounds[:,1],self.dim),
                                      function = self.function,
                                      bounds   = self.bounds) for _ in range(self.n_employed_bees) ]
        elif self.initialization == 'cahotic':
            # Define cahotic map and iterate over it
            cahotic_map = np.random.rand(self.n_employed_bees,self.dim)
            for _ in range(300):
                cahotic_map = np.sin(cahotic_map * np.pi)    
                
            cahotic_pop  = [Bee(position = self.bounds[:,0] + (self.bounds[:,1] - self.bounds[:,0]) * cahotic_map[i,:],
                                function = self.function,
                                bounds   = self.bounds) for i in range(self.n_employed_bees) ]
            opposite_pop = [Bee(position = self.bounds[:,0] + self.bounds[:,1] - cahotic_pop[i].position,
                                function = self.function,
                                bounds   = self.bounds) for i in range(self.n_employed_bees) ]
            
            self.employed_bees = sorted(cahotic_pop+opposite_pop, key=lambda bee: bee.fitness, reverse=True)[:self.n_employed_bees]
            
        self.colony_history.append(copy.deepcopy(self.employed_bees))
        self.optimal_bee = copy.deepcopy(max(self.employed_bees,key=lambda bee: bee.fitness))
        self.optimal_bee_history.append(copy.deepcopy(self.optimal_bee))
        
        # Optimization Loop
        for _ in trange(self.max_iters,desc='Running Optimization',disable= not verbose,bar_format='{l_bar}{bar}|[{elapsed}<{remaining}]'):
            
            self.actual_iters += 1
            
            self.employees_phase_()
            self.onlookers_phase_()
            self.scouts_phase_()
            
            self.colony_history.append(copy.deepcopy(self.employed_bees))
            self.optimal_bee = copy.deepcopy(max(self.employed_bees,key=lambda bee: bee.fitness))
            self.optimal_bee_history.append(copy.deepcopy(self.optimal_bee))
            
            # Stagnation
            if (np.std([bee.fitness for bee in self.employed_bees]) < self.stagnation_tol):
                if verbose:
                    tqdm.write(f"Early termination: Optimization stagnated at iteration {self.actual_iters} / {self.max_iters}")
                break
    
    #------------------------------------------------------------------------------------------------------------------
    
    def employees_phase_(self):
        """
        Performs the employed bees phase, where each employed bee explores the search space by generating candidate solutions.

        .. note::
            Updates the employed bees with better candidate solutions based on the greedy selection.
            Adjusts the scaling factor if self-adaptive scaling is enabled.
        """
        
        succesful_mutations = 0
        for bee_idx, bee in enumerate(self.employed_bees):
            candidate_bee = self.get_candidate_neighbor_(bee=bee,bee_idx=bee_idx,population=self.employed_bees)
            # Greedy Selection
            if candidate_bee.fitness >= bee.fitness:
                self.employed_bees[bee_idx] = candidate_bee
                succesful_mutations += 1
        
                if self.mutation == 'DirectedABC':
                    if (bee.position != candidate_bee.position).any(): # this is needed when candidate_bee.fitness == bee.fitness
                        # Retrieve the index mutated and update the direction
                        j = np.where(bee.position != candidate_bee.position)[0][0]
                        self.directions[bee_idx,j] = np.sign(candidate_bee.position[j] - bee.position[j])
                
            else:
                bee.trial += 1
                
                if self.mutation == 'DirectedABC':
                    # Retrieve the index mutated and update the direction
                    j = np.where(bee.position != candidate_bee.position)[0][0]
                    self.directions[bee_idx,j] = 0
                
        if self.self_adaptive_sf:
            self.update_SF_(succesful_mutations_ratio= (succesful_mutations / self.n_employed_bees) )
            
    #------------------------------------------------------------------------------------------------------------------        
        
    def waggle_dance_(self):
        """
        Implements the waggle dance, which determines the probability of selecting employed bees for the onlooker phase.

        Returns:
            array: Indices of the selected employed bees (based on the chosen selection strategy).
        """
        
        fitness_values = np.array([bee.fitness for bee in self.employed_bees])
        
        if self.selection == 'RouletteWheel':
            selection_probabilities  = fitness_values / np.sum(fitness_values)
            dance_winners = np.random.choice(range(self.n_employed_bees),size=self.n_onlooker_bees,p=selection_probabilities,replace=True)
            return dance_winners
        
        if self.selection == 'Tournament':
            dance_winners = []
            for _ in range(self.n_onlooker_bees):
                tournament_indices = np.random.choice(range(self.n_employed_bees),size=self.tournament_size,replace=False)
                tournament_fitness = fitness_values[tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                dance_winners.append(winner_idx)
            return dance_winners
    
    #------------------------------------------------------------------------------------------------------------------    
    
    def onlookers_phase_(self):
        """
        Performs the onlooker bees phase, where onlookers exploit the information shared by employed bees to explore the search space.

        .. note::
            Updates employed bees with better solutions discovered by onlookers.
            Adjusts the scaling factor if self-adaptive scaling is enabled.
        """
        
        dance_winners = self.waggle_dance_()
        
        self.onlooker_bees = [Bee(position = self.employed_bees[winner_idx].position,
                                  function = self.function,
                                  bounds   = self.bounds) for winner_idx in dance_winners
                              ]
        
        succesful_mutations = 0
        for bee_idx, winner_idx in enumerate(dance_winners):
            bee = self.onlooker_bees[bee_idx]
            # Get Candidate Neighbor
            candidate_bee = self.get_candidate_neighbor_(bee=bee,bee_idx=bee_idx,population=self.onlooker_bees)
            # Greedy Selection
            if candidate_bee.fitness >= bee.fitness:
                self.employed_bees[winner_idx] = candidate_bee
                succesful_mutations += 1
                
                if self.mutation == 'DirectedABC':
                    if (bee.position != candidate_bee.position).any(): # this is needed when candidate_bee.fitness == bee.fitness
                        # Retrieve the index mutated and update the direction
                        j = np.where(bee.position != candidate_bee.position)[0][0]
                        self.directions[winner_idx,j] = np.sign(candidate_bee.position[j] - bee.position[j])
                
            else:
                
                if self.mutation == 'DirectedABC':
                    # Retrieve the index mutated and update the direction
                    j = np.where(bee.position != candidate_bee.position)[0][0]
                    self.directions[winner_idx,j] = 0
                
        if self.self_adaptive_sf:
            self.update_SF_(succesful_mutations_ratio= (succesful_mutations / self.n_onlooker_bees) )
            
    #------------------------------------------------------------------------------------------------------------------
    
    def scouts_phase_(self):
        """
        Performs the scout bees phase, where employed bees that exceed the trial limit are forced to explore a new solution

        .. note::
            Depending on the initialization strategy, scouts are reinitialized either randomly or using a chaotic map.
        """
        n_scouts = 0
        
        for bee_idx, bee in enumerate(self.employed_bees):
            if bee.trial > self.limit:
                n_scouts += 1
                if self.initialization == 'random':
                    self.employed_bees[bee_idx] = Bee(position = np.random.uniform(self.bounds[:,0],self.bounds[:,1],self.dim),
                                                      function = self.function,
                                                      bounds   = self.bounds)
                elif self.initialization == 'cahotic':
                    cahotic_map = np.random.rand(self.n_employed_bees,self.dim)
                    for _ in range(300):
                        cahotic_map = np.sin(cahotic_map * np.pi)    
                        
                    cahotic_pop  = [Bee(position = self.bounds[:,0] + (self.bounds[:,1] - self.bounds[:,0]) * cahotic_map[i,:],
                                        function = self.function,
                                        bounds   = self.bounds) for i in range(self.n_employed_bees) ]
                    opposite_pop = [Bee(position = self.bounds[:,0] + self.bounds[:,1] - cahotic_pop[i].position,
                                        function = self.function,
                                        bounds   = self.bounds) for i in range(self.n_employed_bees) ]
            
                    self.employed_bees[bee_idx] = sorted(cahotic_pop+opposite_pop,
                                                         key=lambda bee: bee.fitness, reverse=True)[0]
            
            if n_scouts >= self.max_scouts:
                break 
    
    #------------------------------------------------------------------------------------------------------------------            
    
    def get_candidate_neighbor_(self,bee,bee_idx,population):
        """
        Generates a candidate neighbor solution for a given bee based on the chosen mutation strategy.

        Parameters:
            bee (Bee)         : The bee for which a candidate neighbor is generated.
            bee_idx (int)     : The index of the bee in the population.
            population (list) : The population of bees from which donors are selected (i.e., employed or onlooker bees).

        Returns:
            Bee: A new candidate bee solution.
        """
        
        if self.mutation == 'StandardABC':
            phi = np.random.uniform(-self.sf,self.sf)
            donor_bee = self.get_donor_bees_(n_donors=1,bee_idx=bee_idx,population=population)[0]
            candidate_bee = copy.deepcopy(bee)
            j = np.random.randint(0,self.dim)
            candidate_bee.position[j] = bee.position[j] + phi*(bee.position[j] - donor_bee.position[j])
            candidate_bee.position[j] = np.clip(candidate_bee.position[j],self.bounds[j][0],self.bounds[j][1])
            
        if self.mutation == 'ModifiedABC':
            donor_bee = self.get_donor_bees_(n_donors=1,bee_idx=bee_idx,population=population)[0]
            candidate_bee = copy.deepcopy(bee)
            phi = np.random.uniform(-self.sf,self.sf,self.dim)
            mutation_mask = np.random.uniform(size=self.dim) <= self.mr
            candidate_bee.position[mutation_mask] = bee.position[mutation_mask] + phi[mutation_mask] * (bee.position[mutation_mask] - donor_bee.position[mutation_mask])
            candidate_bee.position = np.clip(candidate_bee.position,self.bounds[:,0],self.bounds[:,1])
            
        if self.mutation == 'ABC/best/1':
            phi = np.random.uniform(-self.sf,self.sf)
            donor1,donor2 = self.get_donor_bees_(n_donors=2,bee_idx=bee_idx,population=population)
            candidate_bee = copy.deepcopy(bee)
            j = np.random.randint(0,self.dim)
            candidate_bee.position[j] = self.optimal_bee.position[j] + phi*(donor1.position[j] - donor2.position[j])
            candidate_bee.position[j] = np.clip(candidate_bee.position[j],self.bounds[j][0],self.bounds[j][1])
            
        if self.mutation == 'ABC/best/2':
            phi = np.random.uniform(-self.sf,self.sf)
            donor1,donor2,donor3,donor4 = self.get_donor_bees_(n_donors=4,bee_idx=bee_idx,population=population)
            candidate_bee = copy.deepcopy(bee)
            j = np.random.randint(0,self.dim)
            candidate_bee.position[j] = self.optimal_bee.position[j] + phi*(donor1.position[j] - donor2.position[j]) \
                                        + phi*(donor3.position[j] - donor4.position[j])
            candidate_bee.position[j] = np.clip(candidate_bee.position[j],self.bounds[j][0],self.bounds[j][1])
        
        if self.mutation == 'DirectedABC':
            donor_bee = self.get_donor_bees_(n_donors=1,bee_idx=bee_idx,population=population)[0]
            candidate_bee = copy.deepcopy(bee)
            directions = self.directions[bee_idx,:]
            j = np.random.randint(0,self.dim)
            r = (directions[j] == 0)  * np.random.uniform(-self.sf,self.sf) + \
                (directions[j] == 1)  * np.random.uniform(0,self.sf)        + \
                (directions[j] == -1) * np.random.uniform(-self.sf,0)
                
            candidate_bee.position[j] = bee.position[j] + r * np.abs(bee.position[j] - donor_bee.position[j])
            candidate_bee.position[j] = np.clip(candidate_bee.position[j],self.bounds[j][0],self.bounds[j][1])

        return candidate_bee
    
    #------------------------------------------------------------------------------------------------------------------
    
    def get_donor_bees_(self,n_donors=1,bee_idx=None,population=None):
        """
        Selects donor bees from the population (for a given bee)

        Args:
            n_donors (int, optional)         : The number of donor bees to return. Defaults to 1.
            bee_idx (int, optional)          : The index of the bee for which donors are selected. Defaults to None.
            population (list[Bee], optional) : The population of bees from which donors are selected. Defaults to None.

        Returns:
            list: A list of donor bees (as Bee instances).
        """
        
        available_indices = np.delete(np.arange(len(population)), bee_idx)
        k_list = np.random.choice(available_indices,size=n_donors,replace=False)
        return [copy.deepcopy(population[k]) for k in k_list]
    
    #------------------------------------------------------------------------------------------------------------------
    
    def update_SF_(self,succesful_mutations_ratio):
        """
        Updates the scaling factor (SF) adaptively based on the ratio of successful mutations
        
        Args:
            successful_mutations_ratio(float): The ratio of successful mutations used to update the scaling factor.
            
        .. note::
            Self-adaptiveness is based on the "one-fifth" rule (Rechenberg 1971)
        """
        
        if succesful_mutations_ratio > 1/5:
            self.sf = self.sf / 0.85
        elif succesful_mutations_ratio < 1/5:
            self.sf = self.sf * 0.85
    
    #------------------------------------------------------------------------------------------------------------------
    def reset(self):
        """
        Resets the ABC object to its initial state.
        """
        self.employed_bees       = []
        self.onlooker_bees       = []
        self.colony_history      = []
        self.optimal_bee         = None
        self.optimal_bee_history = []
        self.actual_iters        = 0
        self.sf                  = self.initial_sf
        self.directions          = np.zeros((self.n_employed_bees,self.dim))
    #------------------------------------------------------------------------------------------------------------------
        