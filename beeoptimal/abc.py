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
    
    Attributes:
        colony_size (int)                : The total number of bees in the colony.
        n_employed_bees (int)            : The number of employed bees.
        n_onlooker_bees (int)            : The number of onlooker bees.
        max_scouts (int)                 : The maximum number of scout bees per iteration. Defaults to None (will be set to n_employed_bees).
        dim (int)                        : The dimensionality of the search space.
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
    
    def __init__(self,colony_size,function,bounds,n_employed_bees=None,max_scouts=None):
        """
        Initializes the ABC
        
        Args:
            colony_size (int)               : The total number of bees in the colony.
            function (callable)             : The objective function to optimize.
            bounds (array-like)             : The bounds for each dimension of the search space, provided as a 2D array [(lower1, upper1), ..., (lowerD, upperD)].
            n_employed_bees (int, optional) : The number of employed bees. Defaults to half the total number of bees.
            max_scouts (int, optional)      : The maximum number of scout bees per iteration. Defaults to None (will be set to n_employed_bees).

        Raises:
            TypeError  : If the function is not callable.
            TypeError  : If the bounds are not a numpy array.
            ValueError : If the bounds do not have shape `(D, 2)` or `(2, D)`.
            ValueError : If any dimension has its lower bound greater than the upper bound.
            TypeError  : If the colony size is not an integer.
            ValueError : If the colony size is less than 10.
            TypeError  : If the number of employed bees is not an integer.
            ValueError : If the number of employed bees is less than 5 or it is s.t the number of onlookers is less than 5.
            TypeError  : If the maximum number of scouts is not an integer.
            ValueError : If the maximum number of scouts is less than 0 or greater than the number of employed bees.
        
        .. note::
            Constraints about colony size, n_employed_bees and max_scouts ensure compatibility across all mutation types.
        """
        
        if not callable(function):
            raise TypeError("`function` must be callable.")
        self.function = function
        
        if not isinstance(bounds, np.ndarray):
            raise TypeError("`bounds` must be a numpy array.")
        if not ((bounds.shape[0] == 2) or (bounds.shape[1] == 2)):
            raise ValueError("`bounds` must have shape `(D, 2)` or `(2, D)`, but got {bounds.shape}.") 
        if not np.all(bounds.reshape(-1,2)[:, 0] <= bounds.reshape(-1,2)[:, 1]):
            raise ValueError("Each lower bound must be less than or equal to its upper bound.")
        self.bounds              = bounds.reshape(-1,2)
        self.dim                 = self.bounds.shape[0]
        
        if not isinstance(colony_size,int):
            raise TypeError("`colony_size` must be an integer.")
        if colony_size < 10:
            raise ValueError(f"`colony_size` must be at least 10 to ensure compatibility across all mutation types, but got {colony_size}") 
        self.colony_size         = colony_size
        
        if n_employed_bees is not None:
            if not isinstance(n_employed_bees,int):
                raise TypeError("`n_employed_bees` must be an integer.")
            if not ((n_employed_bees >= 5) and ((self.colony_size-n_employed_bees)>=5)):
                raise ValueError("It must hold 5 <= `n_employed_bees` <= (`colony_size` - 5) to ensure compatibility across all mutation types")
            self.n_employed_bees     = n_employed_bees
        else:
            self.n_employed_bees     = self.colony_size // 2
            
        self.n_onlooker_bees     = self.colony_size - self.n_employed_bees    
        
        if max_scouts is not None:
            if not isinstance(max_scouts,int):
                raise TypeError("`max_scouts` must be an integer.")
            if not ( (max_scouts >= 0) and (max_scouts <= self.n_employed_bees) ):
                raise ValueError(f"`max_scouts` must be beteen 0 and `n_employed_bees`, but got {max_scouts} and {self.n_employed_bees}, respectively.")
            self.max_scouts = max_scouts
        else:
            self.max_scouts = self.n_employed_bees
      
        
        self.employed_bees       = []
        self.onlooker_bees       = []
        self.colony_history      = []
        self.optimal_bee         = None
        self.optimal_bee_history = []
            
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
        
        Raises:
            ValueError: If `max_iters` is not a positive integer.
            ValueError: If `mutation` is not one of ['StandardABC', 'ModifiedABC', 'ABC/best/1', 'ABC/best/2', 'DirectedABC'].
            ValueError: If `initialization` is not one of ['random', 'cahotic'].
            ValueError: If `selection` is not one of ['RouletteWheel', 'Tournament'].
            ValueError: If `mr` is not a float value between 0.0 and 1.0.
            ValueError: If `selection` is 'Tournament' and `tournament_size` is not an integer between 1 and `n_employed_bees`.
            ValueError: If `sf` is not a float value greater than 0.
            ValueError: If `limit` is less than or equal to 0.
            TypeError : If `self_adaptive_sf` is not a boolean.
            TypeError : If `stagnation_tol` is not a float.
        """
    
        # Sanity checks and setting optimization parameters
        #.....................................................................................................................................
        if not (isinstance(max_iters, int) and max_iters >= 0):
            raise ValueError("`max_iters` must be a positive integer.")
        self.max_iters    = max_iters
        self.actual_iters = 0
        
        valid_mutations_ = ['StandardABC', 'ModifiedABC', 'ABC/best/1', 'ABC/best/2', 'DirectedABC']
        if mutation not in valid_mutations_:
            raise ValueError(f"{mutation} is an invalid mutation strategy. Choose one among {', '.join(valid_mutations_)}.")
        self.mutation = mutation
        
        valid_initializations_ = ['random','cahotic']
        if initialization not in valid_initializations_:
            raise ValueError(f"{initialization} is an invalid intialization. Choose one among {', '.join(valid_initializations_)}.")
        self.initialization = initialization
        
        valid_selections_ = ['RouletteWheel','Tournament']
        if selection not in valid_selections_:
            raise ValueError(f"{selection} is an invalid selection. Choose one among {', '.join(valid_selections_)}.")
        self.selection = selection
        
        if not (isinstance(mr,float) and  (0.0 <= mr <= 1.0)):
            raise ValueError("`mr` must be a float value between 0.0 and 1.0")
        self.mr = mr
        
        if self.selection == 'Tournament':
            if tournament_size is None:
                raise ValueError("`Please specify a tournament_size`.")
            if not (isinstance(tournament_size, int) and (1 <= tournament_size <= self.n_employed_bees)):
                raise ValueError("`tournament_size` must be an integer between 1 and `n_employed_bees` when using 'Tournament' selection.")
            self.tournament_size  = tournament_size
        
        if self.selection == 'DirectedABC':
            self.directions       = np.zeros((self.n_employed_bees,self.dim))
        
        if not (isinstance(sf,(int,float)) and (sf>0.0)):
            raise ValueError(f"`sf` must be greater than 0, but got {sf}")
        self.sf               = sf 
        self.initial_sf       = sf
        if not isinstance(self_adaptive_sf,bool):
            raise TypeError("`self_adaptive_sf` must be bool")
        self.self_adaptive_sf = self_adaptive_sf         
        
        self.limit = limit if (limit != 'default') else (0.6 * self.n_employed_bees * self.dim)  
        if not (self.limit>0):
            raise ValueError("`limit` must be greater than 0, but got {self.limit}. If this error occurs when `limit`='default', change your configuration.")
        
        if not isinstance(stagnation_tol,float):
            raise TypeError('`stagnation_tol` must be float')
        self.stagnation_tol   = stagnation_tol
        
        if not isinstance(verbose,bool):
            raise TypeError("`verbose` must be bool")
        
        #.....................................................................................................................................          
        
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
        
        #.....................................................................................................................................
        
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
        #.....................................................................................................................................
    
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
            
            if n_scouts > self.max_scouts:
                break
    
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
    #------------------------------------------------------------------------------------------------------------------
        