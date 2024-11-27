from ABC.abc import ArtificialBeeColony

def objective_function(x):
    return sum([x_i for x_i in x])


n_bees      = 30
limit       = 3
max_iters   = 1000
lower_bound = [-6,-4]
upper_bound = [7,10]


ABC = ArtificialBeeColony(n_bees=30,limit=3,max_iters=1000,function=objective_function,lower_bound=[-6,-4],upper_bound=[5,5])
ABC.optimize()

print('-'*40)
print(f"Objective function: f(x) = sum(x)")
print(f"Lower Bounds: {lower_bound}")
print(f"Upper Bounds: {upper_bound}")
print('-'*40)
print(f"Optimal solution {ABC.optimal_source[0]}")
print(f"Optimal value {ABC.optimal_source[1]}")
print('-'*40)