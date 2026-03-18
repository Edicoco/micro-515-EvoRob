import numpy as np

# Load the files
x = np.load('results/20260309_171800_neural_controller_ckpts/20/x_best.npy')          # Population of solutions
f = np.load('results/20260309_171800_neural_controller_ckpts/20/f.npy')          # Fitness values of the population
x_best = np.load('results/20260309_171800_neural_controller_ckpts/20/x_best.npy') # Best solution found so far
f_best = np.load('results/20260309_171800_neural_controller_ckpts/20/f_best.npy') # Fitness of the best solution

print(f"Population shape: {x.shape}")
print(f"Best solution shape: {x_best.shape}")
print(f"Best fitness: {f_best}")
print(f"Best solution: {x_best}")