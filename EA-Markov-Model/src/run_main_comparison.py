"""
run_main_comparison.py
----------------------
Compare multiple independent EA simulation runs to the Markov model prediction.
Calculates RMSE and R² between the simulation mean and the Markov model mean.

Part of the EA-Markov-Model project.
"""

import matplotlib.pyplot as plt
import numpy as np
from markov_model import (
    generate_crossover_transition_tensor, 
    run_markov_process, 
    expected_fitness_from_distribution, 
    compute_fitness_distribution
)
from sim_blind import EA_Loop, initial_population


#Parameters
populationSize = 500
genomeLength = 10
generations = 30
mutationRate = 0.05
tournamentSize = 2
num_runs = 100  # number of EA runs 

#Crossover Tensor
P = generate_crossover_transition_tensor(genomeLength, mutationRate)

#Initial population (shared between Markov and all EA runs)
population = initial_population(populationSize, genomeLength)
v0 = compute_fitness_distribution(population, genomeLength)

#Run markov model
history = run_markov_process(v0, P, tournament_size=tournamentSize, generations=generations)
expected_fit = expected_fitness_from_distribution(history)

#Run multiple EA simulations
all_sim_runs = []
for _ in range(num_runs):
    sim_avg_fit = EA_Loop(
        populationSize, 
        genomeLength, 
        generations, 
        tournamentSize, 
        mutationRate, 
    )
    all_sim_runs.append(sim_avg_fit)

all_sim_runs = np.array(all_sim_runs)
mean_fit = np.mean(all_sim_runs, axis=0)
std_fit = np.std(all_sim_runs, axis=0)

#Compute RMSE and R^2 between model and simulation 
min_length = min(len(expected_fit), len(mean_fit))  # match lengths
rmse = np.sqrt(np.mean((mean_fit[:min_length] - expected_fit[:min_length])**2))
ss_res = np.sum((mean_fit[:min_length] - expected_fit[:min_length])**2)
ss_tot = np.sum((mean_fit[:min_length] - np.mean(mean_fit[:min_length]))**2)
r2 = 1 - (ss_res / ss_tot)

print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

#Plotting
plt.figure(figsize=(8, 6))
plt.plot(range(generations), mean_fit, label='EA Simulation (Mean)', color='blue', marker='o')
plt.plot(range(generations + 1), expected_fit, label='Markov Model (Expected)', linestyle='--', color='orange')
plt.xlabel('Generation')
plt.ylabel('Average Fitness')
plt.title('EA Simulation vs Markov Model\n(RMSE: {:.3f}, R²: {:.3f})'.format(rmse, r2))
plt.legend()
plt.grid(True)
plt.show()
