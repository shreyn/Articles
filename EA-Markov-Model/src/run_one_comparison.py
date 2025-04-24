"""
run_one_comparison.py
---------------------
Compare one run of the evolutionary algorithm simulation against
the Markov model prediction, using the same initial population.

Part of the EA-Markov-Model project.
"""

import matplotlib.pyplot as plt
from markov_model import generate_crossover_transition_tensor, run_markov_process, expected_fitness_from_distribution, compute_fitness_distribution
from sim_blind import EA_Loop, initial_population

#Parameters
populationSize = 500
genomeLength = 10
generations = 40
mutationRate = 0.05
tournamentSize = 2

#Generate crossover tensor P
P = generate_crossover_transition_tensor(genomeLength, mutationRate)

#Run Markov Model
# Generate initial random population like EA simulation
population = initial_population(populationSize, genomeLength)
v0 = compute_fitness_distribution(population, genomeLength)

#Run EA Sim, using same initial pop
sim_avg_fit = EA_Loop(populationSize, genomeLength, generations, tournamentSize, mutationRate, initial_population_override=population)

history = run_markov_process(v0, P, tournament_size=tournamentSize, generations=generations)
expected_fit = expected_fitness_from_distribution(history)


#Plotting both
plt.plot(range(generations), sim_avg_fit, label='EA Simulation (Empirical)', marker='o')
plt.plot(range(generations + 1), expected_fit, label='Markov Model (Expected)', linestyle='--')
plt.xlabel('Generation')
plt.ylabel('Average Fitness')
plt.title('EA Simulation vs Markov Model')
plt.legend()
plt.grid(True)
plt.show()
