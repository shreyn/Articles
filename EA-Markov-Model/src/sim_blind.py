"""
sim_blind.py
-------------
Implements the baseline evolutionary algorithm (EA) with:
- Tournament selection
- Single-point crossover
- Bit-flip mutation
- Blind generational replacement.

Part of the EA-Markov-Model project.
"""

import random as rand
import matplotlib.pyplot as plt

def random_genome(length):
    genome = []
    for i in range(length):
        bit = 1 if rand.random()<0.5 else 0
        genome.append(bit)
    return genome

def initial_population(populationSize, genomeLength):
    population = []
    for i in range(populationSize):
        population.append(random_genome(genomeLength))
    return population

def fitness_function(genome):
    return sum(genome)

def tournament_selection(population, k):
    competitors = rand.sample(population, k)
    best = competitors[0]
    for individual in competitors[1:]:
        if (fitness_function(individual) > fitness_function(best)):
            best = individual
    return best

def crossover(parent1, parent2):
    point = rand.randint(1, len(parent1)-1)
    return parent1[:point] + parent2[point:]

def mutate(genome, p_m):
    child = []
    for bit in genome:
        if (rand.random() < p_m):
            child.append(1-bit)
        else: child.append(bit)
    return child

def EA_Loop(populationSize, genomeLength, generations, tournamentSize, mutationRate, initial_population_override = None):
    if (initial_population_override != None): population = initial_population_override
    else: population = initial_population(populationSize, genomeLength)
    
    averageFitnessOverTime = []

    for generation in range(generations):
        offspring = []
        for i in range(populationSize):
            parent1 = tournament_selection(population, tournamentSize)
            parent2 = tournament_selection(population, tournamentSize)
            child = mutate(crossover(parent1, parent2), mutationRate) #crossover, then mutation
            offspring.append(child)
        population = offspring #blind replacement of pop. (µ, lambda)

        #stats
        fitnesses = []
        for i in population:
            fitnesses.append(fitness_function(i))
        best = max(fitnesses)
        average = sum(fitnesses) / len(fitnesses)
        averageFitnessOverTime.append(average)
        #print(f'Gen {generation}: Best = {best}, Avg = {average}')
    
    return averageFitnessOverTime

        

    # plt.plot(averageFitnessOverTime, label='Average Fitness')
    # plt.xlabel('Generation')
    # plt.ylabel('Fitness')
    # plt.title('Average Fitness Over Time')
    # plt.grid(True)
    # plt.legend()
    # plt.show()


#EA_Loop(populationSize, genomeLength, generations, tournamentSize, mutationRate)
