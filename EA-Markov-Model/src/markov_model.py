"""
markov_model.py
---------------
Implements the Markov chain model for evolutionary algorithms with:
- Fitness distribution
- Tournament-based parent selection distribution
- Crossover transition tensor generation
- Markov process updates

Part of the EA-Markov-Model project.
"""

import random as rand
from sim_blind import crossover, mutate, fitness_function, initial_population

### population fitness distribution
def compute_fitness_distribution(population, genomeLength):
    populationSize = len(population)
    v = [0.0 for _ in range(genomeLength + 1)]  # fitness levels 0 to n
    for genome in population:
        fitness = fitness_function(genome)
        v[fitness] += 1 # Count how many individuals have each fitness

    for j in range(genomeLength + 1):
        v[j] /= populationSize # Normalize

    return v  # v[j] is the fraction of population at fitness j

##Rho (parent pair) distribution
def compute_parentpair_dist(v, tournament_size):
    n = len(v) - 1  # fitness levels 0 to n

    #Compute cumulative sums (prefix sums)
    cumsum_v = [0.0 for _ in range(n + 1)]
    running_sum = 0.0
    for i in range(n + 1):
        running_sum += v[i]
        cumsum_v[i] = running_sum

    #Compute single parent selection probabilities (rho_single[i])
    rho_single = [0.0 for _ in range(n + 1)]
    for i in range(n + 1):
        prob_leq_i = cumsum_v[i]
        prob_leq_i_minus_1 = cumsum_v[i - 1] if i > 0 else 0.0
        rho_single[i] = (prob_leq_i ** tournament_size) - (prob_leq_i_minus_1 ** tournament_size)

    #Compute parent-pair distribution 
    rho_pair = [[0.0 for _ in range(n + 1)] for _ in range(n + 1)]
    for i1 in range(n + 1):
        for i2 in range(n + 1):
            rho_pair[i1][i2] = rho_single[i1] * rho_single[i2]

    return rho_pair  # returns (n+1) x (n+1)

#### P tensor generation
def generate_genome_with_fitness(fitness_level, genome_length):
    genome = [1]*fitness_level + [0]*(genome_length - fitness_level)
    rand.shuffle(genome)
    return genome

def generate_crossover_transition_tensor(genomeLength, mutationRate, trials_per_pair=100):
    P = [[[0 for _ in range(genomeLength + 1)] for _ in range(genomeLength + 1)] for _ in range(genomeLength + 1)]

    for i1 in range(genomeLength + 1): #parent 1
        for i2 in range(genomeLength + 1): #parent 2
            count = [0 for _ in range(genomeLength + 1)] #each fitness
            for _ in range(trials_per_pair): #repeat over large number of trials
                parent1 = generate_genome_with_fitness(i1, genomeLength)
                parent2 = generate_genome_with_fitness(i2, genomeLength)
                child = crossover(parent1, parent2)
                child = mutate(child, mutationRate)
                f = fitness_function(child)
                count[f] += 1 #increase the count for that fitness
            total = sum(count)
            if total > 0:
                for j in range(genomeLength + 1):
                    P[i1][i2][j] = count[j] / total #normalize the fitness (prob distribution)
    return P

#P = generate_crossover_transition_tensor(genomeLength=10, mutationRate=0.05)


##### Markov Chain
def markov_update(v, P, tournament_size): #one step
    n = len(v) - 1  # genome length

    #Compute rho (parent pair distribution)
    rho = compute_parentpair_dist(v, tournament_size)

    #Compute the offspring distribution (next generation v)
    v_next = [0.0 for _ in range(n + 1)]
    for j in range(n + 1):  # offspring fitness level
        total = 0.0
        for i1 in range(n + 1):  # parent 1 fitness
            for i2 in range(n + 1):  # parent 2 fitness
                total += rho[i1][i2] * P[i1][i2][j]
        v_next[j] = total

    total_sum = sum(v_next)
    if total_sum > 0.0:
        v_next = [x / total_sum for x in v_next]

    return v_next  # returns the next generation fitness distribution

def run_markov_process(v_initial, P, tournament_size, generations):
    v = v_initial
    history = [v.copy()]
    for t in range(generations):
        v = markov_update(v, P, tournament_size)
        history.append(v.copy())
    return history  # list of v over time

def expected_fitness_from_distribution(v_history):
    expected_fitness = []
    n = len(v_history[0]) - 1  #genome length
    for v in v_history:
        avg_fit = sum(j * v[j] for j in range(n + 1))
        expected_fitness.append(avg_fit)
    return expected_fitness

