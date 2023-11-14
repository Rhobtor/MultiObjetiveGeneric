import pygad
import numpy as np
from graphRouting import PatrollingGraphRoutingProblem, mutation_operation, cross_operation_between_paths, create_multiagent_random_paths_from_nodes
from copy import deepcopy

""" Optimization of a function with DEAP. """
navigation_map = np.genfromtxt('map.txt', delimiter=' ')
importance_map= importance_map = [ np.genfromtxt('interest_map1.txt', delimiter=' '), 
                   np.genfromtxt('interest_map2.txt', delimiter=' '),
                   np.genfromtxt('interest_map3.txt', delimiter=' ') ]
# importance_map= importance_map = np.genfromtxt('map_interested1.txt', delimiter=' ')

N_agents = 3
initial_positions = np.array([10,20,30,40])[:N_agents]
final_positions = np.array([10,20,30,40])[:N_agents]
scale = 3
mut_gen_prob = 1

environment = PatrollingGraphRoutingProblem(navigation_map = navigation_map,
                                                importance_map=importance_map,
												n_agents=N_agents, 
												initial_positions=initial_positions,
                                                final_positions=final_positions,
												scale=scale,
												max_distance=350.0,
												ground_truth='shekel',

	)


def similar_paths(path1, path2):

    for i in path1.keys():

        if path1[i] != path2[i]:
            return False

    return True


def evaluate(individual):
    """ Evaluate an individual. """
    
    reward_pre = environment.evaluate_path(individual, render=False)
    
    reward_p=tuple(reward_pre)
    
    
    # Apply a death penalty when the path is too long.
    for agent_id in individual.keys():
        path = individual[agent_id]
        # reward = reward_pre[agent_id]
        distance = sum([environment.G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)])
        if distance > environment.max_distance:
            reward = 10000
            distance = 10000
    
    #print(len(reward_p))

    return tuple(list(reward_p)+ [distance])

def mutate(individual):
    """ Mutate all individuals. """
    new_individual = deepcopy(individual)

    for i in individual.keys():
        new_individual[i] = mutation_operation(environment.G, individual[i], mut_gen_prob)

    return new_individual,

def crossover(ind1, ind2):

    new_ind1 = deepcopy(ind1)
    new_ind2 = deepcopy(ind2)

    for i in ind1.keys():
        new_ind1[i], new_ind2[i] = cross_operation_between_paths(environment.G, ind1[i], ind2[i])

    return new_ind1, new_ind2

def initDict(container, G, initial_positions, distance):

    new_ind = container(create_multiagent_random_paths_from_nodes(G, initial_positions, distance))
    new_ind.fitness.values = 0.0, 0.0, 0.0 , 0.0
    return new_ind






num_generations = 100 # Number of generations.
num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool.

sol_per_pop = 50 # Number of solutions in the population.
num_genes = 4

last_fitness = 0

def on_generation(ga_instance):
    global last_fitness
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
    print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}")
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       fitness_func=evaluate,
                       on_generation=on_generation,
                       mutation_probability=1)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

ga_instance.plot_fitness()

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

prediction = numpy.sum(numpy.array(solution))
print(f"Predicted output based on the best solution : {prediction}")

if ga_instance.best_solution_generation != -1:
    print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

# Saving the GA instance.
filename = 'genetic' # The filename to which the instance is saved. The name is without extension.
ga_instance.save(filename=filename)

# Loading the saved GA instance.
loaded_ga_instance = pygad.load(filename=filename)
loaded_ga_instance.plot_fitness()