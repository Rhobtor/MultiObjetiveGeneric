import deap
from deap import base, creator, tools, algorithms
import numpy as np
from graphRouting import PatrollingGraphRoutingProblem, mutation_operation, cross_operation_between_paths, create_multiagent_random_paths_from_nodes
from copy import deepcopy
""" Optimization of a function with DEAP. """
navigation_map = np.genfromtxt('map.txt', delimiter=' ')
importance_map= importance_map = [ np.genfromtxt('map_interested1.txt', delimiter=' '), 
                   np.genfromtxt('map_interested2.txt', delimiter=' '),
                   np.genfromtxt('map_interested3.txt', delimiter=' ') ]
N_agents = 4
initial_positions = np.array([10,20,30,40])[:N_agents]
scale = 3
mut_gen_prob = 0.1

environment = PatrollingGraphRoutingProblem(navigation_map = navigation_map,
                                                importance_map=importance_map,
												n_agents=N_agents, 
												initial_positions=initial_positions,
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
    
    reward=[]
    # Apply a death penalty when the path is too long.
    for agent_id in individual.keys():
        path = individual[agent_id]
        # reward = reward_pre[agent_id]
        distance = sum([environment.G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)])
        if distance > environment.max_distance:
            reward = 10000
            distance = 10000
    
    print(reward_pre)
    return reward_pre, distance, 

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
    new_ind.fitness.values = 0.0, 0.0

    return new_ind


def plot_frente():
    """
    Representación del frente de Pareto que hemos obtenido
    """
    datos_pareto = np.loadtxt("fitnessmultioptimize_deap.txt", delimiter=",")    
    plt.scatter(datos_pareto[:, 0], datos_pareto[:, 1], s=30)    
    
    # obtenermos el Pareto óptimo
    with open("zdt1_front.json") as optimal_front_data:
        pareto_optimo = np.array(json.load(optimal_front_data))
    plt.scatter(pareto_optimo[:, 0], pareto_optimo[:, 1], 
                s=10, alpha=0.4)
    plt.xlabel("FZDT11")
    plt.ylabel("FZDT12")
    plt.grid(True)
    plt.legend(["Pareto obtenido","Pareto óptimo"], loc="upper right")
    plt.savefig("ParetoBenchmark.pdf", dpi=300, bbox_inches="tight") 



""" Create the DEAP environment. """
creator.create("FitnessMax", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", dict, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", initDict, creator.Individual, environment.G, initial_positions, 150)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", crossover)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selNSGA2)


if __name__ == '__main__':

    """ Run the optimization. """

    # Create the population.
    pop = toolbox.population(n=100)
    
    # Evaluate the entire population.
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    

    # Create the statistics and hall of fame.
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    hof = tools.ParetoFront(similar=similar_paths)

    # Parameters for the optimization.
    NGEN = 1
    MU = 200
    LAMBDA = 200
    CXPB = 0.6
    MUTPB = 0.3

    # Run the optimization.
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats, halloffame=hof)


    res_individuos = open("individuosmultioptimize_deap.txt", "w")
    res_fitness = open("fitnessmultioptimize_deap.txt", "w")
    test = open("test.txt", "w")
    for ind in hof:
        
        res_individuos.write(str(ind))
        res_individuos.write("\n")
        res_fitness.write(str([ind.fitness.values]))
        res_fitness.write("\n")
        test.write(str(hof))
        test.write("\n")
    res_fitness.close()
    res_individuos.close()



    # Plot the Pareto front.

    import matplotlib.pyplot as plt

    front = np.array([ind.fitness.values for ind in hof])

    plt.scatter(front[:,0], front[:,1], c="b")

    plt.axis("tight")
    plt.show()

    # Evaluate the best individual.
    best_ind = hof[0]

    environment.evaluate_path(best_ind, render=True)

    plt.show()



