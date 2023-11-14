import torch
import torch.nn as nn
import numpy as np
from deap import base, creator, tools, algorithms

# Definir el modelo de PyTorch
class MiModelo(nn.Module):
    def __init__(self):
        super(MiModelo, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

# Crear la función de aptitud
def evaluate(individual):
    input_data = torch.tensor(individual, dtype=torch.float)
    target = torch.tensor([5.0], dtype=torch.float)  # Objetivo a alcanzar (ajusta según tu problema)

    modelo = MiModelo()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(modelo.parameters(), lr=0.01)

    for _ in range(1):  # Número de épocas de entrenamiento (ajusta según tu problema)
        optimizer.zero_grad()
        output = modelo(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Calcular la calidad de la solución (en este caso, el valor de la función de pérdida)
    calidad_solucion = loss.item()

    return calidad_solucion,

# Configurar DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimizar la función de aptitud
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.rand)  # Genes aleatorios entre 0 y 1
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

if __name__ == '__main__':
    # Crear la población inicial
    population = toolbox.population(n=50)

    # Estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # Pareto Hall of Fame (para conservar las mejores soluciones)
    hof = tools.ParetoFront()

    # Ejecutar el algoritmo evolutivo
    algorithms.eaMuPlusLambda(population, toolbox, mu=50, lambda_=200, cxpb=0.7, mutpb=0.2, ngen=100,
                             stats=stats, halloffame=hof)

    # Imprimir las mejores soluciones en el frente de Pareto
    print("Mejores soluciones en el frente de Pareto:")
    for ind in hof:
        print(ind.fitness.values, ind)

