import matplotlib.pyplot as plt
import numpy as np


# datos_pareto = np.loadtxt("fitnessmultioptimize_deap.txt", delimiter=",")    
# plt.scatter(datos_pareto[:, 0], datos_pareto[:, 1], s=30)
# plt.axis("tight")
# plt.show()

import re
import numpy as np

# Lee los datos del archivo como una lista de cadenas
with open("fitnessmultioptimize_deap.txt", "r") as file:
    lines = file.readlines()

# Función para procesar una cadena y extraer los valores como números de punto flotante
def procesar_linea(linea):
    valores = re.findall(r"[-+]?\d*\.\d+|\d+", linea)  # Encuentra todos los números en la línea
    valores_float = [float(valor) for valor in valores]  # Convierte los valores en punto flotante
    return valores_float

# Procesa todas las líneas de datos y rellena con ceros si es necesario
datos_limpios = [procesar_linea(linea) for linea in lines]
max_length = max(len(row) for row in datos_limpios)
datos_limpios = [row + [0.0] * (max_length - len(row)) for row in datos_limpios]

# Convierte la lista de listas en un arreglo NumPy
data = np.array(datos_limpios)

# Guarda los datos limpios en un archivo de texto
np.savetxt("datos_limpios.txt", data, delimiter=",", fmt='%.6f')

datos_pareto = np.loadtxt("datos_limpios.txt", delimiter=",")    
# plt.scatter(datos_pareto[:, 1], datos_pareto[:, 4], s=30)
# plt.axis("tight")
# plt.show()
# Crea cuatro subgráficos en una sola figura
fig, axs = plt.subplots(2, 2)

# Grafica en el primer subgráfico
axs[0, 0].scatter(datos_pareto[:, 4], datos_pareto[:, 0], s=30)
axs[0, 0].set_xlabel("Distancia")
axs[0, 0].set_ylabel("Peso mapa1")

# Grafica en el segundo subgráfico
axs[0, 1].scatter(datos_pareto[:, 4], datos_pareto[:, 1], s=30)
axs[0, 1].set_xlabel("Distancia")
axs[0, 1].set_ylabel("Peso mapa2")

# Grafica en el tercer subgráfico
axs[1, 0].scatter(datos_pareto[:, 4], datos_pareto[:, 2], s=30)
axs[1, 0].set_xlabel("Distancia")
axs[1, 0].set_ylabel("Peso mapa3")

# Grafica en el cuarto subgráfico
axs[1, 1].scatter(datos_pareto[:, 4], datos_pareto[:, 3], s=30)
axs[1, 1].set_xlabel("Distancia")
axs[1, 1].set_ylabel("Peso mapa4")

# Ajusta el diseño de la figura
plt.tight_layout()

# Muestra la figura con los cuatro subgráficos
plt.show()