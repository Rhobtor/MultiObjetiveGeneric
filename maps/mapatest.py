import numpy as np

# Load your custom map from a text file ('map.txt')
your_custom_map = np.genfromtxt('map.txt', delimiter=' ')

# Tamaño del mapa
map_size = your_custom_map.shape

# Crear un mapa de puntos de interés basado en tu mapa personalizado
interest_map = np.zeros(map_size)

# Generar puntos de interés en las ubicaciones donde tu mapa tiene el valor 1
for y in range(map_size[0]):
    for x in range(map_size[1]):
        if your_custom_map[y, x] == 1:
            # Calcular el valor de interés (en este caso, valor absoluto)
            #interest_value = abs(sum([(1 / ((x - i) ** 2 + (y - j) ** 2 + 1)) for i, j in [(20, 20), (40, 40), (60, 60), (80, 80)]])) #mapa1
            #interest_value = abs(sum([(1 / ((x - i) ** 2 + (y - j) ** 2 + 1)) for i, j in [(40, 20), (20, 60), (80, 60), (70, 30)]])) #mapa2
            #interest_value = abs(sum([(1 / ((x - i) ** 2 + (y - j) ** 2 + 1)) for i, j in [(70, 20), (60, 40), (20, 10), (30, 50)]])) #mapa3
            #interest_value = abs(sum([(1 / ((x - i) ** 2 + (y - j) ** 2 + 1)) for i, j in [(30, 20), (20, 40), (60, 10), (70, 80)]])) #mapa4
            #interest_value = abs((1 - x / map_size[1]) ** 2 + 100 * ((y / map_size[0]) - (x / map_size[1]) ** 2) ** 2) #mapa2
            # interest_value = abs((x**2 + y - 11)**2 + (x + y**2 - 7)**2) #mapa3
            interest_map[y, x] = interest_value
            #mapa4
             # Calcular el valor de interés utilizando la función de Rastrigin
            # A = 10
            # interest_map[y,x]= A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))
            
# Guardar el mapa de puntos de interés en un archivo de texto
np.savetxt('interest_map2.txt', interest_map, delimiter=' ')
