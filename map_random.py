import numpy as np

# Load the reference map from a text file
reference_map = np.genfromtxt('map.txt', delimiter=' ')

# Create a mask for locations where the reference map has a value of 1
mask = (reference_map == 1)

# Generate random values between 0 and 1 with the same shape as the reference map
random_values = np.random.rand(*reference_map.shape)

# Create the new map by copying the reference map and replacing values where the mask is True
new_map = np.copy(reference_map)

# Replace values where the mask is True with random values between 0 and 1
new_map[mask] = random_values[mask]

# Save the new map to a text file
np.savetxt('map_interested4.txt', new_map, delimiter=' ')

