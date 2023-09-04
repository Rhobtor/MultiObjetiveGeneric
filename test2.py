import networkx as nx

# Create a graph
G = nx.Graph()

# Add nodes to the graph
nodes = [1, 2, 3, 4]
G.add_nodes_from(nodes)

# Add a single value attribute to each node
single_value = 1
attributes = {node: single_value for node in nodes}
nx.set_node_attributes(G, attributes, 'value')

# Modify the value of a specific node
new_value = 0.1
node_to_modify = 2
G.nodes[node_to_modify]['value'] = new_value

# Access the modified attribute of the specific node


import numpy as np

for i in range(7):
	current_value = G.nodes[node_to_modify]['value'] # Obtén el valor actual del atributo 'value'
	new_value = np.min([current_value + 0.05, 1])
	print(new_value) # Calcula el nuevo valor
	# if new_value > 1:
	#     new_value=1
	G.nodes[node_to_modify]['value'] = new_value  # Act
	
print("Node", node_to_modify, "new value:", G.nodes[node_to_modify]['value'])   