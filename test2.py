import networkx as nx

# Create a graph
G = nx.Graph()

# Add nodes to the graph
nodes = [1, 2, 3, 4]
G.add_nodes_from(nodes)

# Add a single value attribute to each node
single_value = 42
attributes = {node: single_value for node in nodes}
nx.set_node_attributes(G, attributes, 'value')

# Modify the value of a specific node
new_value = 99
node_to_modify = 2
G.nodes[node_to_modify]['value'] = new_value

# Access the modified attribute of the specific node
print("Node", node_to_modify, "new value:", G.nodes[node_to_modify]['value'])
