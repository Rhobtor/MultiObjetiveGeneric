import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
from AlgaeBloomGroundTruth import algae_bloom
from ShekelGroundTruth import shekel
from itertools import cycle
from GaussianProcessModel import GaussianProcessModel
from sklearn.metrics import mean_squared_error as mse


class PatrollingGraphRoutingProblem:

	def __init__(self, navigation_map: np.ndarray, 
			  	importance_map:np.ndarray,
				scale: int, 
				n_agents: int, 
				max_distance: float, 
				initial_positions: np.ndarray,
				ground_truth: str,
				final_positions: np.ndarray = None):

		self.navigation_map = navigation_map
		self.information_map= importance_map
		self.scale = scale
		self.n_agents = n_agents
		self.max_distance = max_distance
		self.initial_positions = initial_positions
		self.final_positions = final_positions
		self.waypoints = {agent_id: [] for agent_id in range(n_agents)}
		
		# self.visited[self.agent_pos[0]][self.agent_pos[1]] = 255
		# self.S[self.agent_pos[0]][self.agent_pos[1]] = 0
		# self.R_abs[self.agent_pos[0]][self.agent_pos[1]] -= 50
		self.S = np.ones(self.navigation_map.shape)
		max_importance_map = tuple([np.sum(item) for item in importance_map])
		print(self.S.shape)
		self.rho_next = {}
		self.rho_act = {}
		self.rI_next = {}
		self.rI_act = {}
		self.roo_next = np.empty(n_agents, dtype=float)
		self.roo_act = np.empty(n_agents, dtype=float)
		self.roI_next = np.empty(n_agents, dtype=float)
		self.roI_act = np.empty(n_agents, dtype=float)
		benchmark = ground_truth

		# Create the graph
		self.G = create_graph_from_map(self.navigation_map, self.scale,self.information_map)
		self.L = create_graph_from_map2(self.navigation_map, self.scale)
		# Create the grund truth #
		""" Create the benchmark """
		if benchmark == 'shekel':
			self.ground_truth = shekel(self.navigation_map, max_number_of_peaks=6, seed = 0, dt=0.05)
		elif benchmark == 'algae_bloom':
			self.ground_truth = algae_bloom(self.navigation_map, dt=0.5, seed=0)
		else:
			raise ValueError('Unknown benchmark')

		self.model = GaussianProcessModel(navigation_map = navigation_map)
		self.model.reset()
		self.information_map = np.zeros_like(self.navigation_map)

		# Rendering variables #
		self.fig = None
		self.colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'olive', 'cyan', 'magenta']
		self.markers = ['o', 'v', '*', 'p', '>', 's', 'p', '*', 'h', 'H', 'D', 'd']

	def reset(self):
		# Reset all the variables of the scenario #

		self.ground_truth.reset()
		self.model.reset()

		self.agent_positions = self.initial_positions.copy()
		self.agent_pos_ant= self.agent_positions
		
		# Reset the rewards #
		self.rewards = {}


		self.waypoints = {agent_id: [list(self.G.nodes[initial_position]['position'])] for agent_id, initial_position in zip(range(self.n_agents), self.initial_positions)}
		self.agent_distances = {agent_id: 0 for agent_id in range(self.n_agents)}

		# Input the initial positions to the model
		new_position_coordinates = np.array([self.G.nodes[new_position]['position'] for new_position in self.agent_positions])
		new_samples = self.ground_truth.read(new_position_coordinates)

		# # Input the previous positions to the model
		# new_preposition_coordinates = np.array([self.G.nodes[new_preposition]['position'] for new_preposition in self.agent_positions])
		# new_presamples = self.ground_truth.read(new_preposition_coordinates)




		# Update the model
		self.model.update(new_position_coordinates, new_samples)

	def update_maps(self):
		""" Update the idleness and information maps """

		# Input the initial positions to the model

		

		new_position_coordinates = np.array([self.G.nodes[new_position]['position'] for new_position in self.agent_positions if new_position != -1])
		
		# Check if no new positions are available
		if new_position_coordinates.shape[0] != 0:
			new_samples = self.ground_truth.read(new_position_coordinates)

			# Update the model
			self.model.update(new_position_coordinates, new_samples)
			self.information_map = self.model.predict() # esta es la y , la w en el en paper

			# Initialize the reward variables
			# self.rewards = {'Error': mse(self.model.predict(), self.ground_truth.read()),
			# 				'Uncertainty': 0,
			#                 'Visited':0}

	def compute_coverage_mask(self, position: np.ndarray) -> np.ndarray:
		""" Obtain a circular mask centered in position with a radius of coverage_radius """

		mask = np.zeros_like(self.navigation_map)
		
		# Compute the positions 
		x, y = np.meshgrid(np.arange(self.navigation_map.shape[1]), np.arange(self.navigation_map.shape[0]))
		x = x - position[0]
		y = y - position[1]
		r = np.sqrt(x**2 + y**2)

		# Create the mask
		mask[r <= self.coverage_radius] = 1

		return mask.astype(bool)

	def step(self, new_positions: np.ndarray):

		# Check if the new positions are neighbors of the current positions of the agents
		for i in range(self.n_agents):

			if new_positions[i] == -1:
				continue

			if new_positions[i] not in list(self.G.neighbors(self.agent_positions[i])):
				raise ValueError('The new positions are not neighbors of the current positions of the agents')

			# Compute the distance traveled by the agents using the edge weight
			self.agent_distances[i] += self.G[self.agent_positions[i]][new_positions[i]]['weight']
			self.G.nodes[self.agent_positions[i]]['values']=0,1

		# Update the positions of the agents
		
		self.agent_positions = new_positions.copy()
		# self.L[self.agent_positions[0]]['visited']=2
		# print(self.L.nodes[1]['value'])
		
		# self.visited[self.agent_pos_ant[0]][self.agent_pos_ant[1]] = 127 # Casilla anterior sombreada !
		# self.visited[self.agent_positions[0]][self.agent_positions[1]] = 255 # Casilla visitada bien marcada!


		# # Se purga el interés de la casilla de la que venimos # #
		# self.R_abs[self.agent_pos_ant[0]][self.agent_pos_ant[1]] -= 50 
		
		# self.rewards = {}  # Crear un diccionario para almacenar los valores 'Visited' por agente
		visited_values = []
		#self.rewards=()
		#rh_reward = np.array([0]*len(self.G.nodes[1]['rh_reward']), dtype = float)
###################################################################################################################################################################
		for i in range(self.n_agents):
			# Procesamos el reward #
			self.rho_next[i] = self.G.nodes.get(self.agent_positions[i], {'value': 0.0})['value'] 
			self.rho_act[i] = self.G.nodes.get(self.agent_pos_ant[i], {'value': 0.0})['value']
			# self.rI_next[i] = self.G.nodes.get(self.agent_positions[i], {'importance': 0.0})['importance'] 
			# self.rI_act[i] = self.G.nodes.get(self.agent_pos_ant[i], {'importance': 0.0})['importance']
			# b = self.rho_act[0]
			# a = [element * b for element in ([self.rI_act[i]] if isinstance(self.rI_act[i], (float, int)) else self.rI_act[i])]
			# c = self.rho_next[0]
			# d = [element * c for element in ([self.rI_next[i]] if isinstance(self.rI_next[i], (float, int)) else self.rI_next[i])]
			# # a=[element * b for element in self.rI_act[i]]
			# Check if the node exists in the graph before accessing its attributes	
			node_id = self.agent_positions[i]
			if node_id in self.G.nodes:
				self.G.nodes[node_id]['rh_reward'] = self.rho_next[i] - self.rho_act[i]
			#else:
    # Handle the case where the node doesn't exist, e.g., print an error message
				#print(f"Node {node_id} does not exist in the graph.")

			#self.G.nodes[self.agent_positions[i]]['rh_reward']= self.rho_next[i] - self.rho_act[i]			
		reward = np.array([0]*len(self.G.nodes[1]['importance']), dtype = float)
		idle = [self.G.nodes[node]['rh_reward'] for node in self.agent_positions if node in self.G.nodes]
		imp = [self.G.nodes[node]['importance'] for node in self.agent_positions if node in self.G.nodes]
		#reward = [0.0] * len(imp[0])  # Initialize reward with zeros
		for imp_index in range(len(self.G.nodes[1]['importance'])):
			for ship_index in range(len(idle)):
				if imp_index < len(imp[ship_index]) and ship_index < len(idle):
					reward[imp_index] += np.array(idle[ship_index]) * np.array(imp[ship_index][imp_index])

		# print('bbb',reward)
		for node in range(1, len(self.G)):
			if node in self.agent_positions:
				self.G.nodes[node]['importance'] = list(np.array(self.G.nodes[node]['importance']) - 0.2*np.array(self.G.nodes[node]['importance']))
				for index in range(len(self.G.nodes[node]['importance'])):
					if self.G.nodes[node]['importance'][index] < 0:
						self.G.nodes[node]['importance'][index] = 0
		
		
		self.rewards = reward
##################################################################################################################################################################
	
		# 	# self.roI_next[i] = np.array(self.rI_next[i])
		# 	# self.roI_act[i] = np.array(self.rI_act[i])
		# 	# print(self.roI_next)
		# 	# self.roo_next = np.array(self.roo_next)
		# 	# self.roo_act = np.array(self.roo_act)
		# 	# self.roI_next = np.array(self.roI_next)
		# 	# self.roI_act = np.array(self.roI_act)
			
		# 	visited_value= [x - y for x, y in zip(d, a)]
		# 	# #visited_value = [rho_next * rI_next - rho_act * rI_act for rI_next,rI_act in zip(rI_next,rI_act)]
		# 	# visited_value = (self.roo_next * self.roI_next) - (self.roo_act * self.roI_act)
			
			
		# 	self.rewards_t[i]= visited_value
		# 	self.rewards[i]=tuple(self.rewards_t[i])
			# print(reward)
			# print(self.rewards[i])
			#visited_values.append(tuple([visited_value]))  # Convert the value to a list before creating a tuple
			#self.rewards = tuple([v3])

#### Parte del codigo antigua, se piensa recompensa de manera individual para cada robot , despues del pensamiento mapa pensar si es posible realizar un cambio en optimize_deap_mo.py para ver si es posible realizar este pensamiento			
########################################################################################################################
# 		for i in range(self.n_agents):
# 			# Procesamos el reward #
# 			self.rho_next[i] = self.G.nodes.get(self.agent_positions[i], {'value': 0.0})['value']
# 			self.rho_act[i] = self.G.nodes.get(self.agent_pos_ant[i], {'value': 0.0})['value']

		
# # Calcular el valor 'Visited' para cada agente y almacenarlo en self.rewards
# 		self.rewards = {}  # Crear un diccionario para almacenar los valores 'Visited' por agente
# 		for i in range(self.n_agents):
# 			visited_value = self.rho_next[i] - self.rho_act[i]
			
# 			self.rewards[i] = (-110)*visited_value + 111
# 		# self.rewards[i]=(self.rewards_t[i])
# 		# print(self.rewards[i])
#####################################################################################################################

		# Calcular el valor 'Visited' para cada agente y almacenarlo en self.rewards
		# self.rewards = {}  # Crear un diccionario para almacenar los valores 'Visited' por agente
		# visited_values = []
		# for i in range(self.n_agents):
		# 	visited_value = (np.array(self.rho_next[i])*np.array(self.rI_next[i]) - np.array(self.rho_act[i])*np.array(self.rI_act[i]))
			
		# 	# self.rewards[i] = (-110)*visited_value + 111  #funcion mia de recompensa de prueba
		# 	# self.rewards[i]= tuple(visited_value)
		# 	visited_values.append(tuple(visited_value))
		# 	self.rewards[i]= tuple(visited_value)
        # Ojito que aquí decidimos cuánto penalizamos visitar una ilegal una anterior o una nueva #
		#reward = (1-ilegal)*((5.505/255)*(reward-255)+5) - ilegal*(10)
		
		for node_index in self.G.nodes:
			current_value = self.G.nodes[node_index]['value']  # Obtén el valor actual del atributo 'value'
			new_value = min([current_value + 0.05, 1]) # Calcula el nuevo valor
			
			self.G.nodes[node_index]['value'] = new_value  # Act

		for i in range(self.n_agents):
			if self.agent_positions[i] in self.L.nodes:
				self.G.nodes[self.agent_positions[i]]['value'] = 0.1
		
		
		
		
		
		self.agent_pos_ant= self.agent_positions #casi al final


		# Update the waypoints
		for agent_id, new_position in enumerate(new_positions):
			if new_position != -1:
				# Append the position from the node
				self.waypoints[agent_id].append(list(self.G.nodes[new_position]['position']))

		# Update the idleness and information maps with the rewards
		self.update_maps()

		done = np.asarray([agent_distance > self.max_distance for agent_distance in self.agent_distances.values()]).all()

		done = done or np.asarray([agent_position == -1 for agent_position in self.agent_positions]).all()
		
		# Return the rewards
		return self.rewards, done

	def evaluate_path(self, multiagent_path: dict, render = False) -> dict:
	
		# """ Evaluate a path """
		
		# self.reset()

		# if render:
		# 	self.render()

		# done = False
		# t = 0
		# step_rewards = [[] for _ in range(self.n_agents)]

		# #new_rewards = np.array([0]*len(self.G.nodes[1]['importance']), dtype = float)
		# new_reward=[]
		# final_reward={}
		# while not done:

		# 	next_positions = np.zeros_like(self.agent_positions)
			
		# 	for i in range(self.n_agents):
		# 		if t < len(multiagent_path[i]):
					
		# 			next_positions[i] = multiagent_path[i][t]

		# 		else:
		# 			next_positions[i] = -1
			
		# 	new_rewards, done = self.step(next_positions)
		# 	# new_reward.append(tuple(new_rewards))
		# 	final_reward=new_reward.append(tuple(new_rewards))
		# 	#print('a',new_rewards)

		# 	if render:
		# 		self.render()
			
		# 	t += 1

		# return final_reward
	
		""" Evaluate a path """
		
		self.reset()

		if render:
			self.render()
		
		done = False
		t = 0

		final_rewards = np.array([0]*len(self.G.nodes[1]['importance']), dtype = float)
		while not done:
			next_positions = np.zeros_like(self.agent_positions)
			
			for i in range(self.n_agents):
				if t < len(multiagent_path[i]):
					
					next_positions[i] = multiagent_path[i][t]
				else:
					next_positions[i] = -1

			new_rewards, done = self.step(next_positions)

			#print('esto',new_rewards)
			final_rewards+=new_rewards
			# print('esttta',final_rewards)
			# for key in new_rewards.keys():
			# 	final_rewards[key] += new_rewards[key]
			# print(new_rewards)
			if render:
				self.render()
			
			t += 1
		return final_rewards

	def render(self):

		if self.fig is None:

			self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 10))

			self.d1 = self.ax[0].imshow(self.information_map, cmap='gray', vmin=0, vmax=1)
			self.d2 = self.ax[1].imshow(self.ground_truth.read(), cmap='gray', vmin=0, vmax=1)

			self.agents_render_pos = []
			for i in range(self.n_agents):
				
				# Obtain the agent position from node to position
				agent_position_coords = self.G.nodes[self.agent_positions[i]]['position']
				self.agents_render_pos.append(self.ax[0].plot(agent_position_coords[0], agent_position_coords[1], color=self.colors[i], marker=self.markers[i], markersize=10, alpha=0.35)[0])

		else:

			for i in range(self.n_agents):
				
				traj = np.asarray(self.waypoints[i])
				# Plot the trajectory of the agent
				self.agents_render_pos[i].set_data(traj[:,0], traj[:,1])

			self.d1.set_data(self.information_map)
			self.d2.set_data(self.ground_truth.read())

			
		self.fig.canvas.draw()
		plt.pause(0.01)

def create_graph_from_map(navigation_map: np.ndarray, resolution: int,importance_map: np.ndarray):
	""" Create a graph from a navigation map """

	# Obtain the scaled navigation map
	scaled_navigation_map = navigation_map[::resolution, ::resolution]

	# Obtain the positions of the nodes
	visitable_positions = np.column_stack(np.where(scaled_navigation_map == 1))

	# Create the graph
	G = nx.Graph()

	# Add the nodes
	for i, position in enumerate(visitable_positions):
		G.add_node(i, position=position[::-1]*resolution, coords=position*resolution)
		nx.set_node_attributes(G, {i: 1}, 'value')
		x_index, y_index = position * resolution
		importance_values = [item[x_index, y_index] for item in importance_map]
		nx.set_node_attributes(G, {i:importance_values},'importance')
		nx.set_node_attributes(G, {i: 0}, 'rh_reward')
	
	# for i, position in enumerate(visitable_positions):
	# 	G.add_node(i, position=position[::-1]*resolution, coords=position*resolution)
	# 	nx.set_node_attributes(G, {i: 1}, 'value')
	# 	x_index, y_index = position // resolution
	# 	importance_values = [item[x_index, y_index] for item in importance_map]
	# 	nx.set_node_attributes(G, importance_values,'importance')
	# Add the edges
	for i, position in enumerate(visitable_positions):
		for j, other_position in enumerate(visitable_positions):
			if i != j:
				if np.linalg.norm(position - other_position) <= np.sqrt(2):
					G.add_edge(i, j, weight=np.linalg.norm(position - other_position)*resolution)

	return G




def create_graph_from_map2(navigation_map: np.ndarray, resolution: int):
	""" Create a graph from a navigation map """

	# Obtain the scaled navigation map
	scaled_navigation_map = navigation_map[::resolution, ::resolution]

	# Obtain the positions of the nodes
	visitable_positions = np.column_stack(np.where(scaled_navigation_map == 1))

	# Create the graph
	L = nx.Graph()

	# Add the nodes
	for i, position in enumerate(visitable_positions):
		L.add_node(i, position=position[::-1]*resolution, coords=position*resolution)
		nx.set_node_attributes(L, {i: 1}, 'value')
	# Add the edges
	for i, position in enumerate(visitable_positions):
		for j, other_position in enumerate(visitable_positions):
			if i != j:
				if np.linalg.norm(position - other_position) <= np.sqrt(2):
					L.add_edge(i, j, weight=np.linalg.norm(position - other_position)*resolution)


	

	return L

def plot_graph(G: nx.Graph, path: list = None, ax=None, cmap_str='Reds', draw_nodes=True):

	if ax is None:
		plt.figure()
		ax = plt.gca()

	positions = nx.get_node_attributes(G, 'position')
	positions = {key: np.asarray([value[0], -value[1]]) for key, value in positions.items()}

	if draw_nodes:
		nx.draw(G, pos=positions, with_labels = True, node_color='gray', arrows=True, ax=ax)

	if path is not None:
		cmap = matplotlib.colormaps[cmap_str]
		red_shades = cmap(np.linspace(0, 1, len(path)))
		nx.draw_networkx_nodes(G, pos=positions, nodelist=path, node_color=red_shades, ax=ax)

	return ax

def path_length(G: nx.Graph, path: list) -> float:

	length = 0

	for i in range(len(path)-1):
		length += G[path[i]][path[i+1]]['weight']

	return length


def random_shorted_path(G: nx.Graph, p0: int, p1:int) -> list:

	random_G = G.copy()
	for edge in random_G.edges():
		random_G[edge[0]][edge[1]]['weight'] = np.random.rand()

	return nx.shortest_path(random_G, p0, p1, weight='weight')[1:]


def create_random_path_from_nodes(G : nx.Graph, start_node: int, distance: float, final_node: int = None) -> list:
		""" Select random nodes and create random path to reach them """

		path = []
		remain_distance = distance

		# Append the start node
		path.append(start_node)

		while path_length(G, path) < distance:

			# Select a random node
			next_node = np.random.choice(G.nodes())

			# Obtain a random path to reach it
			new_path = random_shorted_path(G, path[-1], next_node)
			path.extend(new_path)

			# Compute the distance of path
			remain_distance -= path_length(G, new_path)

		# Append the shortest path to the start node
		G_random = G.copy()
		# Generate random weights
		for edge in G_random.edges():
			G_random[edge[0]][edge[1]]['weight'] = np.random.rand()

		# Append the shortest path to the start node
		if final_node is not None:
			path.extend(nx.shortest_path(G_random, path[-1], final_node, weight='weight')[1:])
		else:
			path.extend(nx.shortest_path(G_random, path[-1], start_node, weight='weight')[1:])

		return path[1:]

def create_multiagent_random_paths_from_nodes(G, initial_positions, distance, final_positions=None):

		if final_positions is not None:
			multiagent_path = {agent_id: create_random_path_from_nodes(G, initial_positions[agent_id], distance, final_positions[agent_id]) for agent_id in range(len(initial_positions))}
		else:
			multiagent_path = {agent_id: create_random_path_from_nodes(G, initial_positions[agent_id], distance) for agent_id in range(len(initial_positions))}
		#print(multiagent_path)
		return multiagent_path

def cross_operation_between_paths(G: nx.Graph, path1, path2):
	""" Perform a cross operation between two paths. """

	# Transform the paths into numpy arrays
	path1 = np.asarray(path1)
	path2 = np.asarray(path2)

	# Obtain the split points
	i = np.random.randint(0, len(path1), size=2)
	i.sort()

	j = np.random.randint(0, len(path2), size=2)
	j.sort()

	
	resulting_path_1 = np.concatenate((path1[:i[0]], 
										nx.shortest_path(G, path1[i[0]], path2[j[0]])[:-1],
										path2[j[0]:j[1]],
										nx.shortest_path(G, path2[j[1]], path1[i[1]])[:-1],
										path1[i[1]:]
										))

	resulting_path_2 = np.concatenate((path2[:j[0]], 
										nx.shortest_path(G, path2[j[0]], path1[i[0]])[:-1],
										path1[i[0]:i[1]],
										nx.shortest_path(G, path1[i[1]], path2[j[1]])[:-1],
										path2[j[1]:]
										))		

	return resulting_path_1.tolist(), resulting_path_2.tolist()

def mutation_operation(G: nx.Graph, path, mut_prob=0.1):
	""" Alter a random node to its closest neighbor. """

	new_path = path.copy()

	# Select a random node
	for i in range(1, len(new_path)-1):

		if np.random.rand() < mut_prob:

			# Obtain the common neighbors between the node i and the next node
			common_neighbors_1 = list(nx.neighbors(G, new_path[i-1]))
			common_neighbors_2 = list(nx.neighbors(G, new_path[i+1]))
			common_neighbors = [node for node in common_neighbors_1 if node in common_neighbors_2 and node != new_path[i]]

			# Select a random neighbor
			if len(common_neighbors) > 0:
				new_path[i] = np.random.choice(common_neighbors)

	return new_path


if __name__ == '__main__':

	np.random.seed(0)

	navigation_map = np.genfromtxt('map.txt', delimiter=' ')
	importance_map= importance_map = [ np.genfromtxt('map_interested1.txt', delimiter=' '), 
                   np.genfromtxt('map_interested2.txt', delimiter=' ')]
	N_agents = 4
	initial_positions = np.array([10,20,30,40])[:N_agents]
	final_positions = np.array([10,10,30,40])[:N_agents]
	scale = 3

	environment = PatrollingGraphRoutingProblem(navigation_map = navigation_map,
											 	importance_map=importance_map,
												n_agents=N_agents, 
												initial_positions=initial_positions,
												# final_positions=final_positions,
												scale=scale,
												max_distance=350.0,
												ground_truth='shekel',

	)


	# path = create_multiagent_random_paths_from_nodes(environment.G, initial_positions, 150, final_positions)
	path = create_multiagent_random_paths_from_nodes(environment.G, initial_positions, 150)
	path_1, path_2 = cross_operation_between_paths(environment.G, path[0], path[1])
	path_crossed = {0: path_1, 1: path_2,2: path_1}

	#environment.evaluate_path(path, render=True)
	
	a=environment.evaluate_path(path, render=True)
	reward=[]
	for valor in a:
		reward=valor
		print('f',reward)
	print('d',a)
	#new_reward.append(tuple(new_rewards))
	# environment.evaluate_path(path_crossed, render=True)

	plt.pause(1000)

	# Plot the graph to visualize the crossing
	fig, axs = plt.subplots(2, 2, figsize=(10, 5))
	plot_graph(environment.G, path=path[0], draw_nodes=True, ax=axs[0,0])
	plot_graph(environment.G, path=path[1], draw_nodes=True, ax=axs[0,1], cmap_str='Greens')
	plot_graph(environment.G, path=path[2], draw_nodes=True, ax=axs[1,0], cmap_str='Blues')
	# plot_graph(environment.G, path=path_crossed[0], draw_nodes=True, ax=axs[1,0])
	# plot_graph(environment.G, path=path_crossed[1], draw_nodes=True, ax=axs[1,1], cmap_str='Greens')
	plt.show()




















