import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand
import networkx as nx
from matplotlib.animation import FuncAnimation

def initialise(N):
    return np.random.choice([-1,1], size = N)

def contr(N):
    cont = np.random.uniform(-1, 1, size=N)  # Directly create the array of size N
    return cont

def adj_matrix(N, sparsity):
    if not 0 <= sparsity <= 1:
        raise ValueError("Sparsity must be between 0 and 1")
    adj = np.random.randint(-5, 6, size=(N, N))
    np.fill_diagonal(adj, 0)
    adj = np.triu(adj)
    adj = adj + adj.T
    
    mask = np.random.random((N, N)) < sparsity
    mask = np.triu(mask) + np.triu(mask).T  # Make mask symmetric
    np.fill_diagonal(mask, True)  # Always mask diagonal
    adj[mask] = 0
    
    return adj

def init_field(N):
    return np.random.uniform(-5, 5, size=N)

def move(config, adj, beta, N):
    k = np.random.randint(0, N)
    s = config[k]
    delta = np.sum(2 * s * config * adj[k])
    
    if delta < 0 or rand.rand() < np.exp(-delta * beta):
        config[k] *= -1
    
    return config

def energy(config, adj, field, N):
    energy = -np.sum(config[:, None] * config[None, :] * adj)  # Vectorized calculation
    energy += np.sum(config * field)
    return energy

def mag(config, N):
    return np.mean(config)

def calc_field(field, config, cont, gamma, N):
    return (1 - gamma) * field - gamma * config * cont

# Initialize simulation parameters
N = 20
time_upper = 300  # Reduced time for animation
beta = 0.9
gamma = 0.2
sparsity = 0.6
config = initialise(N)
cont = contr(N)
adj = adj_matrix(N, sparsity)
field = init_field(N)

# Pre-allocate arrays for plots
energies = np.zeros(time_upper)
mags_arr = np.zeros(time_upper)
avg_field = np.zeros(time_upper)

# Initialize plot - 3 subplots (1 for graph, 3 for metrics)
fig = plt.figure(figsize=(16, 12))
ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=3)  # Graph takes full left side
ax2 = plt.subplot2grid((3, 2), (0, 1))  # Magnetization
ax3 = plt.subplot2grid((3, 2), (1, 1))  # Energy
ax4 = plt.subplot2grid((3, 2), (2, 1))  # Average field

# Networkx graph initialization
G = nx.Graph()
G.add_nodes_from(range(N))
for i in range(N):
    for j in range(i+1, N):
        if adj[i, j] != 0:
            G.add_edge(i, j, weight=adj[i, j])

# Plotting setup for graph
pos = nx.spring_layout(G)  # Node positions for visualization
node_colors = ['blue' if config[i] == 1 else 'red' for i in range(N)]

# Create separate lines for each plot
mag_line, = ax2.plot([], [], label='Magnetisation', color='blue')
energy_line, = ax3.plot([], [], label='Energy', color='red')
field_line, = ax4.plot([], [], label='Average Self-field', color='orange')

# Set up individual plot parameters
for ax, title in zip([ax2, ax3, ax4], ['Magnetisation vs Time', 'Energy vs Time', 'Average Field vs Time']):
    ax.set_xlabel('Time')
    ax.set_title(title)
    ax.legend()

# Function to update the graph and plots
def update(frame):
    global config, field, G
    
    # Update the system state
    for j in range(N):
        config = move(config, adj, beta, N)
        field = calc_field(field, config, cont, gamma, N)

    # Update metrics
    energies[frame] = energy(config, adj, field, N)
    mags_arr[frame] = mag(config, N)
    avg_field[frame] = np.mean(field)

    # Update node colors and graph
    node_colors = ['blue' if config[i] == 1 else 'red' for i in range(N)]
    ax1.clear()
    ax1.set_title('Graph with Node States')
    nx.draw(G, pos, ax=ax1, node_color=node_colors, with_labels=True, 
            edge_color='black', width=1, node_size=300)
    
    # Dynamic x-axis adjustment for all plots
    if frame > 0:
        for ax in [ax2, ax3, ax4]:
            ax.set_xlim(max(0, frame - 100), frame + 10)
        
        # Update y-axis limits for each plot
        ax2.set_ylim(min(mags_arr[:frame+1]) - 0.1, max(mags_arr[:frame+1]) + 0.1)
        ax3.set_ylim(min(energies[:frame+1]) - 0.1, max(energies[:frame+1]) + 0.1)
        ax4.set_ylim(min(avg_field[:frame+1]) - 0.1, max(avg_field[:frame+1]) + 0.1)
    
    # Update each line with its corresponding data
    mag_line.set_data(range(frame+1), mags_arr[:frame+1])
    energy_line.set_data(range(frame+1), energies[:frame+1])
    field_line.set_data(range(frame+1), avg_field[:frame+1])

    return mag_line, energy_line, field_line, ax1

# Create the animation
ani = FuncAnimation(fig, update, frames=time_upper, interval=50, blit=False)

plt.tight_layout()
plt.show()