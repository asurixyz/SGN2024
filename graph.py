import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

# Function to calculate the Hamiltonian of the system
def calculate_hamiltonian(spins, J, h):
    N = len(spins)
    interaction_term = 0
    for i in range(N):
        for j in range(i + 1, N):
            interaction_term += J[i, j] * spins[i] * spins[j]
    external_field_term = -np.sum(h * spins)
    H = -interaction_term + external_field_term
    return H

# Function to visualize the system
def visualize_network(spins, J, h):
    G = nx.Graph()
    N = len(spins)
    
    # Add nodes with their spins and external fields
    for i in range(N):
        G.add_node(i, spin=spins[i], h_value=h[i])
    
    # Randomly connect nodes
    num_edges = np.random.randint(N, N * (N - 1) // 4)  # Sparse random edges
    edges_added = 0
    while edges_added < num_edges:
        i, j = np.random.choice(N, 2, replace=False)
        if not G.has_edge(i, j):  # Ensure no duplicate edges
            G.add_edge(i, j, weight=J[i, j])
            edges_added += 1
    
    # Position nodes
    pos = nx.spring_layout(G, k=0.2, seed=42)  # Adjust k for spacing
    
    # Set node colors based on spins: Red (-1) or Blue (+1)
    node_colors = ['red' if spins[i] == -1 else 'blue' for i in range(N)]
    
    # Visualization
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=200, node_color=node_colors, font_size=12, font_weight='bold', width=2, edge_color='black')
    
    # Annotate external field values near nodes
    for i in range(N):
        plt.text(pos[i][0], pos[i][1] + 0.1, f"h={h[i]:.2f}", fontsize=12, ha='center')
    
    # Add legend in the bottom-right corner
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Spin +1 (Blue)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Spin -1 (Red)')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=12, bbox_to_anchor=(1, 0), title="Legend", title_fontsize='13', prop={'weight': 'bold'})
    
    # Add node count in the bottom-left corner
    plt.text(-0.2, -0.2, f'$\mathbf{{N = {N}}}$', fontsize=14, ha='center', va='center', transform=plt.gca().transAxes, fontweight='bold')
    
    # Title
    plt.title("Network Visualization: Spins and External Fields", fontsize=16, fontweight='bold')
    plt.show()

# Initialize the system with 30 nodes
N = 8  # Number of individuals (spins)

# Randomly assign spins (-1 or +1)
spins = np.random.choice([-1, 1], size=N)

# Randomly assign external field values (h values) for each individual between -2 and 2
h = np.random.uniform(-2, 2, size=N)

# Interaction matrix (J_ij = random values between -1 and 1 for interactions)
J = np.random.uniform(-1, 1, size=(N, N))
np.fill_diagonal(J, 0)  # No self-interaction

# Calculate and print the Hamiltonian
H = calculate_hamiltonian(spins, J, h)
print(f"Total Hamiltonian of the system: {H:.2f}")

# Visualize the system as a network
visualize_network(spins, J, h)

def calculate_energy_difference(spins, J, h, node_index):
    """Calculate energy difference if we flip the spin at node_index"""
    N = len(spins)
    delta_E = 0
    
    # Change in interaction energy
    for j in range(N):
        if j != node_index:
            delta_E += 2 * J[node_index, j] * spins[j] * spins[node_index]
    
    # Change in external field energy
    delta_E += 2 * h[node_index] * spins[node_index]
    
    return delta_E

def glauber_flip_probability(delta_E, temperature):
    """Calculate flip probability according to Glauber dynamics"""
    return 1 / (1 + np.exp(delta_E / temperature))

def simulate_and_animate(spins, J, h, temperature, num_steps):
    """Simulate the system evolution and create animation"""
    N = len(spins)
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    
    # Network plot subplot
    ax_network = plt.subplot(gs[0])
    # Hamiltonian plot subplot
    ax_energy = plt.subplot(gs[1])
    
    # Initialize network
    G = nx.Graph()
    for i in range(N):
        G.add_node(i)
    
    # Add edges (keep them fixed throughout simulation)
    num_edges = np.random.randint(N, N * (N - 1) // 4)
    edges_added = 0
    while edges_added < num_edges:
        i, j = np.random.choice(N, 2, replace=False)
        if not G.has_edge(i, j):
            G.add_edge(i, j, weight=J[i, j])
            edges_added += 1
    
    pos = nx.spring_layout(G, k=0.2, seed=42)
    
    # Initialize Hamiltonian history
    hamiltonians = [calculate_hamiltonian(spins, J, h)]
    time_points = [0]
    
    # Initialize plot elements
    nodes = nx.draw_networkx_nodes(G, pos, node_color=['red' if s == -1 else 'blue' for s in spins],
                                 ax=ax_network, node_size=200)
    nx.draw_networkx_edges(G, pos, ax=ax_network)
    nx.draw_networkx_labels(G, pos, ax=ax_network)
    
    energy_line, = ax_energy.plot(time_points, hamiltonians, 'r-')
    
    # Setup axis labels and titles
    ax_network.set_title("Spin Network Evolution", fontsize=14, fontweight='bold')
    ax_energy.set_xlabel("Time Step", fontsize=12)
    ax_energy.set_ylabel("Hamiltonian", fontsize=12)
    ax_energy.grid(True)
    
    # Add legend to network plot
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Spin +1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Spin -1')
    ]
    ax_network.legend(handles=legend_elements, loc='upper right')
    
    spins_history = [spins.copy()]
    
    def update(frame):
        current_spins = spins_history[-1].copy()
        
        # Randomly select a spin to potentially flip
        node_index = np.random.randint(0, N)
        delta_E = calculate_energy_difference(current_spins, J, h, node_index)
        
        # Determine if spin should flip
        if np.random.random() < glauber_flip_probability(delta_E, temperature):
            current_spins[node_index] *= -1
        
        # Update visualization
        nodes.set_color(['red' if s == -1 else 'blue' for s in current_spins])
        
        # Update Hamiltonian plot
        H = calculate_hamiltonian(current_spins, J, h)
        hamiltonians.append(H)
        time_points.append(frame)
        
        energy_line.set_data(time_points, hamiltonians)
        ax_energy.relim()
        ax_energy.autoscale_view()
        
        spins_history.append(current_spins)
        return nodes, energy_line
    
    anim = FuncAnimation(fig, update, frames=num_steps, interval=100, blit=True)
    plt.tight_layout()
    return anim

if __name__ == "__main__":
    # Simulation parameters
    N = 20  # Number of nodes
    temperature = 1.0  # Temperature parameter
    num_steps = 100  # Number of simulation steps

    # Initialize system
    spins = np.random.choice([-1, 1], size=N)
    h = np.random.uniform(-2, 2, size=N)
    J = np.random.uniform(-1, 1, size=(N, N))
    np.fill_diagonal(J, 0)

    # Create and display animation
    anim = simulate_and_animate(spins, J, h, temperature, num_steps)
    plt.show()
