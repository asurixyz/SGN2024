import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

def calculate_hamiltonian(spins, J, h):
    N = len(spins)
    interaction_term = 0
    for i in range(N):
        for j in range(i + 1, N):
            interaction_term += J[i, j] * spins[i] * spins[j]
    external_field_term = -np.sum(h * spins)
    H = -interaction_term + external_field_term
    return H

def calculate_energy_difference(spins, J, h, node_index):
    N = len(spins)
    delta_E = 0
    for j in range(N):
        if j != node_index:
            delta_E += 2 * J[node_index, j] * spins[j] * spins[node_index]
    delta_E += 2 * h[node_index] * spins[node_index]
    return delta_E

def glauber_flip_probability(delta_E, temperature):
    return 1 / (1 + np.exp(delta_E / temperature))

def simulate_and_animate(spins, J, h, temperature, num_steps):
    N = len(spins)
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    ax_network = plt.subplot(gs[0])
    ax_energy = plt.subplot(gs[1])
    G = nx.Graph()
    for i in range(N):
        G.add_node(i)
    num_edges = np.random.randint(N, N * (N - 1) // 4)
    edges_added = 0
    while edges_added < num_edges:
        i, j = np.random.choice(N, 2, replace=False)
        if not G.has_edge(i, j):
            G.add_edge(i, j, weight=J[i, j])
            edges_added += 1
    pos = nx.spring_layout(G, k=0.2, seed=42)
    hamiltonians = [calculate_hamiltonian(spins, J, h)]
    time_points = [0]
    nodes = nx.draw_networkx_nodes(G, pos, node_color=['red' if s == -1 else 'blue' for s in spins], ax=ax_network, node_size=200)
    nx.draw_networkx_edges(G, pos, ax=ax_network)
    nx.draw_networkx_labels(G, pos, ax=ax_network)
    energy_line, = ax_energy.plot(time_points, hamiltonians, 'r-')
    ax_network.set_title("Spin Network Evolution", fontsize=14, fontweight='bold')
    ax_energy.set_xlabel("Time Step", fontsize=12)
    ax_energy.set_ylabel("Hamiltonian", fontsize=12)
    ax_energy.grid(True)
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Spin +1'), Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Spin -1')]
    ax_network.legend(handles=legend_elements, loc='upper right')
    spins_history = [spins.copy()]
    def update(frame):
        current_spins = spins_history[-1].copy()
        node_index = np.random.randint(0, N)
        delta_E = calculate_energy_difference(current_spins, J, h, node_index)
        if np.random.random() < glauber_flip_probability(delta_E, temperature):
            current_spins[node_index] *= -1
        nodes.set_color(['red' if s == -1 else 'blue' for s in current_spins])
        H = calculate_hamiltonian(current_spins, J, h)
        hamiltonians.append(H)
        time_points.append(frame)
        energy_line.set_data(time_points, hamiltonians)
        ax_energy.relim()
        ax_energy.autoscale_view()
        spins_history.append(current_spins)
        return nodes, energy_line
    anim = FuncAnimation(fig, update, frames=num_steps, interval=10, blit=True)
    plt.tight_layout()
    return anim

if __name__ == "__main__":
    N = 50
    temperature = 1.5
    num_steps = 50000
    spins = np.random.choice([-1, 1], size=N)
    h = np.clip(np.random.normal(loc=0, scale=0.67, size=N), -2, 2)
    J = np.clip(np.random.normal(loc=0, scale=0.33, size=(N, N)), -1, 1)
    np.fill_diagonal(J, 0)
    anim = simulate_and_animate(spins, J, h, temperature, num_steps)
    plt.show()
