import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand

def initialise(N):
    return np.random.choice([-1,1], N)

def contr(N):
    cont = np.random.choice([-1, 1], size=N)  # Directly create the array of size N
    return cont

def adj_matrix(N, sparsity):
    adj = np.random.choice([-1, 1], size=(N, N))
    for _ in range(int(np.floor(sparsity * N))):
        a = np.random.randint(0, N)
        b = np.random.randint(0, N)
        adj[a, b] = 0
    adj = np.triu(adj)  # Keep upper triangular part
    adj += adj.T  # Add to lower triangular part
    np.fill_diagonal(adj, 0)
    return adj


def init_field(N):
    return np.random.choice([-1,1], size=N)

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

N = 50  # number of nodes
time_upper = 200  # time limit
beta = 1.5  # more beta, more randomness
gamma = 0.4  # more the gamma, faster the opinion flips
sparsity = 0.8

config = initialise(N)
cont = contr(N)
adj = adj_matrix(N, sparsity)
field = init_field(N)

energies = np.zeros(time_upper)
mags_arr = np.zeros(time_upper)
avg_field = np.zeros(time_upper)

energies[0] = energy(config, adj, field, N)
mags_arr[0] = mag(config, N)
avg_field[0] = np.mean(field)

for t in range(1, time_upper):
    for j in range(N):  # 1 time step is N Monte-Carlo simulations
        config = move(config, adj, beta, N)
        field = calc_field(field, config, cont, gamma, N)
    
    # Store values for later plotting
    energies[t] = energy(config, adj, field, N)
    mags_arr[t] = mag(config, N)
    avg_field[t] = np.mean(field)

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(range(time_upper), mags_arr, label='Magnetisation vs Time', color='blue')
plt.xlabel('Time')
plt.ylabel('Magnetisation')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(range(time_upper), energies, label='Energy vs Time', color='red')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(range(time_upper), avg_field, label='Average Self-field vs Time', color='orange')
plt.xlabel('Time')
plt.ylabel('Average Self-field')
plt.legend()

plt.tight_layout()
plt.show()

print(adj)