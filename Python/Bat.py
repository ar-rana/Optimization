import numpy as np

# Define the objective function 
def objective_function(x):
    return np.sum(x**2)

# Initialize the bat population

def initialize_bats(n_bats, dim, lower_bound, upper_bound, f_min, f_max, A0, r0):

    bats = np.random.uniform(lower_bound, upper_bound, (n_bats, dim))
    velocities = np.zeros((n_bats, dim))
    frequencies = np.random.uniform(f_min, f_max, n_bats)  # Initialize frequencies
    pulse_rates = r0 * np.ones(n_bats)  # Initialize pulse rates
    loudness = A0 * np.ones(n_bats)  # Initialize loudness

    return bats, velocities, frequencies, pulse_rates, loudness

# Update position and velocity

def update_position_velocity(bats, velocities, frequencies, best_bat, lower_bound, upper_bound):

    velocities += (bats - best_bat) * frequencies[:, np.newaxis]  # Velocity update
    bats += velocities  # Position update

    # Apply boundaries
    bats = np.clip(bats, lower_bound, upper_bound)

    return bats, velocities

# Local search

def local_search(bat, best_bat, avg_loudness):
    epsilon = np.random.uniform(-1, 1, bat.shape)
    return best_bat + epsilon * avg_loudness

# Bat algorithm main loop start

def bat_algorithm(n_bats, dim, lower_bound, upper_bound, max_iter, f_min=0, f_max=100, alpha=0.9, gamma=0.9, A0=1, r0=0.5):

    # Initialize bats
    bats, velocities, frequencies, pulse_rates, loudness = initialize_bats(n_bats, dim, lower_bound, upper_bound, f_min, f_max, A0, r0)

    fitness = np.array([objective_function(bat) for bat in bats])

    best_bat = bats[np.argmin(fitness)]

    best_fitness = np.min(fitness)

    for t in range(max_iter):
        for i in range(n_bats):
            # Generate new solutions
            bats, velocities = update_position_velocity(bats, velocities, frequencies, best_bat, lower_bound, upper_bound)

            if np.random.rand() > pulse_rates[i]:
                # Perform a local search
                avg_loudness = np.mean(loudness)
                new_bat = local_search(bats[i], best_bat, avg_loudness)

            else:
                new_bat = bats[i]
            new_fitness = objective_function(new_bat)

            if np.random.rand() < loudness[i] and new_fitness < fitness[i]:
                # Accept the new solution
                bats[i] = new_bat
                fitness[i] = new_fitness
                loudness[i] *= alpha  # Update loudness using At+1 = α * At
                pulse_rates[i] = r0 * (1 - np.exp(-gamma * t))  # Update pulse rate using rt+1 = r0 * (1 - exp(-γ * t))

            if new_fitness < best_fitness:
                best_bat = new_bat
                best_fitness = new_fitness
    return best_bat, best_fitness

# Parameters

n_bats = 20

dim = 5

lower_bound = -10
upper_bound = 10
max_iter = 1000

best_solution, best_value = bat_algorithm(n_bats, dim, lower_bound, upper_bound, max_iter)

print("Best solution found:", best_solution)
print("Best objective value:", best_value)

