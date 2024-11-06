import random
import numpy as np

# Define the Particle class used for both PSO and Firefly algorithms
class Particle:
    def __init__(self, position):
        self.position = position
        self.velocity = np.zeros_like(position)
        self.best_position = position
        self.best_fitness = float('inf')

# Objective functions
def F1(x):  # Objective function 1
    return np.sum(x**2)


def F2(x):  # Objective function 2
    return np.max(np.abs(x))

# Hybrid PSO-Firefly Algorithm
def Hybrid_Optimization(ObjF, Pop_size, D, MaxT, LB, UB, gamma=1, beta0=1, alpha=0.1, alpha_damp=0.95):
    swarm_best_position = None
    swarm_best_fitness = float('inf')
    particles = []
    
    # Initialize population of particles (for both PSO and Firefly behavior)
    for _ in range(Pop_size):
        position = np.random.uniform(LB, UB, D)
        particle = Particle(position)
        fitness = ObjF(position)
        particle.best_position = position
        particle.best_fitness = fitness
        particles.append(particle)
        if fitness < swarm_best_fitness:
            swarm_best_fitness = fitness
            swarm_best_position = position

    # Hybrid PSO-Firefly Algorithm Main Loop
    for itr in range(MaxT):
        for i, particle in enumerate(particles):
            # PSO velocity and position update
            w = 0.7  # Inertia weight
            c1 = 1.5  # Personal best attraction coefficient
            c2 = 1.5  # Global best attraction coefficient

            r1 = random.random()
            r2 = random.random()

            # Velocity update using PSO
            particle.velocity = (w * particle.velocity +
                                 c1 * r1 * (particle.best_position - particle.position) +
                                 c2 * r2 * (swarm_best_position - particle.position))
            particle.position += particle.velocity

            # Firefly attractiveness-based movement
            for j, other_particle in enumerate(particles):
                if other_particle.best_fitness < particle.best_fitness:
                    r_ij = np.linalg.norm(particle.position - other_particle.position)
                    beta = beta0 * np.exp(-gamma * r_ij**2)
                    epsilon = alpha * (np.random.rand(D) - 0.5) * (UB - LB)
                    
                    particle.position = (particle.position +
                                         beta * (other_particle.position - particle.position) +
                                         epsilon)
                    # Ensure particles stay within bounds
                    particle.position = np.clip(particle.position, LB, UB)

            # Evaluate new fitness
            fitness = ObjF(particle.position)
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position

            # Update global best fitness
            if fitness < swarm_best_fitness:
                swarm_best_fitness = fitness
                swarm_best_position = particle.position

        # Damping the alpha parameter for firefly behavior
        alpha *= alpha_damp
        
        # Display the progress
        print(f"Iteration {itr + 1}: Best Fitness = {swarm_best_fitness}")
    
    return swarm_best_position, swarm_best_fitness


# Test the hybrid algorithm
Objective_Function = {
    'F1': F1,
    'F2': F2
}

# Parameters
Pop_size = 50
MaxT = 50
D = 3 #dimension
LB = -5 
UB = 5

# Run hybrid optimization for both objective functions
for func_name, ObjF in Objective_Function.items():
    print(f"\nRunning Hybrid Optimization on {func_name}:")
    best_position, best_fitness = Hybrid_Optimization(ObjF, Pop_size, D, MaxT, LB, UB)
    print(f"Best Position: {best_position}")
    print(f"Best Fitness: {best_fitness}\n")
