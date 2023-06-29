import numpy as np
import random


# Define the objective function (MSE between predicted and actual paths)
def objective_function(params, x, y, phi, x_dot, y_dot, phi_dot, teta, M):
    fc, tau_f = params

    # Calculate the predicted paths based on the equations
    x_pred = integrate_path(fc, tau_f, x_dot[0], teta, M)
    y_pred = integrate_path(fc, tau_f, y_dot[0], teta, M)
    phi_pred = integrate_path(fc, tau_f, phi_dot[0], teta, M)

    # Calculate the mean squared error (MSE) between predicted and actual paths
    mse = np.mean((x_pred - x) ** 2 + (y_pred - y) ** 2 + (phi_pred - phi) ** 2)

    return mse


# Helper function to integrate the path based on the equations
def integrate_path(fc, tau_f, initial_value, teta, M):
    dt = 0.01  # Time step for integration
    path = [initial_value]  # Initialize the path with the initial value

    for i in range(1, len(teta)):
        x_dot = path[-1]
        y_dot = (fc / M) * np.sin(teta[i])
        phi_dot = (1 / (M * (x_dot ** 2 + y_dot ** 2))) * (
                    (fc * (x_dot * np.sin(teta[i]) - y_dot * np.cos(teta[i]))) + tau_f)

        # Integrate using Euler's method
        path.append(path[-1] + phi_dot * dt)

    return np.array(path)


# Particle Swarm Optimization (PSO) implementation
def pso_optimization(x, y, phi, x_dot, y_dot, phi_dot, teta, M, num_particles, max_iterations):
    # Define the search space bounds
    fc_lower_bound = 0.0
    fc_upper_bound = 1.0
    tau_f_lower_bound = 0.0
    tau_f_upper_bound = 10.0

    # Initialize the particles
    particles = []
    for _ in range(num_particles):
        particle = {
            'position': [random.uniform(fc_lower_bound, fc_upper_bound),
                         random.uniform(tau_f_lower_bound, tau_f_upper_bound)],
            'velocity': [0, 0],
            'best_position': [0, 0],
            'best_fitness': float('inf')
        }
        particles.append(particle)

    # Initialize the global best position and fitness
    global_best_position = [0, 0]
    global_best_fitness = float('inf')

    # PSO optimization loop
    for _ in range(max_iterations):
        for particle in particles:
            # Update the particle's position and velocity
            for i in range(2):
                # Update velocity
                inertia_weight = 0.5
                cognitive_weight = 1.0
                social_weight = 1.0
                particle['velocity'][i] = (inertia_weight * particle['velocity'][i]
                                           + cognitive_weight * random.uniform(0, 1) * (
                                                       particle['best_position'][i] - particle['position'][i])
                                           + social_weight * random.uniform(0, 1) * (
                                                       global_best_position[i] - particle['position'][i]))

                # Update position
                particle['position'][i] += particle['velocity'][i]

                # Clip the position within the search space bounds
                particle['position'][i] = np.clip(particle['position'][i], fc_lower_bound, fc_upper_bound)

            # Evaluate the fitness of the particle's position
            fitness = objective_function(particle['position'], x, y, phi, x_dot, y_dot, phi_dot, teta, M)

            # Update the particle's best position and fitness
            if fitness < particle['best_fitness']:
                particle['best_position'] = particle['position'].copy()
                particle['best_fitness'] = fitness

            # Update the global best position and fitness
            if fitness < global_best_fitness:
                global_best_position = particle['position'].copy()
                global_best_fitness = fitness

    return global_best_position


# Example usage
x = [...]  # Actual x values
y = [...]  # Actual y values
phi = [...]  # Actual phi values
x_dot = [...]  # Actual x_dot values
y_dot = [...]  # Actual y_dot values
phi_dot = [...]  # Actual phi_dot values
teta = [...]  # Steering angle values
M = 1.0  # Known value for M

num_particles = 50
max_iterations = 100

# Run PSO optimization
best_params = pso_optimization(x, y, phi, x_dot, y_dot, phi_dot, teta, M, num_particles, max_iterations)

# Print the optimized parameters
print("Optimized Parameters:")
print("f_c:", best_params[0])
print("tau_f:", best_params[1])