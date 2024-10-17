import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

# Find the path for this script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, '../1.prob3_data/data.csv')

# Load the uploaded file
data = pd.read_csv(file_path)

# Extract necessary data for distances and demands
locations = data[['XCOORD.', 'YCOORD.']].values
demands = data['DEMAND'].values
ready_time = data['READY TIME'].values
due_time = data['DUE DATE'].values
service_time = data['SERVICE TIME'].values

# Number of customers including the depot
num_customers = len(data)

# Initialize a distance matrix (Euclidean distances)
def calculate_distance_matrix(locations):
    num_customers = len(locations)
    dist_matrix = np.zeros((num_customers, num_customers))
    for i in range(num_customers):
        for j in range(num_customers):
            dist_matrix[i, j] = np.sqrt((locations[i, 0] - locations[j, 0]) ** 2 + (locations[i, 1] - locations[j, 1]) ** 2)
    return dist_matrix

distance_matrix = calculate_distance_matrix(locations).astype(np.float32)

# Initialize pheromone matrix (small positive value)
pheromone_matrix = np.ones((num_customers, num_customers)) * 0.1

# Heuristic matrix (1 / distance)
heuristic_matrix = 1 / (distance_matrix + np.finfo(float).eps)  # Avoid division by zero

# Parameters for ACO
alpha = 1    # Influence of pheromone
beta = 5     # Influence of heuristic (distance)
rho = 0.5    # Evaporation rate
num_vehicles = 3  # Number of vehicles
vehicle_capacity = 700  # Vehicle capacity
num_ants = 10  # Number of ants
num_iterations = 50  # Number of iterations
Q = 100  # Pheromone deposit factor

# Check feasibility of adding a customer (capacity and time window)
def is_feasible(route, next_customer, current_time, current_load):
    if current_load + demands[next_customer] > vehicle_capacity:
        return False
    last_customer = route[-1]
    travel_time = distance_matrix[last_customer, next_customer]
    arrival_time = current_time + travel_time

    # Directly check against ready time and due time
    return arrival_time >= ready_time[next_customer] and arrival_time + service_time[next_customer] <= due_time[next_customer]

# Construct route for a single ant
def construct_single_ant_route(pheromone_matrix):
    vehicle_routes = [[] for _ in range(num_vehicles)]
    vehicle_times = np.zeros(num_vehicles)
    vehicle_loads = np.zeros(num_vehicles)
    visited = np.zeros(num_customers, dtype=bool)
    visited[0] = True

    while np.sum(visited) < num_customers:
        for v in range(num_vehicles):
            if np.sum(visited) == num_customers:
                break

            current_route = vehicle_routes[v]
            if len(current_route) == 0:
                current_route.append(0)

            last_customer = current_route[-1]
            feasible_customers = [
                next_customer for next_customer in range(1, num_customers)
                if not visited[next_customer] and is_feasible(current_route, next_customer, vehicle_times[v], vehicle_loads[v])
            ]

            if not feasible_customers:
                current_route.append(0)
                break

            pheromone = pheromone_matrix[last_customer, feasible_customers] ** alpha
            heuristic = heuristic_matrix[last_customer, feasible_customers] ** beta
            probabilities = pheromone * heuristic
            probabilities /= np.sum(probabilities)
            next_customer = np.random.choice(feasible_customers, p=probabilities)

            current_route.append(next_customer)
            visited[next_customer] = True
            travel_time = distance_matrix[last_customer, next_customer]
            vehicle_times[v] += travel_time
            vehicle_times[v] = max(vehicle_times[v], ready_time[next_customer]) + service_time[next_customer]
            vehicle_loads[v] += demands[next_customer]

    for v in range(num_vehicles):
        vehicle_routes[v].append(0)

    return vehicle_routes

# Construct routes for all ants
def construct_routes(pheromone_matrix):
    all_routes = []
    for _ in range(num_ants):
        route = construct_single_ant_route(pheromone_matrix)
        all_routes.append(route)
    return all_routes

# Pheromone update function
def update_pheromone(pheromone_matrix, best_routes, distance_matrix):
    pheromone_matrix *= (1 - rho)
    for route in best_routes:
        route_length = sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
        for i in range(len(route) - 1):
            pheromone_matrix[route[i], route[i + 1]] += Q / route_length
    return pheromone_matrix

start_time = time.time()

# Run ACO for several iterations
best_routes = None
best_distance = float('inf')
best_distances = []
no_improvement_count = 0  # Early stopping counter

for iteration in range(num_iterations):
    all_routes = construct_routes(pheromone_matrix)
    iteration_best_distance = float('inf')

    # Calculate distance for each set of routes and update pheromones
    for vehicle_routes in all_routes:
        total_distance = 0
        for route in vehicle_routes:
            total_distance += sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))

        if total_distance < best_distance:
            best_distance = total_distance
            best_routes = vehicle_routes

        iteration_best_distance = min(iteration_best_distance, total_distance)

    best_distances.append(iteration_best_distance)
    pheromone_matrix = update_pheromone(pheromone_matrix, best_routes, distance_matrix)

    # Early stopping if no improvement for 10 iterations
    if iteration_best_distance >= best_distance:
        no_improvement_count += 1
    else:
        no_improvement_count = 0

    if no_improvement_count >= 10:
        print(f"Convergence reached at iteration {iteration}")
        break

# Output the best routes and their distance
print("Best routes:", best_routes)
print("Best total distance:", best_distance)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total runtime: {elapsed_time:.2f} seconds")

# Example: Create color map for more than 6 vehicles
color_map =  mpl.colormaps['plasma']

def plot_multiple_routes_with_cmap(vehicle_routes, locations):
    plt.figure(figsize=(10, 6))

    for i, loc in enumerate(locations):
        if i == 0:
            plt.plot(loc[0], loc[1], 'rs', markersize=10, label="Depot")
        else:
            plt.plot(loc[0], loc[1], 'bo')

    # Plot each vehicle's route using the colormap
    for idx, route in enumerate(vehicle_routes):
        color = color_map(idx)

        for i in range(len(route) - 1):
            start, end = route[i], route[i+1]
            plt.plot([locations[start][0], locations[end][0]],
                     [locations[start][1], locations[end][1]],
                     '-', color=color, linewidth=2, label=f"Vehicle {idx+1}" if i == 0 else "")

    plt.title('Vehicle Routes with Colormap')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot using colormap
plot_multiple_routes_with_cmap(best_routes, locations)

# Plot convergence curve
plt.figure(figsize=(8, 5))
plt.plot(best_distances, label='Best Distance')
plt.title('Convergence of ACO')
plt.xlabel('Iteration')
plt.ylabel('Best Distance')
plt.legend()
plt.grid(True)
plt.show()
