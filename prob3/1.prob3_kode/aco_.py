import os
import numpy as np
import pandas as pd
import random
from math import sqrt, exp

# Load data from CSV
def load_data(file_name):
    data = pd.read_csv(file_name)
    customers = []
    depot = None

    for _, row in data.iterrows():
        if row['CUST NO.'] == 0:
            depot = {
                'id': row['CUST NO.'],
                'coord': (row['XCOORD.'], row['YCOORD.']),
                'demand': row['DEMAND'],
                'service_time': row['SERVICE TIME'],
                'time_window': (row['READY TIME'], row['DUE DATE'])
            }
        else:
            customers.append({
                'id': row['CUST NO.'],
                'coord': (row['XCOORD.'], row['YCOORD.']),
                'demand': row['DEMAND'],
                'service_time': row['SERVICE TIME'],
                'time_window': (row['READY TIME'], row['DUE DATE'])
            })
    return customers, depot

# Find the path for this script
script_dir = os.path.dirname(os.path.abspath(__file__))


# Get data from CSV
customers, depot = load_data(os.path.join(script_dir, '../1.prob3_data/data.csv'))

num_customers = len(customers)
vehicle_capacity = 2000

# ACO Parameters
num_ants = 10
num_iterations = 100
alpha = 1  # Pheromone importance
beta = 2   # Heuristic importance (distance)
rho = 0.5  # Evaporation rate
Q = 100    # Pheromone constant


# Euclidean distance function
def euclidean_distance(c1, c2):
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

# Distance matrix
distance_matrix = np.zeros((num_customers, num_customers))
for i in range(num_customers):
    for j in range(num_customers):
        distance_matrix[i, j] = euclidean_distance(customers[i]['coord'], customers[j]['coord'])

# Pheromone matrix
pheromone = np.ones((num_customers, num_customers))

# Heuristic (inverse of distance)
heuristic = 1 / (distance_matrix + np.eye(num_customers))

# Feasibility check for capacity and time windows
def is_feasible(route, next_customer, vehicle_load, current_time):
    customer = customers[next_customer]
    if vehicle_load + customer['demand'] > vehicle_capacity:
        return False
    arrival_time = current_time + distance_matrix[route[-1]][next_customer]
    if arrival_time > customer['time_window'][1]:
        return False
    return True

# Route cost calculation (total distance)
def route_cost(route):
    cost = 0
    for i in range(len(route) - 1):
        cost += distance_matrix[route[i]][route[i+1]]
    return cost

# Ant Solution Construction
def ant_solution():
    route = [0]  # Start from depot (customer 0)
    vehicle_load = 0
    current_time = 0
    while len(route) < num_customers:
        feasible_customers = [c for c in range(1, num_customers) if c not in route and is_feasible(route, c, vehicle_load, current_time)]
        if not feasible_customers:
            break

        # Probabilistic selection of next customer
        probabilities = []
        for c in feasible_customers:
            pheromone_level = pheromone[route[-1]][c]
            heuristic_value = heuristic[route[-1]][c]
            probabilities.append((pheromone_level ** alpha) * (heuristic_value ** beta))

        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()

        next_customer = np.random.choice(feasible_customers, p=probabilities)

        # Update route and load
        route.append(next_customer)
        vehicle_load += customers[next_customer]['demand']
        current_time += distance_matrix[route[-1]][next_customer] + customers[next_customer]['service_time']

    route.append(0)  # Return to depot
    return route

# ACO Main Loop
best_route = None
best_cost = float('inf')

for iteration in range(num_iterations):
    all_routes = []

    for ant in range(num_ants):
        route = ant_solution()
        all_routes.append(route)

        # Update the best solution found
        cost = route_cost(route)
        if cost < best_cost:
            best_cost = cost
            best_route = route

    # Update pheromones
    pheromone *= (1 - rho)

    for route in all_routes:
        cost = route_cost(route)
        for i in range(len(route) - 1):
            pheromone[route[i]][route[i+1]] += Q / cost

print("Best route found:", best_route)
print("Cost of best route:", best_cost)