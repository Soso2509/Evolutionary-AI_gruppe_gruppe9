import os
import numpy as np
import pandas as pd
import random
from math import sqrt
import matplotlib.pyplot as plt

# Load data from CSV with error handling
def load_data(file_name):
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"The file {file_name} was not found.")
    data = pd.read_csv(file_name)
    customers = []

    for _, row in data.iterrows():
        customers.append({
            'id': row['CUST NO.'],
            'coord': (row['XCOORD.'], row['YCOORD.']),
            'demand': row['DEMAND'],
            'service_time': row['SERVICE TIME'],
            'time_window': (row['READY TIME'], row['DUE DATE'])
        })
    return customers

# Find the path for this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get data from CSV
customers = load_data(os.path.join(script_dir, '../1.prob3_data/data.csv'))

num_customers = len(customers)
vehicle_capacity = 200
num_vehicles = 25

# ACO Parameters
num_ants = 10
num_iterations = 500
alpha = 1  # Pheromone importance
beta = 2   # Heuristic importance (distance)
rho = 0.6  # Evaporation rate
Q = 100    # Pheromone constant
penalty_factor = 1000  # Penalty for time window violation

# Euclidean distance function
def euclidean_distance(c1, c2):
    return np.sqrt((c1['coord'][0] - c2['coord'][0])**2 + (c1['coord'][1] - c2['coord'][1])**2)

# Distance matrix
distance_matrix = np.zeros((num_customers, num_customers))
for i in range(num_customers):
    for j in range(num_customers):
        distance_matrix[i, j] = euclidean_distance(customers[i], customers[j])

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
    if arrival_time > customer['time_window'][1]:  # Check against due date
        return False
    return True

# Route cost calculation with penalty for time window violation
def route_cost(route):
    cost = 0
    current_time = 0
    for i in range(len(route) - 1):
        start = route[i]
        end = route[i + 1]
        travel_time = distance_matrix[start][end]
        arrival_time = current_time + travel_time

        # Add distance cost
        cost += travel_time

        # Check time window constraint
        if arrival_time > customers[end]['time_window'][1]:  # If arrival is late
            cost += penalty_factor  # Add penalty to cost

        # Update current time to arrival + service time
        current_time = arrival_time + customers[end]['service_time']
    return cost

# Count time window violations and calculate penalty cost for a set of routes
def calculate_violations_and_penalty(routes):
    violation_count = 0
    penalty_cost = 0
    for route in routes:
        current_time = 0
        for i in range(len(route) - 1):
            start = route[i]
            end = route[i + 1]
            travel_time = distance_matrix[start][end]
            arrival_time = current_time + travel_time

            # Check if arrival is late
            if arrival_time > customers[end]['time_window'][1]:
                violation_count += 1
                penalty_cost += penalty_factor

            # Update current time to arrival + service time
            current_time = arrival_time + customers[end]['service_time']
    return violation_count, penalty_cost

# Ant Solution Construction with Route Validation
def ant_solution():
    all_routes = []  # List of routes, one for each vehicle
    visited = set()

    for vehicle in range(num_vehicles):
        route = [0]  # Start from depot (customer 0)
        vehicle_load = 0
        current_time = 0

        while len(visited) < num_customers - 1:  # Continue until all customers are visited
            feasible_customers = [
                c for c in range(1, num_customers)
                if c not in visited and c not in route and is_feasible(route, c, vehicle_load, current_time)
            ]
            if not feasible_customers:
                break  # If no more feasible customers for this vehicle, end its route

            # Probabilistic selection of next customer
            probabilities = []
            for c in feasible_customers:
                pheromone_level = pheromone[route[-1]][c]
                heuristic_value = heuristic[route[-1]][c]
                probabilities.append((pheromone_level ** alpha) * (heuristic_value ** beta))

            probabilities = np.array(probabilities)

            if probabilities.sum() == 0:  # Check for zero-sum to avoid division by zero
                break  # No feasible path forward, break out of this route construction

            probabilities /= probabilities.sum()  # Safe normalization now

            next_customer = np.random.choice(feasible_customers, p=probabilities)

            # Calculate arrival time to the next customer
            travel_time = distance_matrix[route[-1]][next_customer]
            arrival_time = current_time + travel_time

            # If arrival time is earlier than the customer's ready time, wait until ready
            ready_time = customers[next_customer]['time_window'][0]
            if arrival_time < ready_time:
                arrival_time = ready_time

            # Update route, load, and mark the customer as visited
            route.append(next_customer)
            vehicle_load += customers[next_customer]['demand']
            current_time = arrival_time + customers[next_customer]['service_time']
            visited.add(next_customer)

        # Return to depot when vehicle finishes
        route.append(0)
        all_routes.append(route)

        # Stop if all customers are visited
        if len(visited) == num_customers - 1:
            break

    return all_routes

# ACO Main Loop with violation tracking
best_routes = None
best_cost = float('inf')

for iteration in range(num_iterations):
    all_ants_routes = []
    for ant in range(num_ants):
        routes = ant_solution()
        all_ants_routes.append(routes)

        # Compute total cost for this solution (sum of all vehicle routes)
        total_cost = sum(route_cost(route) for route in routes)

        # Calculate violations and penalty cost
        violation_count, penalty_cost = calculate_violations_and_penalty(routes)

        if total_cost < best_cost:
            best_cost = total_cost
            best_routes = routes

        print(f"Iteration {iteration + 1}/{num_iterations}, Ant {ant + 1}/{num_ants}: Total Cost: {total_cost}, Violations: {violation_count}, Penalty Cost: {penalty_cost}, Best Cost: {best_cost}")

    # Update pheromones
    pheromone *= (1 - rho)

    for routes in all_ants_routes:
        total_cost = sum(route_cost(route) for route in routes)
        for route in routes:
            for i in range(len(route) - 1):
                pheromone[route[i]][route[i+1]] += Q / total_cost

# Calculate final violations and penalty cost
violation_count, penalty_cost = calculate_violations_and_penalty(best_routes)

# Output the best routes found
print("\nBest routes found:", best_routes)
print("Cost of best routes:", best_cost)
print("With", violation_count, "Violations, and a Penalty Cost of", penalty_cost)

# Visualization of the best routes found
def plot_routes(routes, customers):
    plt.figure(figsize=(10, 8))

    # Plot all customers
    for customer in customers:
        if customer['id'] == 0:
            plt.scatter(customer['coord'][0], customer['coord'][1], color='red', label='Depot', s=100, zorder=2)
        else:
            plt.scatter(customer['coord'][0], customer['coord'][1], color='blue', label='Customer' if customer['id'] == 1 else "", s=50, zorder=2)
            plt.text(customer['coord'][0] + 0.5, customer['coord'][1] + 0.5, f"{customer['id']}", fontsize=12)

    colors = [
        'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'brown', 'pink', 'lime', 'gray', 'navy', 'turquoise', 'gold', 'coral', 'darkred', 'teal', 'violet', 'olive', 'indigo'
    ]  # Color for each vehicle route

    for i, route in enumerate(routes):
        for j in range(len(route) - 1):
            start_customer = customers[route[j]]
            end_customer = customers[route[j + 1]]
            plt.plot([start_customer['coord'][0], end_customer['coord'][0]],
                     [start_customer['coord'][1], end_customer['coord'][1]],
                     color=colors[i % len(colors)], linewidth=2, label=f'Vehicle {i+1}' if j == 0 else "")

    plt.title(f"Vehicle Routes with ACO (Total Cost: {best_cost:.2f})")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function to plot the best routes
plot_routes(best_routes, customers)