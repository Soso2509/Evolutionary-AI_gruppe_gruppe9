import os
import numpy as np
import pandas as pd
import random
from math import sqrt
import matplotlib.pyplot as plt

# Load data from CSV
def load_data(file_name):
    data = pd.read_csv(file_name)
    customers = []
    depot = None

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
customers= load_data(os.path.join(script_dir, '../1.prob3_data/data.csv'))

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
    all_routes = []  # List of routes, one for each vehicle
    visited = set()

    for vehicle in range(num_vehicles):
        route = [0]  # Start from depot (customer 0)
        vehicle_load = 0
        current_time = 0

        while len(visited) < num_customers - 1:  # Continue until all customers are visited
            feasible_customers = [c for c in range(1, num_customers) if c not in visited and c not in route and is_feasible(route, c, vehicle_load, current_time)]
            if not feasible_customers:
                break  # If no more feasible customers for this vehicle, end its route

           # Probabilistic selection of next customer
            probabilities = []
            for c in feasible_customers:
                pheromone_level = pheromone[route[-1]][c]
                heuristic_value = heuristic[route[-1]][c]
                probabilities.append((pheromone_level ** alpha) * (heuristic_value ** beta))

            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()

            next_customer = np.random.choice(feasible_customers, p=probabilities)

            # Update route, load, and mark the customer as visited
            route.append(next_customer)
            vehicle_load += customers[next_customer]['demand']
            current_time += distance_matrix[route[-1]][next_customer] + customers[next_customer]['service_time']
            visited.add(next_customer)

        # Return to depot when vehicle finishes
        route.append(0)
        all_routes.append(route)

     # If all customers are visited, no need to send another vehicle
        if len(visited) == num_customers - 1:
            break

    return all_routes

# ACO Main Loop
best_routes = None
best_cost = float('inf')

for iteration in range(num_iterations):
    all_ants_routes = []
    for ant in range(num_ants):
        routes = ant_solution()
        all_ants_routes.append(routes)

        # Compute total cost for this solution (sum of all vehicle routes)
        total_cost = sum(route_cost(route) for route in routes)
        if total_cost < best_cost:
            best_cost = total_cost
            best_routes = routes

        print("Total cost of route for iteration ", iteration+1," and ant nr ",ant+1, " is ", total_cost ,", best overall cost is ",best_cost)


    # Update pheromones
    pheromone *= (1 - rho)

    for routes in all_ants_routes:
        total_cost = sum(route_cost(route) for route in routes)
        for route in routes:
            for i in range(len(route) - 1):
                pheromone[route[i]][route[i+1]] += Q / total_cost


# Output the best routes found
print()
print("Best routes found:", best_routes)
print("Cost of best routes:", best_cost)

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
        'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'brown', 'pink', 'lime', 'gray', 'navy', 'turquoise',
        'gold', 'coral', 'darkred', 'teal', 'violet', 'olive', 'indigo'
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