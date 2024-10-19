import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

# Load data from CSV
def load_data(file_name):
    data = pd.read_csv(file_name)
    customers = []
    depot = None

    # Saves the depot and customers in different arrays so that it's easier to use the data later
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

# Euclidean distance function
def euclidean_distance(c1, c2):
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)     # Formula for calculating the distance between two different points with basis in X and Y positions

# VRPTW class handling the problem constraints
class VRPTW:        # Creates class to create objects so that the problem can be handled easier
    def __init__(self, customers, depot, capacity):
        self.customers = customers  # list of customers with (x, y) coordinates
        self.depot = depot  # depot coordinates
        self.capacity = capacity  # max capacity per vehicle

    # Sort route by time window. This is done to remove an additional dimention of the problem, since more dimentions means more different solutions. It's easier this way
    def sort_route_by_time_window(self, route):
        """Sort the customers in a route based on their time window (ready time)."""
        route.sort(key=lambda customer_id: self.get_time_priority(customer_id))

    # Get the time window, more specifically ready time from a customer
    def get_time_priority(self, customer_id):
        """Return the time window's priority for a customer."""
        customer = next(c for c in self.customers if c['id'] == customer_id)
        ready_time = customer['time_window'][1]  # 0 sorts based on Ready time, 1 sorts based on Due time
        return ready_time

# Particle class representing a solution in PSO
class Particle:
    def __init__(self, vrptw):
        self.vrptw = vrptw
        self.position = self.initialize_position()  # Solution representation
        self.velocity = []  # Velocity for set-based PSO
        self.p_best_position = self.position  # Personal best position
        self.p_best_fitness = float('inf')  # Personal best fitness score
        self.fitness = float('inf')  # Current fitness score
        self.non_improvement_count = 0  # Counter for tracking lack of improvement

    def initialize_position(self):
        position = []
        remaining_customers = set(c['id'] for c in self.vrptw.customers if c['id'] != 0)  # Exclude depot

        if PSO.first_position_array is None:
            # Initialize the first particle's routes and save it to first_position_array.
            current_customer = self.vrptw.depot  # Start from the depot
            while remaining_customers:
                route = []
                load = 0  # Variable for the capacity which will be increasing until it's past the max capacity in every route
                while remaining_customers:
                    nearest_customers = self.get_nearest_customers(current_customer, remaining_customers) # While initializing it looks for a customer which is (1/x)% of the closest in the list of remaining customers
                    if not nearest_customers:
                        break

                    customer_id = random.choice(
                        nearest_customers)  # Chose random customer which isn't located too far away from the current customer
                    customer = next(c for c in self.vrptw.customers if c['id'] == customer_id)

                    if load + customer[
                        'demand'] <= self.vrptw.capacity:  # As long as the new customer doesn't breach the max capacity, add it to the route
                        route.append(customer_id)
                        load += customer['demand']
                        remaining_customers.remove(customer_id)
                        current_customer = customer
                    else:
                        break  # Start on the next route if the addition of the new customer would breach the max capacity

                position.append(route)  # Add the current route to the particle position

            # Save the first position array (all routes of the first particle)
            PSO.first_position_array = [route.copy() for route in position]
        else:
            # Initialize based on first_position_array, it select a random customer from the corresponding route in first_position_array in order to keep the routes more consistent
            # This way the modifications done by the PSO will not be for naught due to all of the particles having completely different general area for their routes
            for route in PSO.first_position_array:
                # Pick a random customer from the same route in first_position_array
                first_customer_id = random.choice(route)  # Randomly select a starting customer from this route
                position.append([first_customer_id])  # Start the new particle's route with the chosen customer

                # Remove the selected customer from remaining customers
                if first_customer_id in remaining_customers:
                    remaining_customers.remove(first_customer_id)

                # Find ID and demand of the different customers
                current_customer = next(c for c in self.vrptw.customers if c['id'] == first_customer_id)
                load = next(c['demand'] for c in self.vrptw.customers if c['id'] == first_customer_id)

                while remaining_customers:
                    nearest_customers = self.get_nearest_customers(current_customer, remaining_customers)
                    if not nearest_customers:
                        break

                    customer_id = random.choice(nearest_customers)
                    customer = next(c for c in self.vrptw.customers if c['id'] == customer_id)

                    if load + customer['demand'] <= self.vrptw.capacity:
                        position[-1].append(customer_id)
                        load += customer['demand']
                        remaining_customers.remove(customer_id)
                        current_customer = customer
                    else:
                        break

        return position


    def get_nearest_customers(self, current_customer, remaining_customers):
        """Find the nearest customers and return the closest (1/X)%"""
        distances = []
        for customer_id in remaining_customers:
            customer = next(c for c in self.vrptw.customers if c['id'] == customer_id)
            distance = euclidean_distance(current_customer['coord'], customer['coord'])
            distances.append((customer_id, distance))

        # Sort by distance and select the closest half
        distances.sort(key=lambda x: x[1])
        halfway_index = max(1, len(distances) // 6)  # Ensure at least one customer is selected
        closest_half = [customer_id for customer_id, _ in distances[:halfway_index]]

        return closest_half

    def calculate_fitness(self):
        """Calculates fitnes and give penalties on time window breaches"""
        # Sort each route by the time window before calculating fitness
        # This removes a dimention of search so that every route doesn't have to specify the spot for the customer
        for route in self.position:
            self.vrptw.sort_route_by_time_window(route)

        #num_vehicles = len(self.position)  # Number of vehicles used
        total_distance = self.calculate_total_distance()  # Total distance traveled by all vehicles

        # Time window violation penalty variables
        time_window_penalty = 0
        penalty_per_violation = 1000  # Define the penalty for each time window breach
        large_late_penalty = 5000  # A larger penalty for significant breaches

        # Iterate through each route and calculate arrival times
        for route in self.position:
            current_time = self.vrptw.depot['time_window'][0]  # Start at depot's ready time
            prev_customer = self.vrptw.depot  # Start from the depot

            for customer_id in route:
                customer = next(c for c in self.vrptw.customers if c['id'] == customer_id)

                # Calculate the travel time to the customer
                travel_time = euclidean_distance(prev_customer['coord'], customer['coord'])
                arrival_time = current_time + travel_time

                # If the vehicle arrives before the customer's ready time, it waits
                if arrival_time < customer['time_window'][0]:
                    arrival_time = customer['time_window'][0]  # Wait until the customer is ready

                # Check if the vehicle arrives after the due date
                if arrival_time > customer['time_window'][1]:
                    # Add a penalty for breaching the time window
                    time_window_penalty += penalty_per_violation
                    # Add a larger penalty if the breach is significant (e.g., more than 1 hour late)
                    if arrival_time > customer['time_window'][1] + 60:  # Assuming time in minutes
                        time_window_penalty += large_late_penalty

                # Update the current time after servicing the customer
                current_time = arrival_time + customer['service_time']

                # Update previous customer
                prev_customer = customer

        # Final fitness is total distance plus penalties for time window breaches
        self.fitness = total_distance + time_window_penalty

        # Check for improvement and reset the non-improvement counter if fitness improves
        # They will be killed off if they haven't imrpoved in X iterations as defined in optimize()
        if self.fitness < self.p_best_fitness:
            self.p_best_position = self.position.copy()
            self.p_best_fitness = self.fitness
            self.non_improvement_count = 0  # Reset the counter if there's an improvement
        else:
            self.non_improvement_count += 1  # Increment the counter if no improvement

    # Calculates total distance of a particle
    def calculate_total_distance(self):
        total_distance = 0
        for route in self.position:
            prev_node = self.vrptw.depot
            for customer_id in route:
                customer = next(c for c in self.vrptw.customers if c['id'] == customer_id)
                total_distance += euclidean_distance(prev_node['coord'], customer['coord'])
                prev_node = customer
            total_distance += euclidean_distance(prev_node['coord'], self.vrptw.depot['coord'])  # Return to depot
        return total_distance


# PSO algorithm class
class PSO:
    first_position_array = None  # Shared across particles

    def __init__(self, vrptw, num_particles, num_iterations, inertia_weight=0.7, cognitive_weight=0.8, social_weight=1.0):
        self.vrptw = vrptw
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.g_best_position = None  # Global best position
        self.g_best_fitness = float('inf')  # Global best fitness
        self.particles = [Particle(vrptw) for _ in range(num_particles)]

    # Plots the best solution
    def plot_best_solution(self):
        """Plots the best particle (global best solution) on a 2D plane."""
        depot_coord = self.vrptw.depot['coord']

        # Set up the plot
        plt.figure(figsize=(10, 8))

        # Plot depot
        plt.scatter(*depot_coord, color='red', s=200, label='Depot', zorder=5)
        plt.text(depot_coord[0], depot_coord[1], "Depot", fontsize=12, ha='right')

        # Plot each route
        for i, route in enumerate(self.g_best_position):
            route_coords = [self.vrptw.depot['coord']]  # Start at the depot
            for customer_id in route:
                customer = next(c for c in self.vrptw.customers if c['id'] == customer_id)
                route_coords.append(customer['coord'])
            route_coords.append(self.vrptw.depot['coord'])  # Return to depot

            # Unpack coordinates for plotting
            xs, ys = zip(*route_coords)

            # Plot the route
            plt.plot(xs, ys, marker='o', label=f'Vehicle {i + 1}')

        # Add labels and legend
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Best Routes in VRPTW Solution')
        plt.legend()
        plt.grid(True)
        plt.show()

    def optimize(self):
        global_best_particle_idx = None  # To track the global best particle's index

        for iteration in range(self.num_iterations):
            for i, particle in enumerate(self.particles):
                # Sort each route by the time window before fitness evaluation
                for route in particle.position:
                    self.vrptw.sort_route_by_time_window(route)

                # Calculate fitness after sorting
                particle.calculate_fitness()

                # Initialize or update the global best position
                if self.g_best_position is None or particle.fitness < self.g_best_fitness:
                    self.g_best_position = particle.position.copy()  # Ensure it's a copy
                    self.g_best_fitness = particle.fitness
                    global_best_particle_idx = i  # Update the index of the global best particle

                # Kill and replace particles after 200 iterations of no improvement
                # Skip killing the global best particle
                if i != global_best_particle_idx and particle.non_improvement_count >= 400:
                    print(f"Killing particle {i} due to no improvement after 400 iterations.")
                    self.particles[i] = Particle(self.vrptw)  # Replace with a new particle

                # Update velocity and position safely
                self.update_velocity(particle)
                self.update_position(particle)

            print(f"Iteration {iteration}: Best Fitness = {self.g_best_fitness}")

        self.print_best_solution()
        self.plot_best_solution()

    def update_velocity(self, particle):
        new_velocity = []

        # Ensure g_best_position and particle.position have the same length
        num_routes = min(len(particle.position), len(self.g_best_position))

        for i in range(num_routes):
            current_route = set(particle.position[i])
            p_best_route = set(particle.p_best_position[i])
            g_best_route = set(self.g_best_position[i])

            # Cognitive term: Remove a percentage of personal best difference
            cognitive_diff = list(p_best_route.difference(current_route))
            remove_cognitive_size = math.floor(
                (1 - self.cognitive_weight) * len(cognitive_diff))  # Determine how many to remove
            cognitive_part = cognitive_diff  # Start with the full difference
            if remove_cognitive_size > 0:
                remove_cognitive = random.sample(cognitive_diff, remove_cognitive_size)  # Randomly remove elements
                cognitive_part = [x for x in cognitive_diff if x not in remove_cognitive]  # Keep the rest

            # Social term: Remove a percentage of global best difference
            social_diff = list(g_best_route.difference(current_route))
            remove_social_size = math.floor((1 - self.social_weight) * len(social_diff))  # Determine how many to remove
            social_part = social_diff  # Start with the full difference
            if remove_social_size > 0:
                remove_social = random.sample(social_diff, remove_social_size)  # Randomly remove elements
                social_part = [x for x in social_diff if x not in remove_social]  # Keep the rest

            # Inertia: Keep the same as before
            inertia_part = particle.velocity[i] if particle.velocity else []

            # Combine the components into new velocity
            new_velocity.append(list(set(inertia_part + cognitive_part + social_part)))

        # Handle any case where particle has more routes than g_best (if needed)
        if len(particle.position) > len(self.g_best_position):
            new_velocity.extend(particle.position[len(self.g_best_position):])

        particle.velocity = new_velocity

    def update_position(self, particle):
        # Track the current load for each route
        route_loads = [sum(next(c for c in self.vrptw.customers if c['id'] == customer_id)['demand']
                           for customer_id in route) for route in particle.position]

        # Iterate over the velocity to apply changes
        for route_index, velocity_suggestions in enumerate(particle.velocity):
            if not velocity_suggestions:
                continue  # Skip if no suggestions in this route

            # How many of the suggested changes from the velocity will be used
            num_changes = math.ceil(0.7 * len(velocity_suggestions))

            # Select a random subset of the suggested changes
            selected_changes = random.sample(velocity_suggestions, num_changes)

            # Process each selected change (customer ID)
            for customer_id in selected_changes:
                # Find the customer details
                customer = next(c for c in self.vrptw.customers if c['id'] == customer_id)

                # Check if adding this customer will exceed the capacity of the current vehicle
                if route_loads[route_index] + customer['demand'] <= self.vrptw.capacity:
                    # If it doesn't exceed capacity, remove customer from current route (if exists)
                    for route in particle.position:
                        if customer_id in route:
                            route.remove(customer_id)
                            break  # No need to check other routes once the customer is removed

                    # Append the customer to the current route (as per velocity suggestion)
                    particle.position[route_index].append(customer_id)
                    # Update the route load
                    route_loads[route_index] += customer['demand']

        # Ensure no customer appears in multiple routes after update
        self.remove_duplicates(particle)

    def remove_duplicates(self, particle):
        """Ensure no customer appears in multiple routes."""
        seen_customers = set()
        for route in particle.position:
            unique_route = []
            for customer in route:
                if customer not in seen_customers:
                    unique_route.append(customer)
                    seen_customers.add(customer)
            # Update the route with only unique customers
            route.clear()
            route.extend(unique_route)

    def print_best_solution(self):
        print("\nBest Position (Routes):")
        for route in self.g_best_position:
            print(f"Route: {route}")
        print(f"Best Fitness: {self.g_best_fitness}")

# Main execution
customers, depot = load_data('../1.prob3_data/data.csv')
vrptw_instance = VRPTW(customers, depot, capacity=200)

pso = PSO(vrptw_instance, num_particles=50, num_iterations=2000)
pso.optimize()
