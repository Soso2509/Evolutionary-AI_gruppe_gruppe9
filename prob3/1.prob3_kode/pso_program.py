import numpy as np
import pandas as pd
import random
import math

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

# Euclidean distance function
def euclidean_distance(c1, c2):
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

# VRPTW class handling the problem constraints
class VRPTW:
    def __init__(self, customers, depot, capacity):
        self.customers = customers  # list of customers with (x, y) coordinates
        self.depot = depot  # depot coordinates
        self.capacity = capacity  # max capacity per vehicle

# Particle class representing a solution in PSO
class Particle:
    def __init__(self, vrptw):
        self.vrptw = vrptw
        self.position = self.initialize_position()  # Solution representation
        self.velocity = []  # Velocity for set-based PSO
        self.p_best_position = self.position  # Personal best position
        self.p_best_fitness = float('inf')  # Personal best fitness score
        self.fitness = float('inf')

    def initialize_position(self):
        # Initialize routes with random or nearest neighbor heuristic
        position = []
        remaining_customers = set(c['id'] for c in self.vrptw.customers)
        while remaining_customers:
            route = []
            load = 0
            while remaining_customers:
                customer_id = random.choice(list(remaining_customers))
                customer = next(c for c in self.vrptw.customers if c['id'] == customer_id)
                if load + customer['demand'] <= self.vrptw.capacity:
                    route.append(customer_id)
                    load += customer['demand']
                    remaining_customers.remove(customer_id)
                else:
                    break
            position.append(route)
        return position

    def calculate_fitness(self):
        # Calculate fitness (number of vehicles + total distance)
        num_vehicles = len(self.position)
        total_distance = self.calculate_total_distance()
        self.fitness = num_vehicles + total_distance
        if self.fitness < self.p_best_fitness:
            self.p_best_position = self.position
            self.p_best_fitness = self.fitness

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
    def __init__(self, vrptw, num_particles, num_iterations, inertia_weight=0.7, cognitive_weight=1, social_weight=3):
        self.vrptw = vrptw
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.g_best_position = None  # Global best position
        self.g_best_fitness = float('inf')  # Global best fitness
        self.particles = [Particle(vrptw) for _ in range(num_particles)]

    def optimize(self):
        for iteration in range(self.num_iterations):
            for particle in self.particles:
                particle.calculate_fitness()

                # Initialize or update the global best position
                if self.g_best_position is None or particle.fitness < self.g_best_fitness:
                    self.g_best_position = particle.position.copy()  # Ensure it's a copy
                    self.g_best_fitness = particle.fitness

                # Update velocity and position safely
                self.update_velocity(particle)
                self.update_position(particle)
                #print(particle.position)

            print(f"Iteration {iteration}: Best Fitness = {self.g_best_fitness}")
        self.print_best_solution()

    def update_velocity(self, particle):
        new_velocity = []

        # Ensure g_best_position and particle.position have the same length
        num_routes = min(len(particle.position), len(self.g_best_position))

        for i in range(num_routes):
            current_route = set(particle.position[i])
            p_best_route = set(particle.p_best_position[i])

            # Ensure g_best_position is not accessed out of range
            g_best_route = set(self.g_best_position[i])

            # Cognitive and social terms (set differences)
            cognitive_term = p_best_route.difference(current_route)
            social_term = g_best_route.difference(current_route)

            # Inertia component (handle if the particle's velocity exists)
            inertia_component = set(particle.velocity[i]) if particle.velocity else set()

            # Combine the components (use set union to simulate adding new arcs)
            new_velocity.append(list(inertia_component.union(cognitive_term).union(social_term)))

        # Handle any case where particle has more routes than g_best (if needed)
        if len(particle.position) > len(self.g_best_position):
            new_velocity.extend(particle.position[len(self.g_best_position):])

        particle.velocity = new_velocity
        #print(particle.velocity)

    import random
    import math

    def update_position(self, particle):
        # Iterate over the velocity to apply changes
        for route_index, velocity_suggestions in enumerate(particle.velocity):
            if not velocity_suggestions:
                continue  # Skip if no suggestions in this route

            # How many of the suggested changes from the velocity will be used
            num_changes = math.ceil(0.3 * len(velocity_suggestions))

            # Select random 5% of the suggested changes
            selected_changes = random.sample(velocity_suggestions, num_changes)

            # Process each selected change (customer ID)
            for customer_id in selected_changes:
                # Check if customer is already in any route and remove them
                for route in particle.position:
                    if customer_id in route:
                        route.remove(customer_id)
                        break  # No need to check other routes once the customer is removed

                # Append the customer to the current route (as per velocity suggestion)
                if customer_id not in particle.position[route_index]:
                    particle.position[route_index].append(customer_id)

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

    def check_constraints(self, route):
        # Check if the route satisfies VRPTW constraints (capacity, time windows, etc.)
        total_demand = 0
        for customer_id in route:
            customer = next(c for c in self.vrptw.customers if c['id'] == customer_id)
            total_demand += customer['demand']
            if total_demand > self.vrptw.capacity:
                return False
        return True

    def construct_route(self, particle, route_index):
        # Construct a route with VRPTW constraints
        route = []
        remaining_customers = set(c['id'] for c in self.vrptw.customers if c['id'] not in particle.position[route_index])
        load = 0
        while remaining_customers:
            customer_id = random.choice(list(remaining_customers))
            customer = next(c for c in self.vrptw.customers if c['id'] == customer_id)
            if load + customer['demand'] <= self.vrptw.capacity:
                route.append(customer_id)
                load += customer['demand']
                remaining_customers.remove(customer_id)
            else:
                break
        return route

# Main execution
customers, depot = load_data('../1.prob3_data/data.csv')
vrptw_instance = VRPTW(customers, depot, capacity=200)

pso = PSO(vrptw_instance, num_particles=10, num_iterations=200000)
pso.optimize()
