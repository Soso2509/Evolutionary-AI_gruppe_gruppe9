import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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

    def sort_route_by_time_window(self, route):
        """Sort the customers in a route based on their time window (ready time)."""
        route.sort(key=lambda customer_id: self.get_time_priority(customer_id))

    def get_time_priority(self, customer_id):
        """Return the time window's priority for a customer."""
        customer = next(c for c in self.customers if c['id'] == customer_id)
        ready_time = customer['time_window'][0]  # Ready time
        return ready_time

PENALTY_PER_BREACH = 1000  # Adjust this value to control the severity of penalty for each time window breach

# Particle class representing a solution in PSO
class Particle:
    def __init__(self, vrptw, cognitive_weight, social_weight):
        self.vrptw = vrptw
        self.position = self.initialize_position()  # Solution representation
        self.velocity = []  # Velocity for set-based PSO
        self.p_best_position = self.position  # Personal best position
        self.p_best_fitness = float('inf')  # Personal best fitness score
        self.fitness = float('inf')  # Current fitness score
        self.non_improvement_count = 0  # Counter for tracking lack of improvement
        self.cognitive_weight = cognitive_weight  # Adaptive cognitive weight
        self.social_weight = social_weight  # Adaptive social weight

    def initialize_position(self):
        position = []
        remaining_customers = set(c['id'] for c in self.vrptw.customers if c['id'] != 0)  # Exclude depot

        if PSO.first_position_array is None:
            # Initialize the first particle's routes and save it to first_position_array
            current_customer = self.vrptw.depot  # Start from the depot
            while remaining_customers:
                route = []
                load = 0
                while remaining_customers:
                    # Ensure we're passing a customer ID that exists
                    if current_customer['id'] == 0:
                        current_customer_id = random.choice(list(remaining_customers))
                    else:
                        current_customer_id = current_customer['id']

                    nearest_customers = self.get_nearest_customers(current_customer_id, remaining_customers)
                    if not nearest_customers:
                        break

                    customer_id = random.choice(nearest_customers)
                    customer = next(c for c in self.vrptw.customers if c['id'] == customer_id)

                    if load + customer['demand'] <= self.vrptw.capacity:
                        route.append(customer_id)
                        load += customer['demand']
                        remaining_customers.remove(customer_id)
                        current_customer = customer
                    else:
                        break

                position.append(route)

            # Save the first position array without altering it later
            PSO.first_position_array = [route.copy() for route in position]
        else:
            # Initialize based on first_position_array
            for route in PSO.first_position_array:
                first_customer_id = route[0]  # Always start with the first customer of that route
                position.append([first_customer_id])  # Start the new particle's route

                if first_customer_id in remaining_customers:
                    remaining_customers.remove(first_customer_id)

                current_customer = next(c for c in self.vrptw.customers if c['id'] == first_customer_id)
                load = current_customer['demand']

                while remaining_customers:
                    nearest_customers = self.get_nearest_customers(current_customer['id'], remaining_customers)
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

        # Calculate fitness for initial routes to determine the highest fitness route
        highest_fitness_route = max(position, key=lambda route: self.calculate_route_distance(route))

        # Add two extra routes based on highest fitness route, limited to 5 customers each
        for _ in range(3):
            if highest_fitness_route:
                new_route = []
                # Select a random starting customer from the highest fitness route
                start_customer_id = random.choice(highest_fitness_route)
                new_route.append(start_customer_id)
                highest_fitness_route.remove(start_customer_id)

                # Attempt to add up to 4 more customers to the new route
                load = next(c['demand'] for c in self.vrptw.customers if c['id'] == start_customer_id)
                available_customers = {c['id'] for c in self.vrptw.customers} - set(new_route)

                while len(new_route) < 5 and available_customers:
                    nearest_customers = self.get_nearest_customers(new_route[-1], available_customers)
                    if not nearest_customers:
                        break

                    next_customer_id = random.choice(nearest_customers)
                    next_customer = next(c for c in self.vrptw.customers if c['id'] == next_customer_id)

                    if load + next_customer['demand'] <= self.vrptw.capacity:
                        new_route.append(next_customer_id)
                        load += next_customer['demand']
                        available_customers.remove(next_customer_id)
                        # Remove the customer from their original route in `position`
                        for route in position:
                            if next_customer_id in route:
                                route.remove(next_customer_id)
                                break
                    else:
                        break

                position.append(new_route)  # Add the new route to the position

        return position

    def get_nearest_customers(self, current_customer_id, remaining_customers):
        """Find the nearest customers based on customer ID and return the closest subset."""
        current_customer = next((c for c in self.vrptw.customers if c['id'] == current_customer_id), None)
        if current_customer is None:
            raise ValueError(f"Customer ID {current_customer_id} not found in VRPTW customers")

        distances = []
        for customer_id in remaining_customers:
            customer = next(c for c in self.vrptw.customers if c['id'] == customer_id)
            distance = euclidean_distance(current_customer['coord'], customer['coord'])
            distances.append((customer_id, distance))

        # Sort by distance and select the closest subset
        distances.sort(key=lambda x: x[1])
        halfway_index = max(1, len(distances) // 30)
        closest_subset = [customer_id for customer_id, _ in distances[:halfway_index]]

        return closest_subset

    def calculate_route_distance(self, route):
        """Helper function to calculate the total distance of a route."""
        total_distance = 0
        prev_node = self.vrptw.depot
        for customer_id in route:
            customer = next(c for c in self.vrptw.customers if c['id'] == customer_id)
            total_distance += euclidean_distance(prev_node['coord'], customer['coord'])
            prev_node = customer
        total_distance += euclidean_distance(prev_node['coord'], self.vrptw.depot['coord'])
        return total_distance


    def calculate_fitness(self):
        # Sort each route by the time window before calculating fitness
        for route in self.position:
            self.vrptw.sort_route_by_time_window(route)

        # Calculate total distance of the current solution
        total_distance = self.calculate_total_distance()

        # Calculate total time window breaches
        total_breaches = 0
        for route in self.position:
            total_breaches = self.check_time_constraints(route) + total_breaches
        #print(f"{self.position}\nTotal breaches: {total_breaches}\n")

        # Calculate fitness as total distance plus penalties for breaches
        self.fitness = total_distance + (total_breaches * PENALTY_PER_BREACH)

        # Check for improvement and reset the non-improvement counter if fitness improves
        if self.fitness < self.p_best_fitness:
            self.p_best_position = self.position.copy()
            self.p_best_fitness = self.fitness
            self.non_improvement_count = 0  # Reset the counter if there's an improvement
        else:
            self.non_improvement_count += 1  # Increment the counter if no improvement

        return total_distance, self.fitness  # Return both total distance and fitness

    def check_time_constraints(self, route):
        """
        Checks for time window violations in a route.
        Counts breaches for late arrivals at each customer and calculates
        time accurately for each step in the route.
        """
        current_time = 0  # Start at time zero (depot departure time)
        breaches = 0  # Breach counter
        prev_node = self.vrptw.depot  # Start from the depot

        for customer_id in route:
            # Retrieve customer details by ID
            customer = next(c for c in self.vrptw.customers if c['id'] == customer_id)

            # Calculate travel time to this customer
            travel_time = euclidean_distance(prev_node['coord'], customer['coord'])
            #print(f"From {prev_node['id']} to {customer_id}: {travel_time}")
            arrival_time = current_time + travel_time  # Arrival time at the customer

            # Handle arrival time adjustments
            if arrival_time < customer['time_window'][0]:
                # Arrive early; wait until the ready time
                current_time = customer['time_window'][0] + customer['service_time']
            elif arrival_time > customer['time_window'][1]:
                # Late arrival; count this as a breach
                breaches += 1
                # Update current time to account for service time even if late
                current_time = arrival_time + customer['service_time']
                #print(f"Arrival: {arrival_time}, Time window: {customer['time_window'][0]}-{customer['time_window'][1]}")
            elif arrival_time >= customer['time_window'][0] and arrival_time <= customer['time_window'][1]:
                # On-time arrival; add service time
                current_time = arrival_time + customer['service_time']
            else:
                print("Neither on time, too late or too early")
            # Move to the next customer, setting this one as the previous node
            prev_node = customer

        # Add time to return to the depot after visiting the last customer
        return_to_depot_time = euclidean_distance(prev_node['coord'], self.vrptw.depot['coord'])
        current_time += return_to_depot_time

        # Return total number of breaches found in the route
        return breaches

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

    def __init__(self, vrptw, num_particles, num_iterations, inertia_weight=0.7, cognitive_weight=0.6, social_weight=1.0):
        self.vrptw = vrptw
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_weight_initial = cognitive_weight  # Initial cognitive weight
        self.social_weight_initial = social_weight  # Initial social weight
        self.g_best_position = None  # Global best position
        self.g_best_fitness = float('inf')  # Global best fitness
        self.g_best_distance = float('inf')  # Global best distance
        self.iterations_without_improvement = 0  # Track the number of iterations without improvement
        self.particles = [Particle(vrptw, cognitive_weight, social_weight) for _ in range(num_particles)]

    def optimize(self):
        self.g_best_position = None  # Store the global best position directly
        self.g_best_fitness = float('inf')  # Initialize the global best fitness
        self.g_best_distance = float('inf')  # Initialize the global best distance

        for iteration in range(self.num_iterations):
            improvement = False  # Track if there's any improvement in this iteration

            for particle in self.particles:
                # Sort each route by the time window before fitness evaluation
                for route in particle.position:
                    self.vrptw.sort_route_by_time_window(route)

                # Calculate fitness after sorting
                total_distance, fitness = particle.calculate_fitness()

                # Check if this particle's position is the new global best
                if self.g_best_position is None or fitness < self.g_best_fitness:
                    # Save the best routes, fitness score, and total distance
                    self.g_best_position = [route.copy() for route in particle.position]
                    self.g_best_fitness = fitness
                    self.g_best_distance = total_distance
                    self.iterations_without_improvement = 0  # Reset counter for improvement
                    improvement = True

            if not improvement:
                self.iterations_without_improvement += 1  # Increment the counter if no improvement
            else:
                # Reset cognitive and social weights for all particles when improvement is found
                self.reset_weights_for_all_particles()

            # Adjust cognitive and social weights for all particles if no improvement in 30 iterations
            if self.iterations_without_improvement > 70:
                self.adjust_weights_for_stagnation()

            # Print best fitness and distance for this iteration
            print(
                f"Iteration {iteration}: Best Distance = {self.g_best_distance:.2f}, Best Fitness = {self.g_best_fitness:.2f}")

            # Kill and replace particles after a set number of iterations without improvement
            kill_threshold = 120
            for particle in self.particles:
                if particle.non_improvement_count >= kill_threshold:
                    print(f"Killing and replacing particle due to no improvement after {kill_threshold} iterations.")
                    # Replace with a new particle
                    self.particles[self.particles.index(particle)] = Particle(self.vrptw, self.cognitive_weight_initial,
                                                                              self.social_weight_initial)

                # Update velocity and position safely
                self.update_velocity(particle)
                self.update_position(particle)

        self.print_best_solution()
        self.plot_best_solution()

    def adjust_weights_for_stagnation(self):
        """
        Increase cognitive weight and decrease social weight for all particles if no improvement for 100 iterations.
        """
        for particle in self.particles:
            particle.cognitive_weight = min(particle.cognitive_weight + 0.05, 1.0)  # Increase cognitive weight up to 1.5
            particle.social_weight = max(particle.social_weight - 0.05, 0.3)  # Decrease social weight down to 0.3
        print(f"Adjusting weights due to stagnation: Increased cognitive weight, decreased social weight.")

    def reset_weights_for_all_particles(self):
        """
        Gradually reset the cognitive and social weights for all particles to their original values when improvement is found.
        """
        for particle in self.particles:
            particle.cognitive_weight = max(particle.cognitive_weight - 0.05, self.cognitive_weight_initial)
            particle.social_weight = min(particle.social_weight + 0.05, self.social_weight_initial)
        print(f"Resetting weights to initial values: Cognitive = {self.cognitive_weight_initial}, Social = {self.social_weight_initial}")

    def update_velocity(self, particle):
        new_velocity = []
        longest_route_index = self.identify_longest_route(particle)  # Pass the particle to access its methods

        # Ensure g_best_position and particle.position have the same length
        num_routes = min(len(particle.position), len(self.g_best_position))

        for i in range(num_routes):
            current_route = set(particle.position[i])
            p_best_route = set(particle.p_best_position[i])
            g_best_route = set(self.g_best_position[i])

            # Cognitive term: Remove a percentage of personal best difference
            cognitive_diff = list(p_best_route.difference(current_route))
            remove_cognitive_size = math.floor(
                (1 - (particle.cognitive_weight * random.uniform(0, 1))) * len(cognitive_diff)
            )  # Determine how many to remove
            cognitive_part = cognitive_diff
            if remove_cognitive_size > 0:
                remove_cognitive = random.sample(cognitive_diff, remove_cognitive_size)
                cognitive_part = [x for x in cognitive_diff if x not in remove_cognitive]

            # Social term: Remove a percentage of global best difference
            social_diff = list(g_best_route.difference(current_route))
            remove_social_size = math.floor(
                (1 - (particle.social_weight * random.uniform(0, 1))) * len(social_diff)
            )  # Determine how many to remove
            social_part = social_diff
            if remove_social_size > 0:
                remove_social = random.sample(social_diff, remove_social_size)
                social_part = [x for x in social_diff if x not in remove_social]

            # Preference for moving customers out of the longest route
            if i == longest_route_index:
                # Increase the weight for removing customers from the longest route by adding more customers to move out
                extra_removals = math.ceil(len(current_route) * 0.2)  # Adjust the percentage as necessary
                extra_customers = random.sample(list(current_route), min(extra_removals, len(current_route)))
                cognitive_part.extend(extra_customers)  # Add these extra customers to the cognitive part

            # Inertia: Keep the same as before
            inertia_part = particle.velocity[i] if particle.velocity else []

            # Combine the components into new velocity
            new_velocity.append(list(set(inertia_part + cognitive_part + social_part)))

        # Handle any case where particle has more routes than g_best (if needed)
        if len(particle.position) > len(self.g_best_position):
            new_velocity.extend(particle.position[len(self.g_best_position):])

        particle.velocity = new_velocity

    def identify_longest_route(self, particle):
        """Identify the index of the longest route in terms of distance."""
        longest_distance = 0
        longest_route_index = 0
        for i, route in enumerate(particle.position):
            route_distance = particle.calculate_route_distance(route)  # Use particle's method
            if route_distance > longest_distance:
                longest_distance = route_distance
                longest_route_index = i
        return longest_route_index

    def print_best_solution(self):
        print("\nBest Position (Routes) with Sorted Time Windows:")
        sorted_best_position = []

        # Sort each route by time window for a clear view of the schedule
        for route in self.g_best_position:
            sorted_route = route.copy()
            self.vrptw.sort_route_by_time_window(sorted_route)  # Sort route by time window
            sorted_best_position.append(sorted_route)

        # Display each sorted route
        for i, route in enumerate(sorted_best_position):
            print(f"Vehicle {i + 1} Route (sorted by time): {route}")

        print(f"Best Fitness: {self.g_best_fitness}")

    def plot_best_solution(self):
        depot_coord = self.vrptw.depot['coord']

        # Set up the plot
        plt.figure(figsize=(10, 8))

        # Plot depot
        plt.scatter(*depot_coord, color='red', s=200, label='Depot', zorder=5)
        plt.text(depot_coord[0], depot_coord[1], "Depot", fontsize=12, ha='right')

        # Define a color map with enough colors for up to 15 routes
        num_routes = min(15, len(self.g_best_position))  # Limit to 15 colors if needed
        colormap = cm.get_cmap('tab20', num_routes)  # Use 'tab20' for 20 distinct colors

        # Plot each route with a unique color up to 10 routes
        for i, route in enumerate(self.g_best_position):
            route_coords = [self.vrptw.depot['coord']]  # Start at the depot
            for customer_id in route:
                customer = next(c for c in self.vrptw.customers if c['id'] == customer_id)
                route_coords.append(customer['coord'])
            route_coords.append(self.vrptw.depot['coord'])  # Return to depot

            # Unpack coordinates for plotting
            xs, ys = zip(*route_coords)

            # Plot the route with a unique color
            plt.plot(xs, ys, marker='o', color=colormap(i), label=f'Vehicle {i + 1}')

        # Add labels and legend
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Best Routes in VRPTW Solution')
        plt.legend()
        plt.grid(True)
        plt.show()

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

# Main execution
customers, depot = load_data('../1.prob3_data/data.csv')
vrptw_instance = VRPTW(customers, depot, capacity=200)

pso = PSO(vrptw_instance, num_particles=50, num_iterations=500)
pso.optimize()
