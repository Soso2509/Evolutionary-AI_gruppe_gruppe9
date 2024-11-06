import os
import numpy as np
import pandas as pd
import math

def load_data(file_name):
    """
    Load data from CSV file and return required components for VRPTW.

    Parameters:
    - file_name (str): Path to the CSV file.

    Returns:
    - demands (list): List of demands for each customer.
    - time_windows (list of tuples): Time windows (start, end) for each customer.
    - service_times (list): List of service times for each customer.
    - travel_time_matrix (2D list): Matrix of travel times between customers.
    """
    data = pd.read_csv(file_name)
    demands = []
    time_windows = []
    service_times = []
    coordinates = []

    for _, row in data.iterrows():
        demands.append(row['DEMAND'])
        time_windows.append((row['READY TIME'], row['DUE DATE']))
        service_times.append(row['SERVICE TIME'])
        coordinates.append((row['XCOORD.'], row['YCOORD.']))

    # Build travel time matrix using Euclidean distances
    num_customers = len(coordinates)
    travel_time_matrix = [[0] * num_customers for _ in range(num_customers)]
    for i in range(num_customers):
        for j in range(num_customers):
            travel_time_matrix[i][j] = np.sqrt(
                (coordinates[i][0] - coordinates[j][0]) ** 2 + (coordinates[i][1] - coordinates[j][1]) ** 2
            )

    return demands, time_windows, service_times, travel_time_matrix

def is_route_feasible(file_name, route, vehicle_capacity):
    """
    Check if a given route is feasible under VRPTW constraints.

    Parameters:
    - file_name (str): Path to the CSV file containing customer data.
    - route (list): List of customer indices representing the route (including depot as first and last node).
    - vehicle_capacity (int): Maximum capacity of the vehicle.

    Returns:
    - bool: True if the route is feasible, False otherwise.
    """
    # Load data
    demands, time_windows, service_times, travel_time_matrix = load_data(file_name)

    # Check if route starts and ends at the depot
    if route[0] != 0 or route[-1] != 0:
        print("Route does not start and end at the depot.")
        return False

    # Initialize variables
    current_time = 0
    current_load = 0

    # Check each leg of the route
    for i in range(len(route) - 1):
        current_customer = route[i]
        next_customer = route[i + 1]

        # Update load and check capacity
        current_load += demands[current_customer]
        if current_load > vehicle_capacity:
            print(f"Capacity exceeded at customer {current_customer}. Load: {current_load}, Capacity: {vehicle_capacity}")
            return False

        # Travel to the next customer
        travel_time = travel_time_matrix[current_customer][next_customer]
        current_time += travel_time

        # Check if arrival time is within the time window of the next customer
        next_time_window = time_windows[next_customer]
        if current_time > next_time_window[1]:  # If we're too late
            print(f"Arrived too late at customer {next_customer}. Arrival time: {current_time}, Time window: {next_time_window}")
            return False

        # If we're early, wait until the start of the time window
        if current_time < next_time_window[0]:
            current_time = next_time_window[0]

        # Add service time for the current customer
        current_time += service_times[next_customer]

    print("Route is feasible.")
    return True


# Example usage:
# Assume `data.csv` contains the VRPTW data and you want to test a route.
routes =[[0, 43, 42, 41, 40, 44, 45, 48, 51, 50, 52, 49, 47, 0], [0, 67, 65, 63, 62, 74, 72, 61, 64, 68, 66, 69, 0], [0, 5, 3, 7, 8, 9, 6, 4, 2, 1, 75, 0], [0, 20, 21, 0], [0, 90, 87, 86, 83, 82, 84, 85, 88, 89, 91, 0], [0, 10, 11, 14, 12, 100, 0], [0, 29, 30, 28, 26, 23, 22, 0], [0, 13, 17, 18, 19, 15, 16, 98, 97, 0], [0, 81, 78, 76, 71, 70, 73, 77, 79, 80, 0], [0, 57, 55, 54, 53, 56, 58, 60, 59, 0], [0, 24, 25, 27, 37, 38, 39, 36, 34, 0], [0, 32, 33, 31, 35, 46, 0], [0, 95, 96, 92, 93, 0], [0, 99, 94, 0]]


script_dir = os.path.dirname(os.path.abspath(__file__))

for i in routes:
    is_route_feasible((os.path.join(script_dir, '../1.prob3_data/data.csv')), i, vehicle_capacity=200)



#[[0, 67, 65, 63, 62, 74, 72, 61, 64, 68, 66, 69, 0], [0, 43, 42, 41, 40, 44, 46, 45, 48, 51, 50, 52, 49, 47, 0], [0, 13, 17, 18, 19, 15, 16, 14, 12, 100, 0], [0, 24, 25, 27, 29, 30, 28, 26, 23, 22, 21, 75, 0], [0, 5, 3, 7, 8, 9, 6, 4, 2, 1, 91, 0], [0, 81, 78, 76, 71, 70, 73, 77, 79, 80, 0], [0, 32, 33, 31, 35, 37, 38, 39, 36, 34, 0], [0, 90, 87, 86, 83, 82, 84, 85, 88, 89, 0], [0, 20, 10, 11, 98, 97, 60, 59, 0], [0, 99, 95, 96, 94, 92, 93, 0], [0, 57, 55, 54, 53, 56, 58, 0]]
#[[0, np.int64(43), np.int64(42), np.int64(41), np.int64(40), np.int64(44), np.int64(45), np.int64(48), np.int64(51), np.int64(50), np.int64(52), np.int64(49), np.int64(47), 0], [0, np.int64(67), np.int64(65), np.int64(63), np.int64(74), np.int64(72), np.int64(61), np.int64(64), np.int64(68), np.int64(66), np.int64(69), np.int64(21), 0], [0, np.int64(57), np.int64(55), np.int64(54), np.int64(53), np.int64(56), np.int64(58), np.int64(60), np.int64(59), 0], [0, np.int64(32), np.int64(33), np.int64(31), np.int64(35), np.int64(37), np.int64(38), np.int64(39), np.int64(36), np.int64(34), 0], [0, np.int64(5), np.int64(3), np.int64(7), np.int64(8), np.int64(9), np.int64(6), np.int64(4), np.int64(2), np.int64(1), np.int64(75), 0], [0, np.int64(20), np.int64(24), np.int64(25), np.int64(27), np.int64(29), np.int64(30), np.int64(28), np.int64(26), np.int64(23), np.int64(22), 0], [0, np.int64(13), np.int64(17), np.int64(18), np.int64(19), np.int64(15), 16, 14, 12, 100, 0], [0, 62, 46, 71, 70, 73, 77, 79, 80, 91, 0], [0, 90, 87, 86, 83, 82, 84, 85, 88, 89, 0], [0, 10, 11, 98, 97, 9, 0], [0, np.int64(99), np.int64(96), np.int64(95), np.int64(94), np.int64(92), 0], [0, np.int64(81), np.int64(78), np.int64(76), 0]]
