import os          # Library for file and directory operations
import numpy as np  # Numerical library for mathematical operations and arrays
import pandas as pd # Data analysis library for handling tabular data

# Find the absolute path to the directory where this script is located
# This is useful to construct relative file paths based on the script's location.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths for input and output files based on the script's location
# Input files: monthly returns and covariance matrix
# Output file: results from the evolutionary algorithm
returns_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')  # File with calculated monthly returns
cov_matrix_file = os.path.join(script_dir, '../3.prob2_output/3.2beregn_kovarians_matrix.csv')  # File with the covariance matrix
results_file = os.path.join(script_dir, '../3.prob2_output/3.3bep.csv')  # File to store results

# Read the monthly returns from the CSV file into a pandas DataFrame
# The 'Date' column is used as the index, and dates are converted to datetime objects.
returns_df = pd.read_csv(returns_file, index_col='Date', parse_dates=True)

# Calculate the expected (average) return for each stock in the portfolio
# The mean monthly return is calculated for each stock.
expected_returns = returns_df.mean().values  # Get average monthly return for each stock
num_assets = len(expected_returns)  # Number of stocks in the portfolio

# Read the covariance matrix from the CSV file and convert it into a NumPy array
# The first column is used as the index.
cov_matrix = pd.read_csv(cov_matrix_file, index_col=0).values

# Define the risk-free rate (e.g., 2% annually)
# Convert the annual risk-free rate to monthly by dividing by 12.
risk_free_rate = 0.02 / 12  # Monthly risk-free rate

# Function to calculate the expected return and risk (standard deviation) of a portfolio
def portfolio_performance(weights, expected_returns, cov_matrix):
    # Calculate the portfolio's expected return as the weighted average of the returns
    expected_return = np.dot(weights, expected_returns)
    
    # Calculate the portfolio's variance (risk) using the covariance matrix and portfolio weights
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Calculate the standard deviation (square root of the variance)
    portfolio_stddev = np.sqrt(portfolio_variance)
    
    # Return the expected return and risk
    return expected_return, portfolio_stddev

# Fitness function: Calculate the Sharpe ratio for a given portfolio
def fitness_function(weights, expected_returns, cov_matrix, risk_free_rate):
    # Calculate the portfolio's expected return and risk
    expected_return, portfolio_stddev = portfolio_performance(weights, expected_returns, cov_matrix)
    
    # Calculate the Sharpe ratio: (expected return - risk-free rate) / risk
    sharpe_ratio = (expected_return - risk_free_rate) / portfolio_stddev
    
    # Return the Sharpe ratio as the fitness value
    return sharpe_ratio

# Function to generate a random portfolio with weights summing to 1
def generate_portfolio(num_assets):
    # Generate an array of random weights
    weights = np.random.random(num_assets)
    
    # Normalize the weights so that they sum to 1 (fully invested portfolio)
    weights /= np.sum(weights)
    
    # Return the weights for the portfolio
    return weights

# Function to generate a population of random portfolios
def generate_population(pop_size, num_assets):
    population = []
    
    # Generate 'pop_size' number of portfolios
    for _ in range(pop_size):
        population.append(generate_portfolio(num_assets))
    
    # Convert the population to a NumPy array and return it
    return np.array(population)

# Function to mutate a portfolio to introduce variation
def mutate_portfolio(portfolio, mutation_rate=0.1):
    # Generate a mutation by drawing from a normal distribution with a standard deviation of mutation_rate
    mutation = np.random.normal(0, mutation_rate, len(portfolio))
    
    # Add the mutation to the original portfolio
    mutated_portfolio = portfolio + mutation
    
    # Clip the weights to be between 0 and 1 (no negative weights)
    mutated_portfolio = np.clip(mutated_portfolio, 0, 1)
    
    # Normalize the weights so that they sum to 1
    mutated_portfolio /= np.sum(mutated_portfolio)
    
    # Return the mutated portfolio
    return mutated_portfolio

# Function to select the best portfolios based on their fitness (Sharpe ratio)
def select_best(population, fitness_scores, num_to_select):
    # Sort the indices of the portfolios based on fitness in ascending order
    selected_indices = np.argsort(fitness_scores)[-num_to_select:]  # Get the indices of the best portfolios
    
    # Return the best portfolios
    return population[selected_indices]

# Evolutionary Programming Algorithm for portfolio optimization
def evolutionary_programming(expected_returns, cov_matrix, population_size, num_generations, risk_free_rate, mutation_rate=0.1):
    num_assets = len(expected_returns)
    
    # Generate the initial population of portfolios
    population = generate_population(population_size, num_assets)
    
    # Start the evolution process for a number of generations
    for generation in range(num_generations):
        # Calculate the fitness (Sharpe ratio) for each portfolio in the population
        fitness_scores = np.array([fitness_function(p, expected_returns, cov_matrix, risk_free_rate) for p in population])
        
        # Select the best portfolios to carry over to the next generation
        num_to_select = population_size // 2  # Keep half of the population
        best_portfolios = select_best(population, fitness_scores, num_to_select)
        
        # Mutate the best portfolios to create variation
        next_generation = []
        for portfolio in best_portfolios:
            next_generation.append(portfolio)  # Keep the original
            next_generation.append(mutate_portfolio(portfolio, mutation_rate))  # Add the mutated version
        
        # Update the population with the next generation
        population = np.array(next_generation)
    
    # After all generations, find the best portfolio in the final population
    final_fitness_scores = np.array([fitness_function(p, expected_returns, cov_matrix, risk_free_rate) for p in population])
    best_portfolio = population[np.argmax(final_fitness_scores)]
    
    # Return the best portfolio and its Sharpe ratio
    return best_portfolio, np.max(final_fitness_scores)

# Define parameter ranges for testing the algorithm
population_sizes = [50, 100, 150, 200, 250, 300]  # Different population sizes
generation_counts = [50, 100, 150, 200, 250, 300]  # Different number of generations
mutation_rates = [0.01, 0.05, 0.1, 0.15, 0.20]  # Different mutation rates

# Calculate the total number of parameter combinations to test
total_combinations = len(population_sizes) * len(generation_counts) * len(mutation_rates)

# Print the total number of combinations to the console
print(f"Total combinations to test: {total_combinations}")

# Initialize variables to track the best combination and Sharpe ratio
best_sharpe = -np.inf  # Start with negative infinity to ensure any positive Sharpe ratio is better
best_combination = None  # Track the parameters for the best combination
best_combination_number = None  # Track the combination number for the best result

# Initialize a list to collect all results from the tests
results = []

# Start the test over all parameter combinations
combination_counter = 1  # Counter to keep track of the combination number

# Loop through all combinations of population sizes, generation counts, and mutation rates
for pop_size in population_sizes:
    for gen_count in generation_counts:
        for mut_rate in mutation_rates:
            # Print information about the current combination
            print(f"Running combination {combination_counter}/{total_combinations}: pop_size={pop_size}, gen_count={gen_count}, mut_rate={mut_rate}")
            
            # Run the evolutionary algorithm with the current parameters
            best_portfolio, sharpe_ratio = evolutionary_programming(expected_returns, cov_matrix, pop_size, gen_count, risk_free_rate, mut_rate)
            
            # Print the Sharpe ratio for this combination
            print(f"Sharpe ratio for combination {combination_counter}/{total_combinations}: {sharpe_ratio}")
            
            # Store the results in the results list, including the combination number
            results.append({
                'combination_number': combination_counter,  # Combination number
                'pop_size': pop_size,
                'gen_count': gen_count,
                'mut_rate': mut_rate,
                'sharpe_ratio': sharpe_ratio
            })
            
            # Check if this Sharpe ratio is better than the best one so far
            if sharpe_ratio > best_sharpe:
                # Update the best Sharpe ratio and parameters
                best_sharpe = sharpe_ratio
                best_combination = (pop_size, gen_count, mut_rate)
                best_combination_number = combination_counter  # Save the combination number
                
            # Update the combination number
            combination_counter += 1

# Convert the results list to a pandas DataFrame for easier storage and analysis
results_df = pd.DataFrame(results)

# Save the results to a CSV file without including the DataFrame index
results_df.to_csv(results_file, index=False)

# After all combinations are tested, print the result for the best combination
print("\nBest combination found")
print(f"Combination number: {best_combination_number}/{total_combinations}")
print(f"Sharpe ratio: {best_sharpe}")
print(f"Population size: {best_combination[0]}, Number of generations: {best_combination[1]}, Mutation rate: {best_combination[2]}")
print(f"Best portfolio weights: {best_portfolio}")

# Save the best portfolio (weights) to a CSV file for further analysis
best_portfolio_df = pd.DataFrame([best_portfolio], columns=returns_df.columns)  # Use stock names as column headers
best_portfolio_df.to_csv(os.path.join(script_dir, '../3.prob2_output/3.3bep_best_portfolio.csv'), index=False)

# Print a message indicating that the best portfolio has been saved
print(f"Best portfolio saved to '3.3bep_best_portfolio.csv'")
