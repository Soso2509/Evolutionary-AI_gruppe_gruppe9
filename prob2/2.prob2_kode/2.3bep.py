import os  # Library for file and directory operations
import numpy as np  # Numerical library for mathematical operations and arrays
import pandas as pd  # Data analysis library for handling tabular data

# Define file paths for input and output files based on the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
returns_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')  # Monthly returns file
cov_matrix_file = os.path.join(script_dir, '../3.prob2_output/3.2beregn_kovarians_matrix.csv')  # Covariance matrix file
results_file = os.path.join(script_dir, '../3.prob2_output/3.3bep.csv')  # File to save results

# Load monthly returns into a pandas DataFrame
returns_df = pd.read_csv(returns_file, index_col='Date', parse_dates=True)

# Calculate expected (average) return for each stock in the portfolio
expected_returns = returns_df.mean().values  # Get average monthly return for each stock
num_assets = len(expected_returns)  # Number of stocks in the portfolio

# Load the covariance matrix and convert it to a NumPy array
cov_matrix = pd.read_csv(cov_matrix_file, index_col=0).values

# Define the risk-free rate (e.g., 2% annually), converted to monthly
risk_free_rate = 0.02 / 12  # Monthly risk-free rate


# Function to calculate portfolio's expected return and risk (standard deviation)
def portfolio_performance(weights, expected_returns, cov_matrix):
    expected_return = np.dot(weights, expected_returns)  # Expected return
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))  # Portfolio variance
    portfolio_stddev = np.sqrt(portfolio_variance)  # Standard deviation (risk)
    return expected_return, portfolio_stddev


# Fitness function: Calculate Sharpe ratio for a given portfolio
def fitness_function(weights, expected_returns, cov_matrix, risk_free_rate):
    expected_return, portfolio_stddev = portfolio_performance(weights, expected_returns, cov_matrix)
    sharpe_ratio = (expected_return - risk_free_rate) / portfolio_stddev  # Sharpe ratio
    return sharpe_ratio


# Function to generate a random portfolio with weights that sum to 1
def generate_portfolio(num_assets):
    weights = np.random.random(num_assets)  # Generate random weights
    weights /= np.sum(weights)  # Normalize weights to sum to 1
    return weights


# Function to generate a population of random portfolios
def generate_population(pop_size, num_assets):
    population = [generate_portfolio(num_assets) for _ in range(pop_size)]  # Generate population
    return np.array(population)


# Function to mutate a portfolio to introduce variation
def mutate_portfolio(portfolio, mutation_rate=0.1):
    mutation = np.random.normal(0, mutation_rate, len(portfolio))  # Generate mutation
    mutated_portfolio = portfolio + mutation  # Add the mutation
    mutated_portfolio = np.clip(mutated_portfolio, 0, 1)  # Clip weights to between 0 and 1
    mutated_portfolio /= np.sum(mutated_portfolio)  # Normalize weights again
    return mutated_portfolio


# Function to select the best portfolios based on their fitness (Sharpe ratio)
def select_best(population, fitness_scores, num_to_select):
    selected_indices = np.argsort(fitness_scores)[-num_to_select:]  # Select the best portfolios
    return population[selected_indices]


# Evolutionary Programming Algorithm for portfolio optimization
def evolutionary_programming(expected_returns, cov_matrix, population_size, num_generations, risk_free_rate,
                             mutation_rate=0.1, num_to_select=1):
    num_assets = len(expected_returns)  # Number of stocks in the portfolio
    population = generate_population(population_size, num_assets)  # Generate initial population

    generation_results = []  # List to store results for each generation

    # Start the evolution process
    for generation in range(num_generations):
        fitness_scores = np.array(
            [fitness_function(p, expected_returns, cov_matrix, risk_free_rate) for p in population])

        # Find the best portfolio in the current generation
        best_index = np.argmax(fitness_scores)
        best_sharpe_ratio = fitness_scores[best_index]
        best_portfolio = population[best_index]

        # Store result for this generation
        generation_results.append({
            'generation': generation + 1,  # Generation number
            'combination_number': combination_counter,
            'pop_size': population_size,
            'gen_count': num_generations,
            'mut_rate': mutation_rate,
            'num_to_select': num_to_select,
            'sharpe_ratio': best_sharpe_ratio
        })

        # Select the best portfolios based on fitness (Sharpe ratio)
        best_portfolios = select_best(population, fitness_scores, num_to_select)  # Select top portfolios

        # Create the next generation
        next_generation = []
        for portfolio in best_portfolios:
            next_generation.append(portfolio)  # Keep the original portfolio
            next_generation.append(mutate_portfolio(portfolio, mutation_rate))  # Add a mutated version
        population = np.array(next_generation)  # Update population

    # Find the best portfolio in the last population
    final_fitness_scores = np.array(
        [fitness_function(p, expected_returns, cov_matrix, risk_free_rate) for p in population])
    best_portfolio = population[np.argmax(final_fitness_scores)]
    best_sharpe_ratio = np.max(final_fitness_scores)

    return best_portfolio, best_sharpe_ratio, generation_results


# Define parameter ranges for testing the algorithm
population_sizes = [150, 200, 300]  # Population sizes
generation_counts = [200, 500, 1000]  # Number of generations
mutation_rates = [0.15, 0.20]  # Mutation rates
num_to_select_list = [5, 10]  # Different values for num_to_select as a tunable hyperparameter

# Calculate the total number of combinations to test
total_combinations = len(population_sizes) * len(generation_counts) * len(mutation_rates) * len(num_to_select_list)
print(f"Total combinations to test: {total_combinations}")

# Initialize variables to track the best combination and Sharpe ratio
best_sharpe = -np.inf  # Start with negative infinity to ensure any positive Sharpe ratio is better
best_combination = None  # Track the parameters for the best combination
best_combination_number = None  # Track the combination number for the best result

# Initialize a list to collect all results from the tests
results = []

# Start testing over all parameter combinations
combination_counter = 1  # Counter to keep track of the combination number

# Loop through all combinations of population sizes, generation counts, mutation rates, and num_to_select values
for pop_size in population_sizes:
    for gen_count in generation_counts:
        for mut_rate in mutation_rates:
            for num_to_select in num_to_select_list:
                print(
                    f"Running combination {combination_counter}/{total_combinations}: pop_size={pop_size}, gen_count={gen_count}, mut_rate={mut_rate}, num_to_select={num_to_select}")

                # Run the evolutionary algorithm with the current parameters
                best_portfolio, sharpe_ratio, generation_results = evolutionary_programming(expected_returns,
                                                                                            cov_matrix, pop_size,
                                                                                            gen_count, risk_free_rate,
                                                                                            mut_rate, num_to_select)

                # Save the results for each generation
                results.extend(generation_results)

                # Update the best Sharpe ratio and combination if necessary
                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_combination = (pop_size, gen_count, mut_rate, num_to_select)
                    best_combination_number = combination_counter

                combination_counter += 1

# Convert results to a pandas DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file in the desired format
results_df.to_csv(results_file, index=False)

# Print the best combination found
print("\nBest combination found")
print(f"Combination number: {best_combination_number}/{total_combinations}")
print(f"Sharpe ratio: {best_sharpe}")
print(
    f"Population size: {best_combination[0]}, Number of generations: {best_combination[1]}, Mutation rate: {best_combination[2]}, Num to select: {best_combination[3]}")
print(f"Best portfolio weights: {best_portfolio}")

# Save the best portfolio to a CSV file
best_portfolio_df = pd.DataFrame([best_portfolio], columns=returns_df.columns)
best_portfolio_df.to_csv(os.path.join(script_dir, '../3.prob2_output/3.3bep_best_portfolio.csv'), index=False)

print(f"Best portfolio saved to '3.3bep_best_portfolio.csv'")
