import os
import numpy as np
import pandas as pd

# Get the path of the script (the directory where this file is located)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths for input and output files
returns_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')
cov_matrix_file = os.path.join(script_dir, '../3.prob2_output/3.2beregn_kovarians_matrix.csv')
results_file = os.path.join(script_dir, '../3.prob2_output/3.5bes.csv')

# Load the monthly returns data into a pandas DataFrame
returns_df = pd.read_csv(returns_file, index_col='Date', parse_dates=True)

# Compute the expected (mean) returns for each asset
expected_returns = returns_df.mean().values
num_assets = len(expected_returns)

# Load the covariance matrix from a CSV file
cov_matrix = pd.read_csv(cov_matrix_file, index_col=0).values

# Define risk-free rate (annual 2% rate, converted to monthly)
risk_free_rate = 0.02 / 12

# Function to calculate portfolio performance (return and risk)
def portfolio_performance(weights, expected_returns, cov_matrix):
    expected_return = np.dot(weights, expected_returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_stddev = np.sqrt(portfolio_variance)
    return expected_return, portfolio_stddev

# Fitness function: Calculate Sharpe ratio
def fitness_function(weights, expected_returns, cov_matrix, risk_free_rate):
    expected_return, portfolio_stddev = portfolio_performance(weights, expected_returns, cov_matrix)
    if portfolio_stddev == 0:
        return -np.inf
    sharpe_ratio = (expected_return - risk_free_rate) / portfolio_stddev
    return sharpe_ratio

# Generate an individual portfolio with weights
def generate_portfolio(num_assets):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    return {'weights': weights}

# Generate a population of portfolios
def generate_population(pop_size, num_assets):
    return [generate_portfolio(num_assets) for _ in range(pop_size)]


# Mutation function with mutation rate
def mutate_portfolio(individual, mutation_rate):
    weights = individual['weights']                                     # Extract current portfolio weights (individual chromosome)
    mutation = np.random.normal(0, mutation_rate, len(weights))         # Apply Gaussian noise to weights for mutation
    weights_prime = weights + mutation                                  # Mutate the weights by adding the noise
    weights_prime = np.clip(weights_prime, 0, 1)                        # Ensure weights stay between 0 and 1 (valid portfolio weights)

    if np.sum(weights_prime) == 0:                                      # If mutation causes weights to sum to zero, regenerate a random portfolio
        weights_prime = generate_portfolio(len(weights))['weights']
    else:
        weights_prime /= np.sum(weights_prime)                          # Normalize the mutated weights to sum to 1
    return {'weights': weights_prime}                                   # Return the mutated portfolio


# Main evolutionary strategy function
def evolutionary_strategy(expected_returns, cov_matrix, population_size, num_generations, risk_free_rate,
                          mutation_rate):
    num_assets = len(expected_returns)                                  # Determine number of assets (genes) in the portfolio
    population = generate_population(population_size, num_assets)       # Generate initial population of portfolios

    # Track the best Sharpe ratios across generations
    sharpe_ratios_per_generation = []

    for generation in range(num_generations):                           # Iterate over generations
        # Compute fitness (Sharpe ratio) for each portfolio in the population
        fitness_scores = np.array(
            [fitness_function(ind['weights'], expected_returns, cov_matrix, risk_free_rate) for ind in population])

        # Get the best Sharpe ratio in the current generation
        best_sharpe_ratio = np.max(fitness_scores)
        sharpe_ratios_per_generation.append(best_sharpe_ratio)          # Store the best Sharpe ratio for this generation

        # Sort population by fitness scores in descending order
        sorted_indices = np.argsort(fitness_scores)[::-1]

        # Create new population through mutation
        new_population = []
        for i in range(population_size):                                # Iterate through each individual in the population
            parent = population[i]                                      # Select parent (individual portfolio)
            offspring = mutate_portfolio(parent, mutation_rate)         # Mutate the parent to create an offspring

            # Compute fitness (Sharpe ratio) of the offspring
            offspring_fitness = fitness_function(offspring['weights'], expected_returns, cov_matrix, risk_free_rate)

            # Replace parent with offspring if the offspring has a better fitness score
            if offspring_fitness > fitness_scores[i]:
                new_population.append(offspring)
            else:
                new_population.append(parent)                           # Keep the parent if offspring is not better

        population = new_population                                     # Update population for the next generation

    # After all generations, find the best portfolio in the final population
    final_fitness_scores = np.array(
        [fitness_function(ind['weights'], expected_returns, cov_matrix, risk_free_rate) for ind in population])
    best_idx = np.argmax(final_fitness_scores)                          # Find the index of the best portfolio
    best_individual = population[best_idx]                              # Get the best portfolio

    return best_individual['weights'], final_fitness_scores[
        best_idx], sharpe_ratios_per_generation                         # Return best portfolio


# Main function to run the evolutionary strategy with various parameters
def run_evolutionary_strategy():
    # Define different parameter combinations for population size, generation count, and mutation rate
    population_sizes = [150, 200, 300]                                  # Different population sizes to test
    generation_counts = [200, 500, 1000]                                # Different number of generations to test
    mutation_rates = [0.15, 0.20, 0.6]                                  # Different mutation rates to test

    total_combinations = len(population_sizes) * len(generation_counts) * len(
        mutation_rates)  # Total number of parameter combinations
    print(f"Total combinations to test: {total_combinations}")

    combination_counter = 1                                             # Initialize combination counter
    best_sharpe = -np.inf                                               # Track the best Sharpe ratio found across all combinations
    best_combination = None                                             # Track the parameters of the best combination
    best_portfolio_overall = None                                       # Track the best portfolio (weights) found
    best_combination_number = None                                      # Track the combination number of the best result
    results = []                                                        # Store results for each generation

    # Iterate over all combinations of population sizes, generations, and mutation rates
    for pop_size in population_sizes:
        for gen_count in generation_counts:
            for mutation_rate in mutation_rates:
                print(
                    f"Running combination {combination_counter}/{total_combinations}: population_size={pop_size}, generations={gen_count}, mutation_rate={mutation_rate}")

                # Run the evolutionary strategy with current parameter combination
                best_portfolio, sharpe_ratio, sharpe_ratios_per_generation = evolutionary_strategy(
                    expected_returns, cov_matrix, pop_size, gen_count, risk_free_rate, mutation_rate)

                print(f"Sharpe ratio for combination {combination_counter}/{total_combinations}: {sharpe_ratio}")

                # Save Sharpe ratios for each generation along with the parameters used
                for generation, generation_sharpe_ratio in enumerate(sharpe_ratios_per_generation):
                    results.append({
                        'combination_number': combination_counter,      # Combination number for identification
                        'population_size': pop_size,                    # Population size used
                        'generations': gen_count,                       # Number of generations
                        'mutation_rate': mutation_rate,                 # Mutation rate used
                        'generation': generation,                       # Current generation number
                        'sharpe_ratio': generation_sharpe_ratio         # Sharpe ratio of the current generation
                    })

                # If the current Sharpe ratio is better than the previous best, update the best combination
                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio                          # Update the best Sharpe ratio found
                    best_combination = (pop_size, gen_count, mutation_rate)  # Store the best parameters
                    best_portfolio_overall = best_portfolio             # Store the best portfolio found
                    best_combination_number = combination_counter       # Store the best combination number

                combination_counter += 1  # Increment combination counter

    # Convert results to a DataFrame and save to CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)                        # Save results to CSV

    # Print the best combination found
    print("\nBest combination found:")
    print(f"Combination number: {best_combination_number}/{total_combinations}")
    print(f"Sharpe ratio: {best_sharpe}")
    print(f"Best portfolio weights:\n{best_portfolio_overall}")

    # Save the best portfolio to a CSV file
    best_portfolio_df = pd.DataFrame([best_portfolio_overall], columns=returns_df.columns)
    best_portfolio_df.to_csv(os.path.join(script_dir, '../3.prob2_output/3.5bes_best_portfolio.csv'), index=False)
    print(f"Best portfolio saved in '3.5bes_best_portfolio.csv'")


# Run the evolutionary strategy algorithm if the script is executed directly
if __name__ == "__main__":
    run_evolutionary_strategy()

