import os           # Library for file and directory operations
import numpy as np  # Numerical library for mathematical operations and arrays
import pandas as pd # Data analysis library for handling tabular data

# Optionally, set a random seed for reproducibility of results
np.random.seed(42)

# Find the path to this script (the directory where the file is running from)
# This allows for constructing relative file paths for input and output files.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths for input and output files based on the script's location
# Input files: monthly returns and covariance matrix
# Output file: results from the algorithm
returns_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')  # File with monthly returns
cov_matrix_file = os.path.join(script_dir, '../3.prob2_output/3.2beregn_kovarians_matrix.csv')  # File with covariance matrix
results_file = os.path.join(script_dir, '../3.prob2_output/3.4aep.csv')  # File to save the results

# Load data from the CSV file with monthly returns into a pandas DataFrame
# Use the 'Date' column as the index and convert it to datetime objects for time series handling.
returns_df = pd.read_csv(returns_file, index_col='Date', parse_dates=True)

# Calculate the expected (average) return for each stock in the portfolio
expected_returns = returns_df.mean().values  # Get average monthly returns as a numpy array
num_assets = len(expected_returns)  # Number of stocks in the portfolio

# Load the covariance matrix from the CSV file and convert it to a NumPy array
cov_matrix = pd.read_csv(cov_matrix_file, index_col=0).values

# Define the risk-free rate (e.g., 2% annually)
# Convert the annual risk-free rate to monthly by dividing by 12.
risk_free_rate = 0.02 / 12  # Monthly risk-free rate

# Function to calculate the portfolio's expected return and risk (standard deviation)
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
    
    # Avoid division by zero if the standard deviation is zero
    if portfolio_stddev == 0:
        return -np.inf  # Return negative infinity to indicate poor fitness
    
    # Calculate the Sharpe ratio: (expected return - risk-free rate) / risk
    sharpe_ratio = (expected_return - risk_free_rate) / portfolio_stddev
    
    # Return the Sharpe ratio as the fitness value
    return sharpe_ratio

# Modify the individual representation to include mutation sizes (sigma)
def generate_portfolio(num_assets):
    # Generate an array of random weights
    weights = np.random.random(num_assets)
    
    # Normalize the weights so that they sum to 1 (fully invested portfolio)
    weights /= np.sum(weights)
    
    # Initialize mutation sizes (sigma) for each weight with values between 0.05 and 0.2
    sigma = np.random.uniform(0.05, 0.2, num_assets)  # Initial mutation sizes
    
    # Return the individual as a dictionary with 'weights' and 'sigma'
    return {'weights': weights, 'sigma': sigma}

# Generate a population of portfolios
def generate_population(pop_size, num_assets):
    # Create a list of 'pop_size' number of portfolios
    return [generate_portfolio(num_assets) for _ in range(pop_size)]

# Self-adaptive mutation function
def mutate_portfolio(individual):
    weights = individual['weights']
    sigma = individual['sigma']
    num_assets = len(weights)
    
    # Mutation parameters for self-adaptive mutation
    tau = 1 / np.sqrt(2 * np.sqrt(num_assets))  # Global learning rate
    tau_prime = 1 / np.sqrt(2 * num_assets)     # Individual learning rate
    
    # Update sigma (mutation sizes)
    sigma_prime = sigma * np.exp(
        tau_prime * np.random.normal() +       # Global component
        tau * np.random.normal(size=num_assets)  # Individual component
    )
    
    # Ensure sigma stays within bounds
    sigma_prime = np.clip(sigma_prime, 1e-6, 1)  # Avoid zero or negative values
    
    # Mutate weights
    weights_prime = weights + sigma_prime * np.random.normal(size=num_assets)
    
    # Clip the weights to be between 0 and 1 (no negative weights)
    weights_prime = np.clip(weights_prime, 0, 1)
    
    # Avoid all weights being zero by reinitializing if necessary
    if np.sum(weights_prime) == 0:
        weights_prime = generate_portfolio(num_assets)['weights']  # Reinitialize
    else:
        # Normalize the weights so that they sum to 1
        weights_prime /= np.sum(weights_prime)
    
    # Return the mutated individual with updated weights and sigma
    return {'weights': weights_prime, 'sigma': sigma_prime}

# Tournament selection function
def tournament_selection(population, fitness_scores, tournament_size):
    selected = []
    pop_size = len(population)
    
    # For each position in the population
    for _ in range(pop_size):
        # Select individuals for the tournament randomly
        participants = np.random.choice(pop_size, tournament_size, replace=False)
        
        # Get the fitness scores for the participants
        participant_fitness = fitness_scores[participants]
        
        # Find the index of the participant with the highest fitness
        winner_idx = participants[np.argmax(participant_fitness)]
        
        # Add the winner to the list of selected individuals
        selected.append(population[winner_idx])
    
    # Return the list of selected parents
    return selected

# Elitism function
def get_elites(population, fitness_scores, num_elites):
    # Sort the indices based on fitness scores in ascending order
    elite_indices = np.argsort(fitness_scores)[-num_elites:]  # Select the best individuals
    
    # Get the best individuals from the population
    elites = [population[i] for i in elite_indices]
    
    # Return the list of elite individuals
    return elites

# Advanced Evolutionary Programming algorithm
def advanced_evolutionary_programming(
        expected_returns, cov_matrix, population_size, num_generations,
        risk_free_rate, tournament_size=3, num_elites=1
):
    num_assets = len(expected_returns)
    population = generate_population(population_size, num_assets)
    best_sharpe_per_generation = []  # List to store the best Sharpe ratio in each generation

    for generation in range(num_generations):
        # Evaluate fitness for each portfolio
        fitness_scores = np.array([
            fitness_function(ind['weights'], expected_returns, cov_matrix, risk_free_rate)
            for ind in population
        ])

        # Find and store the best Sharpe ratio for this generation
        best_sharpe_in_gen = np.max(fitness_scores)
        best_sharpe_per_generation.append(best_sharpe_in_gen)

        # Elitism: Get the elite individuals to carry over to the next generation
        elites = get_elites(population, fitness_scores, num_elites)

        # Selection and mutation to form offspring
        selected_parents = tournament_selection(population, fitness_scores, tournament_size)
        offspring = [mutate_portfolio(parent) for parent in selected_parents]

        # Form the new population with elites and offspring
        population = elites + offspring[:population_size - num_elites]

    # Final evaluation after the last generation
    final_fitness_scores = np.array([
        fitness_function(ind['weights'], expected_returns, cov_matrix, risk_free_rate)
        for ind in population
    ])

    # Find the best individual from the final generation
    best_idx = np.argmax(final_fitness_scores)
    best_individual = population[best_idx]

    # Return the best portfolio's weights, Sharpe ratio, and the list of best Sharpe ratios per generation
    return best_individual['weights'], final_fitness_scores[best_idx], best_sharpe_per_generation


# Main function to run the advanced EP algorithm
def run_advanced_ep():
    # Parameter ranges for testing
    population_sizes = [20, 50, 100]  # Population sizes to test
    generation_counts = [50, 100, 200]  # Number of generations to test
    tournament_sizes = [1]  # Tournament sizes to test
    num_elites_list = [1]  # Number of elites to test
    initial_mutation_rates = [0.01, 0.05, 0.1]  # Initial mutation rates for testing
    alphas = [0.1, 0.3, 0.5]  # Values for alpha to test

    # Calculate the total number of combinations
    total_combinations = (
            len(population_sizes) *
            len(generation_counts) *
            len(tournament_sizes) *
            len(num_elites_list) *
            len(initial_mutation_rates) *
            len(alphas)
    )
    print(f"Total number of combinations to test: {total_combinations}")

    combination_counter = 1  # Counter to keep track of the current combination
    best_sharpe = -np.inf  # Variable to track the best Sharpe ratio
    best_combination = None  # Variable to store the best combination of parameters
    best_portfolio_overall = None  # Variable to store the best portfolio
    best_combination_number = None  # Variable to track the best combination number
    results = []  # List to collect all results

    # Test all combinations
    for pop_size in population_sizes:
        for gen_count in generation_counts:
            for tour_size in tournament_sizes:
                for num_elites in num_elites_list:
                    for init_mut_rate in initial_mutation_rates:
                        for alpha in alphas:
                            print(f"Running combination {combination_counter}/{total_combinations}: "
                                  f"population size={pop_size}, generations={gen_count}, "
                                  f"tournament size={tour_size}, number of elites={num_elites}, "
                                  f"initial mutation rate={init_mut_rate}, alpha={alpha}")

                            # Run the algorithm with the current parameter combination
                            best_portfolio, final_sharpe, sharpe_ratios_per_generation = advanced_evolutionary_programming(
                                expected_returns, cov_matrix, pop_size, gen_count, risk_free_rate,
                                tournament_size=tour_size, num_elites=num_elites
                            )

                            # Store each generation's best Sharpe ratio with additional details
                            for generation_num, gen_sharpe_ratio in enumerate(sharpe_ratios_per_generation, start=1):
                                results.append({
                                    'generation': generation_num,
                                    'combination_number': combination_counter,
                                    'pop_size': pop_size,
                                    'gen_count': gen_count,
                                    'initial_mutation_rate': init_mut_rate,
                                    'alpha': alpha,
                                    'sharpe_ratio': gen_sharpe_ratio,
                                })

                            # Update the best combination if this one has the best final Sharpe ratio
                            if final_sharpe > best_sharpe:
                                best_sharpe = final_sharpe
                                best_combination = (pop_size, gen_count, tour_size, num_elites, init_mut_rate, alpha)
                                best_portfolio_overall = best_portfolio
                                best_combination_number = combination_counter

                            # Update the combination number
                            combination_counter += 1

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    results_df.to_csv(results_file, index=False)

    # Print the best combination found
    print("\nBest combination found:")
    print(f"Combination number: {best_combination_number}/{total_combinations}")
    print(f"Sharpe ratio: {best_sharpe}")
    print(f"Population size: {best_combination[0]}, Generations: {best_combination[1]}, "
          f"Tournament size: {best_combination[2]}, Number of elites: {best_combination[3]}, "
          f"Initial mutation rate: {best_combination[4]}, Alpha: {best_combination[5]}")
    print(f"Best portfolio weights:\n{best_portfolio_overall}")

    # Save the best portfolio to a CSV file
    best_portfolio_df = pd.DataFrame([best_portfolio_overall], columns=returns_df.columns)
    best_portfolio_df.to_csv(os.path.join(script_dir, '../3.prob2_output/3.4aep_best_portfolio.csv'), index=False)
    print(f"Best portfolio saved to '3.4aep_best_portfolio.csv'")


# Run the advanced EP algorithm
if __name__ == "__main__":
    run_advanced_ep()

