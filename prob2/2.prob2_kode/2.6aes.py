import os
import numpy as np
import pandas as pd

# Get the path of the script (the directory where this file is located)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths for input and output files
returns_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')
cov_matrix_file = os.path.join(script_dir, '../3.prob2_output/3.2beregn_kovarians_matrix.csv')
results_file = os.path.join(script_dir, '../3.prob2_output/3.6aes.csv')

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
    # Calculate expected return as dot product of weights and expected returns
    expected_return = np.dot(weights, expected_returns)

    # Calculate portfolio variance as a weighted covariance
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

    # Calculate portfolio risk (standard deviation)
    portfolio_stddev = np.sqrt(portfolio_variance)

    return expected_return, portfolio_stddev


# Fitness function: Calculate Sharpe ratio
def fitness_function(weights, expected_returns, cov_matrix, risk_free_rate):
    # Calculate portfolio return and risk
    expected_return, portfolio_stddev = portfolio_performance(weights, expected_returns, cov_matrix)

    # If risk is zero, return a negative infinite Sharpe ratio (invalid portfolio)
    if portfolio_stddev == 0:
        return -np.inf

    # Calculate Sharpe ratio as (return - risk-free rate) / risk
    sharpe_ratio = (expected_return - risk_free_rate) / portfolio_stddev
    return sharpe_ratio


# Generate an individual portfolio with weights and random sigma (self-adaptive mutation rate)
def generate_portfolio(num_assets):
    # Random weights between 0 and 1, normalized to sum to 1
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)

    # Assign random sigma for self-adaptive mutation
    return {'weights': weights, 'sigma': np.random.uniform(0.05, 0.2, num_assets)}


# Generate a population of portfolios
def generate_population(pop_size, num_assets):
    # Create a list of portfolios
    return [generate_portfolio(num_assets) for _ in range(pop_size)]


# Adjust sigma (mutation rate) based on improvement
def adapt_sigma(sigma, improvement, tau=0.1):
    # Decrease sigma if improvement is positive, otherwise increase it
    if improvement > 0:
        return sigma * np.exp(-tau)  # Reduce sigma on improvement
    else:
        return sigma * np.exp(tau)  # Increase sigma on no improvement


# Mutate portfolio based on self-adaptive sigma and calculate improvement in fitness
def mutate_portfolio(individual, previous_fitness, expected_returns, cov_matrix, risk_free_rate):
    weights = individual['weights']  # Current weights
    sigma = individual['sigma']  # Current mutation rate (sigma)
    num_assets = len(weights)

    # Mutate weights using normal distribution scaled by sigma
    mutation = sigma * np.random.normal(size=num_assets)
    weights_prime = weights + mutation

    # Ensure weights are within valid bounds (0-1) and normalize
    weights_prime = np.clip(weights_prime, 0, 1)
    if np.sum(weights_prime) == 0:
        weights_prime = generate_portfolio(num_assets)['weights']
    else:
        weights_prime /= np.sum(weights_prime)

    # Calculate fitness after mutation (Sharpe ratio)
    current_fitness = fitness_function(weights_prime, expected_returns, cov_matrix, risk_free_rate)

    # Calculate improvement in fitness
    improvement = current_fitness - previous_fitness

    # Adapt sigma based on whether mutation improved fitness
    updated_sigma = adapt_sigma(sigma, improvement)

    # Return updated portfolio with new weights, sigma, and fitness
    return {'weights': weights_prime, 'sigma': updated_sigma, 'sharpe_ratio': current_fitness}


# Recombination (crossover) between two parent portfolios
def recombine_portfolios(parent1, parent2):
    # Select a crossover point (at least one gene from each parent)
    crossover_point = np.random.randint(1, len(parent1['weights']))

    # Create child weights by combining portions of both parents
    child_weights = np.concatenate((parent1['weights'][:crossover_point], parent2['weights'][crossover_point:]))
    child_weights /= np.sum(child_weights)  # Normalize child weights

    # Average the mutation rates (sigma) of both parents
    child_sigma = (parent1['sigma'] + parent2['sigma']) / 2
    return {'weights': child_weights, 'sigma': child_sigma}


# Evolutionary strategy algorithm
def evolutionary_strategy(expected_returns, cov_matrix, population_size, num_generations, risk_free_rate):
    num_assets = len(expected_returns)  # Number of assets
    population = generate_population(population_size, num_assets)  # Initial population

    results_per_generation = []  # Track results for each generation

    for generation in range(num_generations):
        # Calculate fitness (Sharpe ratio) for the entire population
        fitness_scores = np.array(
            [fitness_function(ind['weights'], expected_returns, cov_matrix, risk_free_rate) for ind in population])

        # Save results for this generation
        for idx, individual in enumerate(population):
            results_per_generation.append({
                'generation': generation,
                'combination_number': None,
                'population_size': population_size,  # Save the population size for each iteration
                'sharpe_ratio': fitness_scores[idx],  # Store Sharpe ratio
                'weights': individual['weights']
            })

        new_population = []  # Create the new population

        # Iterate through the current population to generate offspring
        for individual in population:
            previous_fitness = fitness_function(individual['weights'], expected_returns, cov_matrix, risk_free_rate)

            # Mutate individual and get new fitness
            mutated_offspring = mutate_portfolio(individual, previous_fitness, expected_returns, cov_matrix,
                                                 risk_free_rate)

            # 50% chance to recombine with another individual
            if np.random.rand() < 0.5:
                partner = np.random.choice(population)
                mutated_offspring = recombine_portfolios(mutated_offspring, partner)

                # Recalculate fitness for the recombined offspring
                recombined_fitness = fitness_function(mutated_offspring['weights'], expected_returns, cov_matrix,
                                                      risk_free_rate)
                mutated_offspring['sharpe_ratio'] = recombined_fitness  # Update fitness for recombined offspring

            # Replace individual if the offspring has a better fitness score
            if mutated_offspring['sharpe_ratio'] > previous_fitness:
                new_population.append(mutated_offspring)
            else:
                new_population.append(individual)  # Retain the parent if offspring is not better

        population = new_population  # Update population with new offspring

    # Find the best individual from the final population
    final_fitness_scores = np.array(
        [fitness_function(ind['weights'], expected_returns, cov_matrix, risk_free_rate) for ind in population])
    best_idx = np.argmax(final_fitness_scores)
    best_individual = population[best_idx]

    # Return the best portfolio's weights, fitness score, and results per generation
    return best_individual['weights'], final_fitness_scores[best_idx], results_per_generation


# Main function to run the algorithm and save results
def run_evolutionary_strategy():
    # Define population sizes and generation counts to test
    population_sizes = [100, 200, 300]
    generation_counts = [200, 300, 500]

    total_combinations = len(population_sizes) * len(generation_counts)  # Total combinations of params
    print(f"Total combinations to test: {total_combinations}")

    combination_counter = 1
    best_sharpe = -np.inf  # Track the best Sharpe ratio
    best_combination = None  # Track best combination of parameters
    best_portfolio_overall = None  # Track the best portfolio weights
    best_combination_number = None  # Track best combination number
    results = []  # Track results for all generations

    # Iterate through population sizes and generation counts
    for pop_size in population_sizes:
        for gen_count in generation_counts:
            print(
                f"Running combination {combination_counter}/{total_combinations}: population_size={pop_size}, generations={gen_count}")

            # Run the evolutionary strategy for the current parameters
            best_portfolio, sharpe_ratio, generation_results = evolutionary_strategy(
                expected_returns, cov_matrix, pop_size, gen_count, risk_free_rate
            )

            # Update combination number in generation results
            for entry in generation_results:
                entry['combination_number'] = combination_counter

            # Collect results
            results.extend(generation_results)

            # Print the Sharpe ratio for the current combination
            print(f"Sharpe ratio for combination {combination_counter}/{total_combinations}: {sharpe_ratio}")

            # Update the best portfolio if current combination is better
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_combination = (pop_size, gen_count)
                best_portfolio_overall = best_portfolio
                best_combination_number = combination_counter

            combination_counter += 1

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)

    # Output the best result
    print("\nBest combination found:")
    print(f"Combination number: {best_combination_number}/{total_combinations}")
    print(f"Sharpe ratio: {best_sharpe}")
    print(f"Best portfolio weights:\n{best_portfolio_overall}")

    # Save the best portfolio to CSV
    best_portfolio_df = pd.DataFrame([best_portfolio_overall], columns=returns_df.columns)
    best_portfolio_df.to_csv(os.path.join(script_dir, '../3.prob2_output/3.6aes_best_portfolio.csv'), index=False)
    print(f"Best portfolio saved in '3.6aes_best_portfolio.csv'")

# Run the evolutionary strategy algorithm
if __name__ == "__main__":
    run_evolutionary_strategy()
