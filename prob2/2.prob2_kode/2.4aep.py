import os
import numpy as np
import pandas as pd

# Define file paths for the input and output CSV files
script_dir = os.path.dirname(os.path.abspath(__file__))
returns_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')
cov_matrix_file = os.path.join(script_dir, '../3.prob2_output/3.2beregn_kovarians_matrix.csv')
results_file = os.path.join(script_dir, '../3.prob2_output/3.4aep.csv')

# Load data for expected returns and covariance matrix
returns_df = pd.read_csv(returns_file, index_col='Date', parse_dates=True)
expected_returns = returns_df.mean().values  # Calculate average returns for each asset
num_assets = len(expected_returns)  # Total number of assets
cov_matrix = pd.read_csv(cov_matrix_file, index_col=0).values  # Covariance matrix for assets
risk_free_rate = 0.02 / 12  # Monthly risk-free rate


# Calculate portfolio performance in terms of expected return and standard deviation
def portfolio_performance(weights, expected_returns, cov_matrix):
    expected_return = np.dot(weights, expected_returns)  # Weighted return
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))  # Portfolio variance
    portfolio_stddev = np.sqrt(portfolio_variance)  # Portfolio risk (std deviation)
    return expected_return, portfolio_stddev


# Fitness function to calculate Sharpe ratio for a given portfolio
def fitness_function(weights, expected_returns, cov_matrix, risk_free_rate):
    expected_return, portfolio_stddev = portfolio_performance(weights, expected_returns, cov_matrix)
    if portfolio_stddev == 0:  # Avoid division by zero
        return -np.inf
    sharpe_ratio = (expected_return - risk_free_rate) / portfolio_stddev  # Sharpe ratio calculation
    return sharpe_ratio


# Generate an individual portfolio with random weights and initial mutation rates (sigma)
def generate_portfolio(num_assets):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)  # Normalize weights to sum to 1
    sigma = np.random.uniform(0.05, 0.2, num_assets)  # Initial mutation rates for each asset
    return {'weights': weights, 'sigma': sigma}


# Generate an initial population of portfolios
def generate_population(pop_size, num_assets):
    return [generate_portfolio(num_assets) for _ in range(pop_size)]


# Self-adaptive mutation function to modify weights and sigma (mutation rate) for an individual
def mutate_portfolio(individual):
    weights = individual['weights']
    sigma = individual['sigma']
    num_assets = len(weights)

    # Define global and individual mutation rates (self-adaptive parameters)
    tau = 1 / np.sqrt(2 * np.sqrt(num_assets))
    tau_prime = 1 / np.sqrt(2 * num_assets)

    # Update sigma values with self-adaptive mutation strategy
    sigma_prime = sigma * np.exp(tau_prime * np.random.normal() + tau * np.random.normal(size=num_assets))
    sigma_prime = np.clip(sigma_prime, 1e-6, 1)  # Bound sigma to avoid zero or excessively large values

    # Mutate weights using the new sigma values
    weights_prime = weights + sigma_prime * np.random.normal(size=num_assets)
    weights_prime = np.clip(weights_prime, 0, 1)  # Bound weights between 0 and 1
    if np.sum(weights_prime) == 0:
        weights_prime = generate_portfolio(num_assets)['weights']  # Reinitialize weights if they sum to zero
    else:
        weights_prime /= np.sum(weights_prime)  # Normalize weights to sum to 1

    return {'weights': weights_prime, 'sigma': sigma_prime}


# Tournament selection to select parents based on fitness scores
def tournament_selection(population, fitness_scores, tournament_size):
    selected = []
    pop_size = len(population)
    for _ in range(pop_size):
        participants = np.random.choice(pop_size, tournament_size, replace=False)  # Randomly select participants
        participant_fitness = fitness_scores[participants]  # Fitness scores of participants
        winner_idx = participants[np.argmax(participant_fitness)]  # Select the best individual from participants
        selected.append(population[winner_idx])
    return selected


# Main evolutionary programming algorithm
def advanced_evolutionary_programming(
        expected_returns, cov_matrix, population_size, num_generations,
        risk_free_rate, tournament_size=3, num_elites=1
):
    num_assets = len(expected_returns)
    population = generate_population(population_size, num_assets)  # Initial population
    best_sharpe_per_generation = []  # Track best Sharpe ratio in each generation
    generation_details = []  # Store best Sharpe ratio per generation

    # Evolve over multiple generations
    for generation in range(num_generations):
        # Calculate fitness scores for each individual in the population
        fitness_scores = np.array([
            fitness_function(ind['weights'], expected_returns, cov_matrix, risk_free_rate)
            for ind in population
        ])

        # Select the top `num_elites` individuals based on fitness scores
        elite_indices = np.argsort(fitness_scores)[-num_elites:]  # Indices of elite individuals
        elites = [population[idx] for idx in elite_indices]  # Elite individuals to keep in the next generation
        best_sharpe_per_generation.append(fitness_scores[elite_indices[-1]])  # Best Sharpe ratio in this generation

        # Record generation data
        generation_details.append({
            'generation': generation + 1,
            'best_sharpe': fitness_scores[elite_indices[-1]]
        })

        # Select parents using tournament selection and produce offspring through mutation
        selected_parents = tournament_selection(population, fitness_scores, tournament_size)
        offspring = [mutate_portfolio(parent) for parent in selected_parents]

        # Form new population with elites and offspring
        population = elites + offspring[:population_size - num_elites]  # Combine elites with offspring to maintain size

    # Final evaluation to get the best individual in the last generation
    final_fitness_scores = np.array([
        fitness_function(ind['weights'], expected_returns, cov_matrix, risk_free_rate)
        for ind in population
    ])
    best_idx = np.argmax(final_fitness_scores)
    best_individual = population[best_idx]

    # Return the best weights, final Sharpe ratio, and generation details
    return best_individual['weights'], final_fitness_scores[best_idx], generation_details


# Run the evolutionary programming algorithm with different parameter combinations
def run_advanced_ep():
    population_sizes = [100, 200, 300]  # Test with various population sizes
    generation_counts = [200, 300, 500]  # Test with different generation counts
    tournament_sizes = [1]  # Tournament sizes for parent selection
    num_elites_list = [1, 3]  # Number of elites to keep in each generation

    # Calculate the total number of parameter combinations to test
    total_combinations = (
            len(population_sizes) *
            len(generation_counts) *
            len(tournament_sizes) *
            len(num_elites_list)
    )
    print(f"Total number of combinations to test: {total_combinations}")

    combination_counter = 1  # Track the current combination number
    best_sharpe = -np.inf  # Initialize best Sharpe ratio
    best_combination = None  # Store parameters for the best combination
    best_portfolio_overall = None  # Track best portfolio
    results = []  # Store results for each combination and generation

    # Iterate over each combination of parameters
    for pop_size in population_sizes:
        for gen_count in generation_counts:
            for tour_size in tournament_sizes:
                for num_elites in num_elites_list:
                    print(f"Running combination {combination_counter}/{total_combinations}: "
                          f"population size={pop_size}, generations={gen_count}, "
                          f"tournament size={tour_size}, number of elites={num_elites}")

                    # Run the evolutionary programming algorithm with current parameters
                    weights, sharpe_ratio, generation_details = advanced_evolutionary_programming(
                        expected_returns, cov_matrix, pop_size, gen_count, risk_free_rate,
                        tournament_size=tour_size, num_elites=num_elites
                    )

                    # Record each generation's results
                    for gen_detail in generation_details:
                        results.append({
                            'combination_number': combination_counter,
                            'pop_size': pop_size,
                            'gen_count': gen_count,
                            'tournament_size': tour_size,
                            'num_elites': num_elites,
                            'generation': gen_detail['generation'],
                            'sharpe_ratio': gen_detail['best_sharpe'],
                        })

                    # Update best combination if current Sharpe ratio is the highest found
                    if sharpe_ratio > best_sharpe:
                        best_sharpe = sharpe_ratio
                        best_combination = (pop_size, gen_count, tour_size, num_elites)
                        best_portfolio_overall = weights

                    combination_counter += 1  # Move to the next combination

    # Save all results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)

    # Output the best combination found
    print("\nBest combination found:")
    print(f"Sharpe ratio: {best_sharpe}")
    print(f"Best portfolio weights:\n{best_portfolio_overall}")

    # Save the best portfolio weights to a CSV file
    best_portfolio_df = pd.DataFrame([best_portfolio_overall], columns=returns_df.columns)
    best_portfolio_df.to_csv(os.path.join(script_dir, '../3.prob2_output/3.4aep_best_portfolio.csv'), index=False)


# Run the script
if __name__ == "__main__":
    run_advanced_ep()
