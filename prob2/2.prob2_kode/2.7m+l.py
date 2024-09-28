import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fintd the path for this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths for input and output files based on the repository placement of the script
returns_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')

cov_matrix_file = os.path.join(script_dir, '../3.prob2_output/3.2beregn_kovarians_matrix.csv')
results_file = os.path.join(script_dir, '../3.prob2_output/3.7m+l.csv')

# Load data from the CSV file with monthly returns
returns_df = pd.read_csv(returns_file, index_col='Date', parse_dates=True)

# Calculate the expected (mean) return from each asset
expected_returns = returns_df.mean().values
num_assets = len(expected_returns)

# Load the covariance matrix from the CSV file
cov_matrix = pd.read_csv(cov_matrix_file, index_col=0).values


# # Function to calculate portfolio performance (return and risk)
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
    weights = np.random.random(num_assets)  # Tilfeldige vekter
    weights /= np.sum(weights)  # Sørg for at vektene summerer til 1

    sigma = np.random.uniform(0.05, 0.2, num_assets)

    return weights, sigma


# Function to generate a population of portfolios
def generate_population(pop_size, num_assets):
    # Return list of portfolios
    return np.array([generate_portfolio(num_assets) for _ in range(pop_size)])


# Function to mutate portfolios with self-adaptive mutation rate
def mutate_portfolio(portfolio, sigma):
    # Calculates Tau and Tau prime, that are used for controlling the self-adaptive mutation process
        # Scaling factor determining the influence of individual mutations on the mutation rates
    tau = 1 / np.sqrt(2 * np.sqrt(len(portfolio)))
        # Scaling factor influencing the overall adaptation of sigma values
    tau_prime = 1 / np.sqrt(2 * len(portfolio))

    # Update the sigma values, by mutating the old once
    new_sigma = sigma * np.exp(tau_prime * np.random.normal() + tau * np.random.normal(size=len(sigma)))

    # Mutating the portfolio weights based on the new sigma
    mutation = np.random.normal(0, new_sigma, len(portfolio))

    # Mutation values added to original weights, mutating the portfolio
    mutated_portfolio = portfolio + mutation

    # Ensure weights are within valid bounds (0-1) and normalize
    mutated_portfolio = np.clip(mutated_portfolio, 0, 1)
    mutated_portfolio /= np.sum(mutated_portfolio)

    # Return mutated portfolio and updated mutation step sizes
    return mutated_portfolio, new_sigma


# Function for selecting best μ portfolios based on the fitness(Sharpe ratio)
def select_best_individuals(population, expected_returns, cov_matrix, risk_free_rate, num_to_select):
    # Calculating the fitness for each portfolio
    fitness_scores = [fitness_function(weights, expected_returns, cov_matrix, risk_free_rate) for weights, _ in population]

    # Sorts the fitness score and its corresponding portfolio in descending order based on the fitness score
    sorted_population = [portfolio for _, portfolio in sorted(zip(fitness_scores, population), reverse=True)]

    # Returns the given number of the best portfolios
    return sorted_population[:num_to_select]


def run_MuPlussLambda_algorithm(pop_size_mu, pop_size_lambda,num_generations, risk_free_rate):
    #Generate the initial population of Mu, μ (parents)
    population = generate_population(pop_size_mu, num_assets)

    # Starting with lowest possible Sharpe ratio
    runs_best_sharpe_ratio = -np.inf

    # For each generation in the given number of generations
    for generation in range(num_generations):
        # Generate Lambda, λ, children by mutating the parent population
        offspring = [mutate_portfolio(population[np.random.randint(0, pop_size_mu)][0], population[np.random.randint(0, pop_size_mu)][1])for _ in range(pop_size_lambda)]

        # Combining the parent populating and the children population into one
        combined_population = np.vstack((population, offspring))

        # Selecting the top Mu, μ, individuals, based on fitness, from the combined population, and them becoming the new parent population
        population = select_best_individuals(combined_population, expected_returns, cov_matrix, risk_free_rate, pop_size_mu)

        # Finding the best fitness(Sharpe ratio) and corresponding individual(portfolio) from this generation
        generation_best_sharpe_ratio = fitness_function(population[0][0], expected_returns, cov_matrix, risk_free_rate)
        generation_best_portfolio = population[0][0]

        # Updating the runs best Sharpe ratio and portfolio if the best in this generations is better
        if generation_best_sharpe_ratio > runs_best_sharpe_ratio:
            runs_best_sharpe_ratio = generation_best_sharpe_ratio
            runs_best_portfolio = generation_best_portfolio

        # Calculate the expected returns and standard deviation for the generations best portfolio
        generation_best_expected_return, generation_best_portfolio_stddev = portfolio_performance(generation_best_portfolio, expected_returns, cov_matrix)

        # Print the results for the current generation
        print(f"Population Size μ: {pop_size_mu}, Population Size λ: {pop_size_lambda}, Generations: {num_generations}")
        print(f"Generation {generation + 1}: Best Sharpe Ratio = {generation_best_sharpe_ratio:.4f}, "
              f"Expected Return = {generation_best_expected_return:.4f}, "
              f"Portfolio Std Dev = {generation_best_portfolio_stddev:.4f}")

    return runs_best_portfolio, runs_best_sharpe_ratio


# Define the population, Mu and Lambda, sizes, and generation counts to test
population_sizes_mu = [10, 50, 100]
population_sizes_lambda = [20, 100, 200]
generation_counts = [50, 100, 200]

# Define risk-free rate (annual 2% rate, converted to monthly)
risk_free_rate = 0.02 / 12


combination_counter = 1
# Totale combinations of parameters
total_combinations = len(population_sizes_mu) * len(population_sizes_lambda) * len(generation_counts)

best_sharpe = -np.inf # Track the best Sharpe ratio
best_combination = None # Track the best parameter combination
best_combination_number = -1 # Track the best combination number
best_portfolio_overall = None # Track best portfolio weights

results = [] # Track results for all generations


# Iterates through the population sizes and generation
for pop_size_mu in population_sizes_mu:
    for pop_size_lambda in population_sizes_lambda:
        for gen_count in generation_counts:
            print(
                f"Kjører kombinasjon {combination_counter}/{total_combinations}: pop_size_mu={pop_size_mu}, pop_size_lambda={pop_size_lambda} , gen_count={gen_count}")

            # Runs the mu+lambda function for the current parameters
            best_portfolio, sharpe_ratio = run_MuPlussLambda_algorithm(pop_size_mu, pop_size_lambda, gen_count, risk_free_rate)
            print(f"Sharpe-ratio for kombinasjon {combination_counter}/{total_combinations}: {sharpe_ratio}")

            # Collecting the values for each iteration in "results"
            results.append({
                'combination_number': combination_counter,
                'pop_size_mu': pop_size_mu,
                'pop_size_lambda': pop_size_lambda,
                'gen_count': gen_count,
                'sharpe_ratio': sharpe_ratio,
                'best_portfolio_weights': best_portfolio.tolist()

            })

            # Updates the best portfolio if current combination is better
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_combination = (pop_size_mu, pop_size_lambda, gen_count)
                best_portfolio_overall = best_portfolio
                best_combination_number = combination_counter

            combination_counter += 1

# Saving the results in a CSV
results_df = pd.DataFrame(results)
results_df.to_csv(results_file, index=False)


# Output the best results
print("\nBest combination found:")
print(f"Combination number: {best_combination_number}/{total_combinations}")
print(f"Best Parameters: {best_combination}")
print(f"Sharpe ratio: {best_sharpe}")
print(f"Best portfolio weights:\n{best_portfolio_overall}")

# Saving the best portfolio to a CSV
best_portfolio_df = pd.DataFrame([best_portfolio], columns=returns_df.columns)
best_portfolio_df.to_csv(os.path.join(script_dir, '../3.prob2_output/3.7m+l_best_portfolio.csv'), index=False)
print(f"\nBeste portefølje lagret i '3.7m+l_best_portfolio.csv'")




# Visualizing the best portfolio in /4.prob2_visual/3.7_best_portfolio_allocation.png
def plot_best_portfolio_weights(best_portfolio_df):
    # Plot the asset allocation of the best portfolio
    best_portfolio_df.T.plot(kind='bar', legend=False, figsize=(10, 6))
    plt.title('Best Portfolio Asset Allocation')
    plt.ylabel('Weight')
    plt.grid(True)
    plt.savefig(os.path.join(script_dir, '../4.prob2_visual/3.7_best_portfolio_allocation.png'), dpi=300, bbox_inches='tight')
    plt.close()

plot_best_portfolio_weights(best_portfolio_df)