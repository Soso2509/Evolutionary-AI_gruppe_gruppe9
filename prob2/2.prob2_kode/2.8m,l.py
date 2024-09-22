import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Finds the path to this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# This is the data
returns_path = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')
cov_matrix_path = os.path.join(script_dir, '../3.prob2_output/3.2beregn_kovarians_matrix.csv')
results_file = os.path.join(script_dir, '../3.prob2_output/3.8m,l.csv')

# Puts the data in a pandas dataframe and uses Date as index
returns_df = pd.read_csv(returns_path, index_col='Date', parse_dates=True)
expected_returns = returns_df.mean().values
num_assets = len(expected_returns)

# Loads in the covariancematrix generated in 2.2 
cov_matrix = pd.read_csv(cov_matrix_path, index_col=0).values

# These are testing parameters
population_sizes_mu = [20, 50, 100]
population_sizes_lambda = [10, 100, 200]
generation_counts = [50, 100, 200]
mutation_rates = [0.01, 0.05, 0.1]
risk_free_rate = 0.02 / 12

# Method for initializing the original random population
def initialize_population(size, num_assets):
    population = np.random.rand(size, num_assets)
    # Normalizes the weights for all the population using 2D array
    normalized_population = population / population.sum(axis=1)[:, np.newaxis]
    
    print(f"Initialized population shape: {normalized_population.shape}")  # Debugging statement
    return normalized_population

# Calculates the portfolios expected return and risk using standard deviation
def portfolio_performance(weights, expected_returns, cov_matrix):
    expected_return = np.dot(weights, expected_returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_stddev = np.sqrt(portfolio_variance)
    return expected_return, portfolio_stddev

# Method for evaluating the fitness of the current population using Sharpe - Ratio
def fitness_evaluation(weights, expected_returns, cov_matrix, risk_free_rate):
    expected_return, portfolio_stddev = portfolio_performance(weights, expected_returns, cov_matrix)
    sharpe_ratio = (expected_return - risk_free_rate) / portfolio_stddev
    return sharpe_ratio


def select_best_children(offspring, expected_returns, cov_matrix, risk_free_rate, num_to_select):
    # Calculate fitness scores for all offspring
    fitness_scores = np.array([fitness_evaluation(weights, expected_returns, cov_matrix, risk_free_rate) for weights in offspring])

    # Ensure num_to_select does not exceed the number of offspring
    num_to_select = min(num_to_select, len(offspring))
    
    # Get indices of the best children
    best_children_indices = np.argsort(fitness_scores)[-num_to_select:]

    return offspring[best_children_indices]  # Return the best children only


def generate_children(parents, lambda_, mutation_rate):
    num_parents, num_assets = parents.shape  # Ensure parents is 2D
    offspring = np.zeros((lambda_, num_assets))
    for i in range(lambda_):
        parent = parents[np.random.randint(num_parents)]
        mutation = np.random.randn(num_assets) * mutation_rate
        child = parent + mutation
        child = np.maximum(child, 0)
        offspring[i] = child / np.sum(child)  # Normalize the child
    return offspring


# Runs the actual evolution algorithm
def evolution_strategy(expected_returns, cov_matrix, mu, lambda_, num_generations, mutation_rate, risk_free_rate):
    population = initialize_population(mu, num_assets)
    best_sharpe_ratio = -np.inf
    best_portfolio = None

    sharpe_ratios_per_gen = []

    for generation in range(num_generations):
        offspring = generate_children(population, lambda_, mutation_rate)
        population = select_best_children(offspring, expected_returns, cov_matrix, risk_free_rate, mu)

        generation_best_sharpe_ratio = fitness_evaluation(population[0], expected_returns, cov_matrix, risk_free_rate)
        generation_best_portfolio = population[0]

        sharpe_ratios_per_gen.append(generation_best_sharpe_ratio)

        if generation_best_sharpe_ratio > best_sharpe_ratio:
            best_sharpe_ratio = generation_best_sharpe_ratio
            best_portfolio = generation_best_portfolio
        
        generation_best_expected_return, generation_best_portfolio_stddev = portfolio_performance(generation_best_portfolio, expected_returns, cov_matrix)
        print(f"Population Size μ: {mu}, Population Size λ: {lambda_}, Generations: {num_generations}, Mutation Rate: {mutation_rate}")
        print(f"Generation {generation + 1}: Best Sharpe Ratio = {generation_best_sharpe_ratio:.4f}, "
              f"Expected Return = {generation_best_expected_return:.4f}, "
              f"Portfolio Std Dev = {generation_best_portfolio_stddev:.4f}")

    return best_portfolio, best_sharpe_ratio, sharpe_ratios_per_gen

best_portfolio, best_sharpe_ratio, sharpe_ratios_per_gen = evolution_strategy(
    expected_returns, cov_matrix, mu=20, lambda_=200, num_generations=100, mutation_rate=0.01, risk_free_rate=risk_free_rate
)

sharpe_ratios_array = np.array(sharpe_ratios_per_gen)

peaks, _ = find_peaks(sharpe_ratios_array)

plt.figure(figsize=(10, 6))
plt.plot(sharpe_ratios_array, label='Sharpe Ratio Progression', color='white')

plt.plot(peaks, sharpe_ratios_array[peaks], 'ro', label='Local Maxima')

global_max_idx = np.argmax(sharpe_ratios_array)
plt.plot(global_max_idx, sharpe_ratios_array[global_max_idx], 'go', label='Global Maxima')

plt.title("Sharpe Ratio Progression over Generations")
plt.xlabel("Generations")
plt.ylabel("Sharpe Ratio")
plt.grid(True)
plt.legend()

plt.gca().set_facecolor('black')
plt.show()

best_sharpe = -np.inf
best_combination = None
best_combination_number = -1
combination_counter = 1
total_combinations = len(population_sizes_mu) * len(population_sizes_lambda) * len(generation_counts) * len(mutation_rates)
results = []

for pop_size_mu in population_sizes_mu:
    for pop_size_lambda in population_sizes_lambda:
        for gen_countt in generation_counts:
            for mutt_rate in mutation_rates:
                print(
                    f"Running combination {combination_counter}/{total_combinations}: pop_size_mu={pop_size_mu}, pop_size_lambda={pop_size_lambda} , gen_count={"fix"}, mut_rate={"fix"}")
                
                best_portfolio, sharpe_ratio, sharpe_ratios_per_gen = evolution_strategy(expected_returns, cov_matrix, pop_size_mu, pop_size_lambda, gen_countt, mutt_rate, risk_free_rate)
                print(f"Sharpe-ratio for combination {combination_counter}/{total_combinations}: {sharpe_ratio}")

                results.append({
                    'combination_number': combination_counter,
                    'pop_size_mu': pop_size_mu,
                    'pop_size_lambda': pop_size_lambda,
                    'gen_count': gen_countt,
                    'mut_rate': mutt_rate,
                    'sharpe_ratio': sharpe_ratio,
                    'best_portfolio_weights': best_portfolio.tolist()
                })

                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_combination = (pop_size_mu, pop_size_lambda, gen_countt, mutt_rate)
                    best_combination_number = combination_counter

                combination_counter += 1

results_df = pd.DataFrame(results)

results_df.to_csv(results_file, index=False)

print("\nBest Portfolio Weights:", best_portfolio)
print("\nBest Sharpe Ratio:", best_sharpe)
print("Best Parameters:", best_combination)
print("Best Combination Number:", best_combination_number)

best_portfolio_df = pd.DataFrame([best_portfolio], columns=returns_df.columns)
best_portfolio_df.to_csv(os.path.join(script_dir, '../3.prob2_output/3.8m,l_best_portfolio.csv'), index=False)
print(f"\nBest portfolio saved in '3.8m,l_best_portfolio.csv'")

# Sharpe vs Mutation plt <---- Little iffy

sharpe_ratios = results_df['sharpe_ratio']
mutation_rates = results_df['mut_rate']
pop_sizes_mu = results_df['pop_size_mu']
pop_sizes_lambda = results_df['pop_size_lambda']
generation_counts = results_df['gen_count']
combination_numbers = results_df['combination_number']

plt.figure(figsize=(8, 6))
plt.plot(mutation_rates, sharpe_ratios, 'bo-', label='Sharpe Ratio')
plt.xlabel('Mutation Rate')
plt.ylabel('Sharpe Ratio')
plt.title('Sharpe Ratio vs Mutation Rate')
plt.grid(True)
plt.legend()

plt.show()

# Sharpe vs Population Size (μ)  <--- not great

plt.figure(figsize=(8, 6))
plt.plot(pop_sizes_mu, sharpe_ratios, 'go-', label='Sharpe Ratio')
plt.xlabel('Population Size (μ)')
plt.ylabel('Sharpe Ratio')
plt.title('Sharpe Ratio vs Population Size (μ)')
plt.grid(True)
plt.legend()

plt.show()

# Sharpe vs Population Size (λ)  <--- not great

plt.figure(figsize=(8, 6))
plt.plot(pop_sizes_lambda, sharpe_ratios, 'ro-', label='Sharpe Ratio')
plt.xlabel('Population Size (λ)')
plt.ylabel('Sharpe Ratio')
plt.title('Sharpe Ratio vs Population Size (λ)')
plt.grid(True)
plt.legend()

plt.show()

# Sharpe vs Generation Count

plt.figure(figsize=(8, 6))
plt.plot(generation_counts, sharpe_ratios, 'mo-', label='Sharpe Ratio')
plt.xlabel('Generation Count')
plt.ylabel('Sharpe Ratio')
plt.title('Sharpe Ratio vs Generation Count')
plt.grid(True)
plt.legend()

plt.show()

# Sharpe vs Combination Number

plt.figure(figsize=(8, 6))
plt.plot(combination_numbers, sharpe_ratios, 'co-', label='Sharpe Ratio')
plt.xlabel('Combination Number')
plt.ylabel('Sharpe Ratio')
plt.title('Sharpe Ratio vs Combination Number')
plt.grid(True)
plt.legend()

plt.show()
