import numpy as np
import pandas as pd
import os

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

# These are testing parameters <--- Make adaptive?
population_sizes_mu = [10, 25, 50]
population_sizes_lambda = [100, 1000, 2000]
generation_counts = [100, 300, 600]
risk_free_rate = 0.02 / 12

# Calculates the portfolios expected return and risk using standard deviation
def portfolio_performance(weights, expected_returns, cov_matrix):
    expected_return = np.dot(weights, expected_returns) # Calculates 
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_stddev = np.sqrt(portfolio_variance)
    return expected_return, portfolio_stddev

# Method for evaluating the fitness of the current population using Sharpe - Ratio
def fitness_evaluation(weights, expected_returns, cov_matrix, risk_free_rate):
    expected_return, portfolio_stddev = portfolio_performance(weights, expected_returns, cov_matrix)
    sharpe_ratio = (expected_return - risk_free_rate) / portfolio_stddev
    return sharpe_ratio

sigma = 1.0
success_count = 0
mutation_attempts = 5

def adapt_sigma(sigma, success_count, mutation_attempts, tau=0.1, min_sigma=0.001, max_sigma=1.0):
    success_rate = success_count / mutation_attempts

    if success_rate > 0.2:
        sigma = sigma * np.exp(tau)
    else:
        sigma = sigma * np.exp(-tau)

    tau = 0.1 if success_rate < 0.1 else 0.05 if success_rate > 0.3 else tau

    sigma = max(min_sigma, min(sigma, max_sigma))

    return sigma
    
########################################################################################################

# Method for initializing the original random population
def initialize_population(size, num_assets):
    population = np.random.rand(size, num_assets)
    # Normalizes the weights for all the population
    normalized_population = population / population.sum(axis=1)[:, np.newaxis] # New axis to make the array 2d
    return normalized_population

def select_best_children(offspring, expected_returns, cov_matrix, risk_free_rate, num_to_select):
    # Calculate fitness scores for all offspring
    fitness_scores = np.array([fitness_evaluation(weights, expected_returns, cov_matrix, risk_free_rate) for weights in offspring])

    # Ensure num_to_select does not exceed the number of offspring
    num_to_select = min(num_to_select, len(offspring))
    
    # Get indices of the best children
    best_children_indices = np.argsort(fitness_scores)[-num_to_select:]

    return offspring[best_children_indices]  # Return the best children only


def mutate_population(parents, lambda_, sigma, expected_returns, cov_matrix, risk_free_rate):
    num_parents, num_assets = parents.shape  # Ensure parents is 2D
    offspring = np.zeros((lambda_, num_assets))

    parent_fitness = np.array([fitness_evaluation(parent, expected_returns, cov_matrix, risk_free_rate) for parent in parents])
    
    for i in range(lambda_):
        parent = parents[np.random.randint(num_parents)]
        mutation = np.random.randn(num_assets) * sigma
        child = parent + mutation
        child = np.maximum(child, 0)
        offspring[i] = child / np.sum(child)  # Normalize the child

    offspring_fitness = np.array([fitness_evaluation(child, expected_returns, cov_matrix, risk_free_rate) for child in offspring])

    improvement = np.max(offspring_fitness) - np.max(parent_fitness)

    new_sigma = adapt_sigma(sigma, success_count, mutation_attempts)

    return offspring, new_sigma

# Runs the evolution algorithm
def evolution_strategy(expected_returns, cov_matrix, mu, lambda_, num_generations, risk_free_rate, initial_sigma=0.01):
    #Initializations
    population = initialize_population(mu, num_assets) # Call to initialize population
    sigma = initial_sigma
    best_sharpe_ratio = -np.inf # Initializes the Sharpe Ratio so the algorithm can keep track of the highest Sharpe Ratio so far
    best_portfolio = None # Initializes the best_portfolio variable
    sharpe_ratios_per_gen = [] # Initializes the Sharpe Ratios per generation array
    sigmas_per_gen = []

    # Loops through the generation
    for generation in range(num_generations):
        offspring, sigma = mutate_population(population, lambda_, sigma, expected_returns, cov_matrix, risk_free_rate)
        population = select_best_children(offspring, expected_returns, cov_matrix, risk_free_rate, mu)

        generation_best_sharpe_ratio = fitness_evaluation(population[0], expected_returns, cov_matrix, risk_free_rate)
        generation_best_portfolio = population[0]

        sharpe_ratios_per_gen.append(generation_best_sharpe_ratio)
        sigmas_per_gen.append(sigma)

        if generation_best_sharpe_ratio > best_sharpe_ratio:
            best_sharpe_ratio = generation_best_sharpe_ratio
            best_portfolio = generation_best_portfolio
        
        generation_best_expected_return, generation_best_portfolio_stddev = portfolio_performance(generation_best_portfolio, expected_returns, cov_matrix)
        print(f"Population Size μ: {mu}, Population Size λ: {lambda_}, Generations: {num_generations}, Mutation Rate: {sigma:.5f}")
        print(f"Generation {generation + 1}: Best Sharpe Ratio = {generation_best_sharpe_ratio:.4f}, "
              f"Expected Return = {generation_best_expected_return:.4f}, "
              f"Portfolio Std Dev = {generation_best_portfolio_stddev:.4f}")

    return best_portfolio, best_sharpe_ratio, sharpe_ratios_per_gen, sigmas_per_gen

best_sharpe = -np.inf
best_combination = None
best_combination_number = -1
combination_counter = 1
total_combinations = len(population_sizes_mu) * len(population_sizes_lambda) * len(generation_counts)
results = []

for pop_size_mu in population_sizes_mu:
    for pop_size_lambda in population_sizes_lambda:
        for gen_countt in generation_counts:
            print(
                f"Running combination {combination_counter}/{total_combinations}: pop_size_mu={pop_size_mu}, pop_size_lambda={pop_size_lambda} , gen_count={"fix"}, mut_rate={"fix"}")
            
            best_portfolio, sharpe_ratio, sharpe_ratios_per_gen, sigmas_per_gen = evolution_strategy(expected_returns, cov_matrix, pop_size_mu, pop_size_lambda, gen_countt, risk_free_rate)
            print(f"Sharpe-ratio for combination {combination_counter}/{total_combinations}: {sharpe_ratio}")

            for generation, (sharpe_ratios_per_gen, sigma_gen) in enumerate(zip(sharpe_ratios_per_gen, sigmas_per_gen)):
                results.append({
                    'combination_number': combination_counter,
                    'pop_size_mu': pop_size_mu,
                    'pop_size_lambda': pop_size_lambda,
                    'gen_count': gen_countt,
                    'mut_rate': f"{sigmas_per_gen[generation]:.5f}",
                    'generation': generation + 1,
                    'sharpe_ratio': sharpe_ratios_per_gen,
                    'best_portfolio_weights': best_portfolio.tolist()
                })

            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_combination = (pop_size_mu, pop_size_lambda, gen_countt, sigmas_per_gen)
                best_combination_number = combination_counter

            combination_counter += 1
            

# Saving the results
results_df = pd.DataFrame(results)
results_df.to_csv(results_file, index=False)

print("\nBest Portfolio Weights:", best_portfolio, flush=True )
print("\nBest Sharpe Ratio:", best_sharpe, flush=True )
print("Best Parameters:", best_combination, flush=True )
print("Best Combination Number:", best_combination_number, flush=True )

best_portfolio_df = pd.DataFrame([best_portfolio], columns=returns_df.columns)
best_portfolio_df.to_csv(os.path.join(script_dir, '../3.prob2_output/3.8m,l_best_portfolio.csv'), index=False)
print(f"\nBest portfolio saved in '3.8m,l_best_portfolio.csv'", flush=True )

csv_file_path = os.path.join(script_dir, '../3.prob2_output/3.8m,l.csv')

df = pd.read_csv(csv_file_path)

summary_table = df[['combination_number', 'pop_size_mu', 'pop_size_lambda', 'gen_count']].drop_duplicates()

print(summary_table.to_string(index=False))
