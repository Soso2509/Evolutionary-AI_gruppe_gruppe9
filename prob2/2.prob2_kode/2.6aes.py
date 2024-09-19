import os
import numpy as np
import pandas as pd

# Finn stien til dette skriptet (der filen kjører fra)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Definer stier for input- og output-filer basert på skriptets plassering
returns_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')
cov_matrix_file = os.path.join(script_dir, '../3.prob2_output/3.2beregn_kovarians_matrix.csv')
results_file = os.path.join(script_dir, '../3.prob2_output/3.6aes.csv')

# Last inn dataene fra CSV-filen med månedlige avkastninger
returns_df = pd.read_csv(returns_file, index_col='Date', parse_dates=True)

# Beregn forventet (gjennomsnittlig) avkastning for hver aksje
expected_returns = returns_df.mean().values  # Gjennomsnittlig månedlig avkastning for hver aksje
num_assets = len(expected_returns)  # Antall aksjer i porteføljen

# Last inn kovariansmatrisen fra CSV-filen du allerede har generert
cov_matrix = pd.read_csv(cov_matrix_file, index_col=0).values


# Funksjon for å beregne porteføljens forventede avkastning og standardavvik (risiko)
def portfolio_performance(weights, expected_returns, cov_matrix):
    expected_return = np.dot(weights, expected_returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_stddev = np.sqrt(portfolio_variance)
    return expected_return, portfolio_stddev


# Fitness-funksjon: Beregn Sharpe-ratio
def fitness_function(weights, expected_returns, cov_matrix, risk_free_rate):
    expected_return, portfolio_stddev = portfolio_performance(weights, expected_returns, cov_matrix)
    sharpe_ratio = (expected_return - risk_free_rate) / portfolio_stddev
    return sharpe_ratio


# Funksjon for å generere en tilfeldig portefølje
def generate_portfolio(num_assets):
    weights = np.random.random(num_assets)  # Tilfeldige vekter
    weights /= np.sum(weights)  # Sørg for at vektene summerer til 1
    return weights


# Funksjon for å generere en populasjon av porteføljer
def generate_population(pop_size, num_assets):
    return np.array([generate_portfolio(num_assets) for _ in range(pop_size)])


# Funksjon for å mutere porteføljer med selv-adaptiv mutasjonsrate
def mutate_portfolio(portfolio, mutation_rate):
    mutation = np.random.normal(0, mutation_rate, len(portfolio))
    mutated_portfolio = portfolio + mutation
    mutated_portfolio = np.clip(mutated_portfolio, 0, 1)  # Sikrer at ingen vekt er negativ
    mutated_portfolio /= np.sum(mutated_portfolio)  # Sikrer at vektene fortsatt summerer til 1
    return mutated_portfolio


def blx_alpha_recombination(parents, alpha=0.3):
    """
    Blend Crossover (BLX-α) method for generating an offspring from two parent portfolios.

    Args:
        parents (np.ndarray): A 2D array of shape (2, num_assets) where each row is a parent portfolio.
        alpha (float): The alpha parameter controls the range extension beyond the parents' values.

    Returns:
        np.ndarray: A 1D array representing the offspring portfolio weights.
    """
    parent1, parent2 = parents
    num_assets = len(parent1)
    offspring = np.zeros(num_assets)

    for i in range(num_assets):
        c_min = min(parent1[i], parent2[i])
        c_max = max(parent1[i], parent2[i])
        range_extension = alpha * (c_max - c_min)

        # Random value within the extended range [c_min - range_extension, c_max + range_extension]
        offspring[i] = np.random.uniform(c_min - range_extension, c_max + range_extension)

    # Clip to ensure values are within [0, 1] and normalize to sum to 1
    offspring = np.clip(offspring, 0, 1)
    offspring /= np.sum(offspring)

    return offspring


# Recombine Portfolios
def recombine_portfolios(parents, alpha=0.5):
    return blx_alpha_recombination(parents, alpha)

# Run ES Algorithm
def run_es_algorithm(pop_size, num_generations, initial_mutation_rate, risk_free_rate, alpha):
    population = generate_population(pop_size, num_assets)
    best_sharpe_ratio = -np.inf
    best_portfolio = None

    mutation_rate = initial_mutation_rate
    mutation_step_size = 0.1

    for generation in range(num_generations):
        new_population = []
        for _ in range(pop_size):
            parents_indices = np.random.choice(pop_size, 2, replace=False)
            parent_portfolios = population[parents_indices]
            child_portfolio = recombine_portfolios(parent_portfolios, alpha)
            mutated_portfolio = mutate_portfolio(child_portfolio, mutation_rate)
            new_population.append(mutated_portfolio)

        population = np.array(new_population)
        fitness_scores = np.array([fitness_function(ind, expected_returns, cov_matrix, risk_free_rate) for ind in population])
        best_fitness_index = np.argmax(fitness_scores)
        generation_best_sharpe_ratio = fitness_scores[best_fitness_index]
        generation_best_portfolio = population[best_fitness_index]

        if generation_best_sharpe_ratio > best_sharpe_ratio:
            best_sharpe_ratio = generation_best_sharpe_ratio
            best_portfolio = generation_best_portfolio

        if generation_best_sharpe_ratio > best_sharpe_ratio:
            mutation_rate = max(0.01, mutation_rate - mutation_step_size)
        else:
            mutation_rate += mutation_step_size

        generation_best_expected_return, generation_best_portfolio_stddev = portfolio_performance(
            generation_best_portfolio, expected_returns, cov_matrix)
        print(f"Population Size: {pop_size}, Generations: {num_generations}, Initial Mutation Rate: {initial_mutation_rate}, Current Mutation Rate: {mutation_rate}, Alpha: {alpha}")
        print(f"Generation {generation + 1}: Best Sharpe Ratio = {generation_best_sharpe_ratio:.4f}, "
              f"Expected Return = {generation_best_expected_return:.4f}, "
              f"Portfolio Std Dev = {generation_best_portfolio_stddev:.4f}")

    return best_portfolio, best_sharpe_ratio

# Hyperparameter Testing
population_sizes = [20, 50, 100]
generation_counts = [50, 100, 200]
initial_mutation_rates = [0.01, 0.05, 0.1]
alphas = [0.1, 0.3, 0.5, 0.8, 1.0]  # Different values of alpha for BLX-α

risk_free_rate = 0.02 / 12
best_sharpe = -np.inf
best_combination = None
best_combination_number = -1
combination_counter = 1
total_combinations = len(population_sizes) * len(generation_counts) * len(initial_mutation_rates) * len(alphas)
results = []

for pop_size in population_sizes:
    for gen_count in generation_counts:
        for initial_mutation_rate in initial_mutation_rates:
            for alpha in alphas:
                print(f"Kjører kombinasjon {combination_counter}/{total_combinations}: pop_size={pop_size}, gen_count={gen_count}, initial_mutation_rate={initial_mutation_rate}, alpha={alpha}")

                best_portfolio, sharpe_ratio = run_es_algorithm(pop_size, gen_count, initial_mutation_rate, risk_free_rate, alpha)

                print(f"Sharpe-ratio for kombinasjon {combination_counter}/{total_combinations}: {sharpe_ratio}")

                results.append({
                    'combination_number': combination_counter,
                    'pop_size': pop_size,
                    'gen_count': gen_count,
                    'initial_mutation_rate': initial_mutation_rate,
                    'alpha': alpha,
                    'sharpe_ratio': sharpe_ratio,
                    'best_portfolio_weights': best_portfolio.tolist()
                })

                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_combination = (pop_size, gen_count, initial_mutation_rate, alpha)
                    best_combination_number = combination_counter

                combination_counter += 1

# Konverter resultatlisten til en pandas DataFrame for enklere lagring og analyse
results_df = pd.DataFrame(results)

# Lagre resultatene til en CSV-fil uten å inkludere DataFrame-indeksen
results_df.to_csv(results_file, index=False)

# Output the best results
print("\nBest Portfolio Weights:", best_portfolio)
print("\nBest Sharpe Ratio:", best_sharpe)
print("Best Parameters:", best_combination)
print("Best Combination Number:", best_combination_number)


best_portfolio_df = pd.DataFrame([best_portfolio], columns=returns_df.columns)
best_portfolio_df.to_csv(os.path.join(script_dir, '../3.prob2_output/3.6aes_best_portfolio.csv'), index=False)
print(f"\nBeste portefølje lagret i '3.6aes_best_portfolio.csv'")
