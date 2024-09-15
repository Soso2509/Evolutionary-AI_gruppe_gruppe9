import os
import numpy as np
import pandas as pd

# Finn stien til dette skriptet (der filen kjører fra)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Definer stier for input- og output-filer basert på skriptets plassering
returns_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')
cov_matrix_file = os.path.join(script_dir, '../3.prob2_output/3.2beregn_kovarians_matrix.csv')
results_file = os.path.join(script_dir, '../3.prob2_output/3.7m+l.csv')

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


# Funksjon for å velge top μ porteføljer basert på deres Sharpe ratio
def select_best_individuals(population, expected_returns, cov_matrix, risk_free_rate, num_to_select):
    # Beregne fitness for hver portefølje
    fitness_scores = [fitness_function(weights, expected_returns, cov_matrix, risk_free_rate) for weights in population]
    # Sorter basert på fitness(høyeste Sharpe-ratio først)
    sorted_population = [weights for _, weights in sorted(zip(fitness_scores, population), reverse=True)]
    return sorted_population[:num_to_select]


def run_MuPlussLambda_algorithm(pop_size_mu, pop_size_lambda,num_generations, mutation_rate, risk_free_rate):
    population = generate_population(pop_size_mu, num_assets) # Genererer den initielle foreldrepopulasjonen (μ foreldre)
    best_sharpe_ratio = -np.inf  # Starter med laveste mulige Sharpe-ratio
    best_portfolio = None  # Beste portefølje initialisert som None

    for generation in range(num_generations):   # For hver generasjon i det angitte antallet generasjoner
        offspring = [mutate_portfolio(population[np.random.randint(0, pop_size_mu)], mutation_rate) for _ in range(pop_size_lambda)] # Genererer λ barn ved å mutere foreldrepopulasjonen

        combined_population = np.vstack((population, offspring)) # Kombiner foreldre og barn til en populasjon

        population = select_best_individuals(combined_population, expected_returns, cov_matrix, risk_free_rate, pop_size_mu) # Velg topp μ porteføljer basert på fitness

        # Hent den beste Sharpe-ratio og tilhørende portefølje for denne generasjonen
        generation_best_sharpe_ratio = fitness_function(population[0], expected_returns, cov_matrix, risk_free_rate)
        generation_best_portfolio = population[0]

        # Oppdater den beste Sharpe-ratio og portefølje hvis den beste i denne generasjonen er bedre
        if generation_best_sharpe_ratio > best_sharpe_ratio:
            best_sharpe_ratio = generation_best_sharpe_ratio
            best_portfolio = generation_best_portfolio

        generation_best_expected_return, generation_best_portfolio_stddev = portfolio_performance(generation_best_portfolio, expected_returns, cov_matrix)  # Beregn porteføljens forventede avkastning og standardavvik for den beste porteføljen i denne generasjonen
        # Skriv ut resultater for den nåværende generasjonen
        print(f"Population Size μ: {pop_size_mu}, Population Size λ: {pop_size_lambda}, Generations: {num_generations}, Mutation Rate: {mutation_rate}")
        print(f"Generation {generation + 1}: Best Sharpe Ratio = {generation_best_sharpe_ratio:.4f}, "
              f"Expected Return = {generation_best_expected_return:.4f}, "
              f"Portfolio Std Dev = {generation_best_portfolio_stddev:.4f}")
    return best_portfolio, best_sharpe_ratio



# Parametere å teste
population_sizes_mu = [20, 50, 100]
population_sizes_lambda = [10, 100, 200]
generation_counts = [50, 100, 200]
mutation_rates = [0.01, 0.05, 0.1]
risk_free_rate = 0.02 / 12

# Inisialiserer "best" variabler som vil fylles med de beste parameterne underveis
best_sharpe = -np.inf
best_combination = None
best_combination_number = -1
combination_counter = 1
total_combinations = len(population_sizes_mu) * len(population_sizes_lambda) * len(generation_counts) * len(mutation_rates)
results = []


# Kjører gjennom ES slik at alle kombinasjonene av "parametere å teste" kan få sin gjennomgang. Tre for loops er alt som trengs :|
for pop_size_mu in population_sizes_mu:
    for pop_size_lambda in population_sizes_lambda:
        for gen_count in generation_counts:
            for mut_rate in mutation_rates:
                print(
                    f"Kjører kombinasjon {combination_counter}/{total_combinations}: pop_size_mu={pop_size_mu}, pop_size_lambda={pop_size_lambda} , gen_count={gen_count}, mut_rate={mut_rate}")

                # Kjører ES funksjonen med de nåværende parameterne

                best_portfolio, sharpe_ratio = run_MuPlussLambda_algorithm(pop_size_mu, pop_size_lambda,gen_count, mut_rate, risk_free_rate)
                print(f"Sharpe-ratio for kombinasjon {combination_counter}/{total_combinations}: {sharpe_ratio}")

                # Lagrer alle verdiene for hver iterasjon i "results" slik at det senere kan accesses
                results.append({
                    'combination_number': combination_counter,
                    'pop_size_mu': pop_size_mu,
                    'pop_size_lambda': pop_size_lambda,
                    'gen_count': gen_count,
                    'mut_rate': mut_rate,
                    'sharpe_ratio': sharpe_ratio,
                    'best_portfolio_weights': best_portfolio.tolist()

                })

                # Om nåværende iterasjon har bedre sharpe ratio enn det som er registrert atm så skal kombinasjonen av test parameter og kombinasjons nr. oppdateres til det nåværende
                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_combination = (pop_size_mu, pop_size_lambda, gen_count, mut_rate)
                    best_combination_number = combination_counter

                # NEXT combination! Back to the top of the loop now
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
best_portfolio_df.to_csv(os.path.join(script_dir, '../3.prob2_output/3.7m+l_best_portfolio.csv'), index=False)
print(f"\nBeste portefølje lagret i '3.7m+l_best_portfolio.csv'")