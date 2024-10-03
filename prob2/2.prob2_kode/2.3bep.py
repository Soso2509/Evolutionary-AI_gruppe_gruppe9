import os          # Bibliotek for fil- og katalogoperasjoner
import numpy as np  # Numerisk bibliotek for matematiske operasjoner og matriser
import pandas as pd # Dataanalysebibliotek for håndtering av tabulære data

# Finn den absolutte stien til katalogen der skriptet er plassert
script_dir = os.path.dirname(os.path.abspath(__file__))

# Definer stier for inndata- og utdatafiler basert på skriptets plassering
returns_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')  # Fil med beregnede månedlige avkastninger
cov_matrix_file = os.path.join(script_dir, '../3.prob2_output/3.2beregn_kovarians_matrix.csv')  # Fil med kovariansmatrisen
results_file = os.path.join(script_dir, '../3.prob2_output/3.3bep.csv')  # Fil for å lagre resultater

# Les de månedlige avkastningene fra CSV-filen inn i en pandas DataFrame
returns_df = pd.read_csv(returns_file, index_col='Date', parse_dates=True)

# Beregn forventet (gjennomsnittlig) avkastning for hver aksje i porteføljen
expected_returns = returns_df.mean().values  # Få gjennomsnittlig månedlig avkastning for hver aksje
num_assets = len(expected_returns)  # Antall aksjer i porteføljen

# Les kovariansmatrisen fra CSV-filen og konverter den til en NumPy-matrise
cov_matrix = pd.read_csv(cov_matrix_file, index_col=0).values

# Definer den risikofrie renten (f.eks. 2% årlig), konvertert til månedlig
risk_free_rate = 0.02 / 12  # Månedlig risikofri rente

# Funksjon for å beregne forventet avkastning og risiko (standardavvik) for en portefølje
def portfolio_performance(weights, expected_returns, cov_matrix):
    expected_return = np.dot(weights, expected_returns)  # Forventet avkastning
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))  # Variansen til porteføljen
    portfolio_stddev = np.sqrt(portfolio_variance)  # Standardavvik (risiko)
    return expected_return, portfolio_stddev

# Fitness-funksjon: Beregn Sharpe-ratioen for en gitt portefølje
def fitness_function(weights, expected_returns, cov_matrix, risk_free_rate):
    expected_return, portfolio_stddev = portfolio_performance(weights, expected_returns, cov_matrix)
    sharpe_ratio = (expected_return - risk_free_rate) / portfolio_stddev  # Sharpe-ratio
    return sharpe_ratio

# Funksjon for å generere en tilfeldig portefølje med vekter som summerer til 1
def generate_portfolio(num_assets):
    weights = np.random.random(num_assets)  # Generer tilfeldige vekter
    weights /= np.sum(weights)  # Normaliser vektene slik at de summerer til 1
    return weights

# Funksjon for å generere en populasjon av tilfeldige porteføljer
def generate_population(pop_size, num_assets):
    population = [generate_portfolio(num_assets) for _ in range(pop_size)]  # Generer populasjon
    return np.array(population)

# Funksjon for å mutere en portefølje for å introdusere variasjon
def mutate_portfolio(portfolio, mutation_rate=0.1):
    mutation = np.random.normal(0, mutation_rate, len(portfolio))  # Generer mutasjon
    mutated_portfolio = portfolio + mutation  # Legg til mutasjonen
    mutated_portfolio = np.clip(mutated_portfolio, 0, 1)  # Begrens vektene til mellom 0 og 1
    mutated_portfolio /= np.sum(mutated_portfolio)  # Normaliser vektene igjen
    return mutated_portfolio

# Funksjon for å velge de beste porteføljene basert på deres fitness (Sharpe-ratio)
def select_best(population, fitness_scores, num_to_select):
    selected_indices = np.argsort(fitness_scores)[-num_to_select:]  # Velg de beste porteføljene
    return population[selected_indices]

# Evolutionary Programming Algorithm for porteføljeoptimalisering
def evolutionary_programming(expected_returns, cov_matrix, population_size, num_generations, risk_free_rate, mutation_rate=0.1):
    num_assets = len(expected_returns)  # Antall aksjer i porteføljen
    population = generate_population(population_size, num_assets)  # Generer initial populasjon

    generation_results = []  # Liste for å lagre resultater for hver generasjon

    # Start evolusjonsprosessen
    for generation in range(num_generations):
        fitness_scores = np.array([fitness_function(p, expected_returns, cov_matrix, risk_free_rate) for p in population])
        
        # Finn den beste porteføljen i den nåværende generasjonen
        best_index = np.argmax(fitness_scores)
        best_sharpe_ratio = fitness_scores[best_index]
        best_portfolio = population[best_index]
        
        # Lagre resultatet for denne generasjonen
        generation_results.append({
            'generation': generation + 1,  # Lagre generasjonsnummeret
            'combination_number': combination_counter,
            'pop_size': population_size,
            'gen_count': num_generations,
            'mut_rate': mutation_rate,
            'sharpe_ratio': best_sharpe_ratio
        })
        
        # Velg de beste porteføljene basert på fitness (Sharpe-ratio)
        num_to_select = population_size // 2  # Velg halvparten av populasjonen
        best_portfolios = select_best(population, fitness_scores, num_to_select)  # Velg de beste porteføljene
        
        # Lag den neste generasjonen
        next_generation = []
        for portfolio in best_portfolios:
            next_generation.append(portfolio)  # Behold den originale porteføljen
            next_generation.append(mutate_portfolio(portfolio, mutation_rate))  # Legg til mutert versjon
        population = np.array(next_generation)  # Oppdater populasjonen

    # Finn den beste porteføljen i den siste populasjonen
    final_fitness_scores = np.array([fitness_function(p, expected_returns, cov_matrix, risk_free_rate) for p in population])
    best_portfolio = population[np.argmax(final_fitness_scores)]
    best_sharpe_ratio = np.max(final_fitness_scores)

    return best_portfolio, best_sharpe_ratio, generation_results

# Definer parameterområder for testing av algoritmen
population_sizes = [150, 200, 300]  # Ulike populasjonsstørrelser
generation_counts = [200, 500, 1000]  # Ulike antall generasjoner
mutation_rates = [0.15, 0.20, 0.6]  # Ulike mutasjonsrater

# Beregn totalt antall kombinasjoner som skal testes
total_combinations = len(population_sizes) * len(generation_counts) * len(mutation_rates)
print(f"Total combinations to test: {total_combinations}")

# Initialiser variabler for å spore den beste kombinasjonen og Sharpe-ratioen
best_sharpe = -np.inf  # Start med negativ uendelig for å sikre at enhver positiv Sharpe-ratio er bedre
best_combination = None  # Spor parameterne for den beste kombinasjonen
best_combination_number = None  # Spor kombinasjonsnummeret for det beste resultatet

# Initialiser en liste for å samle alle resultater fra testene
results = []

# Start testen over alle parameterkombinasjoner
combination_counter = 1  # Teller for å holde styr på kombinasjonsnummeret

# Løkke gjennom alle kombinasjoner av populasjonsstørrelser, antall generasjoner og mutasjonsrater
for pop_size in population_sizes:
    for gen_count in generation_counts:
        for mut_rate in mutation_rates:
            print(f"Running combination {combination_counter}/{total_combinations}: pop_size={pop_size}, gen_count={gen_count}, mut_rate={mut_rate}")

            # Kjør den evolusjonære algoritmen med de nåværende parameterne
            best_portfolio, sharpe_ratio, generation_results = evolutionary_programming(expected_returns, cov_matrix, pop_size, gen_count, risk_free_rate, mut_rate)

            # Lagre resultatene for hver generasjon
            results.extend(generation_results)

            # Oppdater den beste Sharpe-ratioen og kombinasjonen hvis nødvendig
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_combination = (pop_size, gen_count, mut_rate)
                best_combination_number = combination_counter

            combination_counter += 1

# Konverter resultatene til en pandas DataFrame
results_df = pd.DataFrame(results)

# Lagre resultatene til en CSV-fil med det ønskede formatet
results_df.to_csv(results_file, index=False)

# Skriv ut den beste kombinasjonen
print("\nBest combination found")
print(f"Combination number: {best_combination_number}/{total_combinations}")
print(f"Sharpe ratio: {best_sharpe}")
print(f"Population size: {best_combination[0]}, Number of generations: {best_combination[1]}, Mutation rate: {best_combination[2]}")
print(f"Best portfolio weights: {best_portfolio}")

# Lagre den beste porteføljen til en CSV-fil
best_portfolio_df = pd.DataFrame([best_portfolio], columns=returns_df.columns)
best_portfolio_df.to_csv(os.path.join(script_dir, '../3.prob2_output/3.3bep_best_portfolio.csv'), index=False)

print(f"Best portfolio saved to '3.3bep_best_portfolio.csv'")
