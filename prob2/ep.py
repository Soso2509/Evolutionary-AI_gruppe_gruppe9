import numpy as np
import pandas as pd

# Last inn dataene fra CSV-filen med månedlige avkastninger
returns_df = pd.read_csv('mnd_avk_aksjer.csv', index_col='Date', parse_dates=True)

# Beregn forventet (gjennomsnittlig) avkastning for hver aksje
expected_returns = returns_df.mean().values  # Gjennomsnittlig månedlig avkastning for hver aksje
num_assets = len(expected_returns)  # Antall aksjer i porteføljen

# Beregn kovariansmatrisen for aksjene
cov_matrix = returns_df.cov().values

# Definer risikofri rente (f.eks. 2% årlig, 0.02)
risk_free_rate = 0.02 / 12  # For månedlig risikofri rente

# Funksjon for å beregne porteføljens forventede avkastning og standardavvik (risiko)
def portfolio_performance(weights, expected_returns, cov_matrix):
    # Forventet avkastning
    expected_return = np.dot(weights, expected_returns)
    # Standardavvik (risiko)
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
    population = []
    for _ in range(pop_size):
        population.append(generate_portfolio(num_assets))
    return np.array(population)

# Funksjon for å mutere porteføljer
def mutate_portfolio(portfolio, mutation_rate=0.1):
    mutation = np.random.normal(0, mutation_rate, len(portfolio))
    mutated_portfolio = portfolio + mutation
    mutated_portfolio = np.clip(mutated_portfolio, 0, 1)  # Sikrer at ingen vekt er negativ
    mutated_portfolio /= np.sum(mutated_portfolio)  # Sikrer at vektene fortsatt summerer til 1
    return mutated_portfolio

# Funksjon for å velge de beste porteføljene
def select_best(population, fitness_scores, num_to_select):
    selected_indices = np.argsort(fitness_scores)[-num_to_select:]  # Velger de med høyest fitness
    return population[selected_indices]

# Evolutionary Programming Algoritme
def evolutionary_programming(expected_returns, cov_matrix, population_size, num_generations, risk_free_rate, mutation_rate=0.1):
    num_assets = len(expected_returns)
    
    # Generer startpopulasjon
    population = generate_population(population_size, num_assets)
    
    for generation in range(num_generations):
        # Beregn fitness for hver portefølje
        fitness_scores = np.array([fitness_function(p, expected_returns, cov_matrix, risk_free_rate) for p in population])
        
        # Velg de beste porteføljene for neste generasjon
        num_to_select = population_size // 2
        best_portfolios = select_best(population, fitness_scores, num_to_select)
        
        # Muter de valgte porteføljene og generer nye porteføljer
        next_generation = []
        for portfolio in best_portfolios:
            next_generation.append(portfolio)  # Behold de beste
            next_generation.append(mutate_portfolio(portfolio, mutation_rate))  # Muter for nye løsninger
        
        # Oppdater populasjon
        population = np.array(next_generation)
    
    # Returner den beste porteføljen etter siste generasjon
    final_fitness_scores = np.array([fitness_function(p, expected_returns, cov_matrix, risk_free_rate) for p in population])
    best_portfolio = population[np.argmax(final_fitness_scores)]
    
    return best_portfolio, np.max(final_fitness_scores)

# Parameterområder for testing
population_sizes = [50, 100, 150, 200, 250, 300]  # Test med forskjellig populasjon 
generation_counts = [50, 100, 150, 200, 250, 300]  # Test med forskjellige generasjoner
mutation_rates = [0.01, 0.05, 0.1, 0.15, 0.20]  # Test med forskjellige mutasjonsrate

# Variabel for å holde styr på den beste kombinasjonen
best_sharpe = -np.inf
best_combination = None

# Test alle kombinasjoner
for pop_size in population_sizes:
    for gen_count in generation_counts:
        for mut_rate in mutation_rates:
            print(f"Testing: pop_size={pop_size}, gen_count={gen_count}, mut_rate={mut_rate}")
            
            # Kjør algoritmen med den nåværende kombinasjonen av parametere
            best_portfolio, sharpe_ratio = evolutionary_programming(expected_returns, cov_matrix, pop_size, gen_count, risk_free_rate, mut_rate)
            
            print(f"Sharpe-ratio: {sharpe_ratio}")
            
            # Lagre den beste kombinasjonen
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_combination = (pop_size, gen_count, mut_rate)

# Resultat
print("\nBeste kombinasjon funnet:")
print(f"Populasjonsstørrelse: {best_combination[0]}, Antall generasjoner: {best_combination[1]}, Mutasjonsrate: {best_combination[2]}")
print(f"Beste Sharpe-ratio: {best_sharpe}")
