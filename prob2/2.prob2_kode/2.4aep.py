import os
import numpy as np
import pandas as pd

# Valgfritt: Sett en tilfeldig frø for reproduserbarhet
np.random.seed(42)

# Finn stien til dette skriptet
script_dir = os.path.dirname(os.path.abspath(__file__))

# Definer stier for input- og output-filer basert på skriptets plassering
returns_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')
cov_matrix_file = os.path.join(script_dir, '../3.prob2_output/3.2beregn_kovarians_matrix.csv')
results_file = os.path.join(script_dir, '../3.prob2_output/3.4aep.csv')

# Last inn dataene fra CSV-filen med månedlige avkastninger
returns_df = pd.read_csv(returns_file, index_col='Date', parse_dates=True)

# Beregn forventet (gjennomsnittlig) avkastning for hver aksje
expected_returns = returns_df.mean().values  # Gjennomsnittlig månedlig avkastning for hver aksje
num_assets = len(expected_returns)  # Antall aksjer i porteføljen

# Last inn kovariansmatrisen fra CSV-filen du allerede har generert
cov_matrix = pd.read_csv(cov_matrix_file, index_col=0).values

# Definer risikofri rente (f.eks. 2% årlig, 0.02)
risk_free_rate = 0.02 / 12  # For månedlig risikofri rente

# Funksjon for å beregne porteføljens forventede avkastning og standardavvik (risiko)
def portfolio_performance(weights, expected_returns, cov_matrix):
    expected_return = np.dot(weights, expected_returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_stddev = np.sqrt(portfolio_variance)
    return expected_return, portfolio_stddev

# Fitness-funksjon: Beregn Sharpe-ratio
def fitness_function(weights, expected_returns, cov_matrix, risk_free_rate):
    expected_return, portfolio_stddev = portfolio_performance(weights, expected_returns, cov_matrix)
    if portfolio_stddev == 0:
        return -np.inf  # Unngå divisjon med null
    sharpe_ratio = (expected_return - risk_free_rate) / portfolio_stddev
    return sharpe_ratio

# Modifiser individrepresentasjonen til å inkludere mutasjonsstørrelse (sigma)
def generate_portfolio(num_assets):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    sigma = np.random.uniform(0.05, 0.2, num_assets)  # Initiale mutasjonsstørrelser
    return {'weights': weights, 'sigma': sigma}

# Generer en populasjon av porteføljer
def generate_population(pop_size, num_assets):
    return [generate_portfolio(num_assets) for _ in range(pop_size)]

# Selv-adaptiv mutasjonsfunksjon
def mutate_portfolio(individual):
    weights = individual['weights']
    sigma = individual['sigma']
    num_assets = len(weights)
    
    # Mutasjonsparametere
    tau = 1 / np.sqrt(2 * np.sqrt(num_assets))
    tau_prime = 1 / np.sqrt(2 * num_assets)
    
    # Oppdater sigma (mutasjonsstørrelser)
    sigma_prime = sigma * np.exp(tau_prime * np.random.normal() + tau * np.random.normal(size=num_assets))
    sigma_prime = np.clip(sigma_prime, 1e-6, 1)  # Sikre at sigma holder seg innenfor grenser
    
    # Muter vekter
    weights_prime = weights + sigma_prime * np.random.normal(size=num_assets)
    weights_prime = np.clip(weights_prime, 0, 1)
    if np.sum(weights_prime) == 0:
        weights_prime = generate_portfolio(num_assets)['weights']  # Re-initialiser hvis alle vekter er null
    else:
        weights_prime /= np.sum(weights_prime)
    
    return {'weights': weights_prime, 'sigma': sigma_prime}

# Turneringseleksjonsfunksjon
def tournament_selection(population, fitness_scores, tournament_size):
    selected = []
    pop_size = len(population)
    for _ in range(pop_size):
        # Velg individer for turneringen tilfeldig
        participants = np.random.choice(pop_size, tournament_size, replace=False)
        participant_fitness = fitness_scores[participants]
        winner_idx = participants[np.argmax(participant_fitness)]
        selected.append(population[winner_idx])
    return selected

# Elitisme-funksjon
def get_elites(population, fitness_scores, num_elites):
    elite_indices = np.argsort(fitness_scores)[-num_elites:]
    elites = [population[i] for i in elite_indices]
    return elites

# Avansert Evolutionary Programming-algoritme
def advanced_evolutionary_programming(expected_returns, cov_matrix, population_size, num_generations, risk_free_rate, tournament_size=3, num_elites=1):
    num_assets = len(expected_returns)
    # Generer startpopulasjon
    population = generate_population(population_size, num_assets)
    
    for generation in range(num_generations):
        # Evaluer fitness
        fitness_scores = np.array([fitness_function(ind['weights'], expected_returns, cov_matrix, risk_free_rate) for ind in population])
        
        # Elitisme: Hent eliteindivider
        elites = get_elites(population, fitness_scores, num_elites)
        
        # Seleksjon: Turneringseleksjon
        selected_parents = tournament_selection(population, fitness_scores, tournament_size)
        
        # Mutasjon: Opprett avkom
        offspring = [mutate_portfolio(parent) for parent in selected_parents]
        
        # Dann ny populasjon
        population = elites + offspring[:population_size - num_elites]
    
    # Endelig evaluering
    final_fitness_scores = np.array([fitness_function(ind['weights'], expected_returns, cov_matrix, risk_free_rate) for ind in population])
    best_idx = np.argmax(final_fitness_scores)
    best_individual = population[best_idx]
    
    return best_individual['weights'], final_fitness_scores[best_idx]

# Hovedfunksjon for å kjøre den avanserte EP-algoritmen
def run_advanced_ep():
    # Parameterområder for testing
    population_sizes = [100, 200, 300]  # Juster etter behov
    generation_counts = [100, 200, 300]
    tournament_sizes = [2, 3, 5]
    num_elites_list = [1, 2, 5]
    
    # Beregn totalt antall kombinasjoner
    total_combinations = len(population_sizes) * len(generation_counts) * len(tournament_sizes) * len(num_elites_list)
    print(f"Totalt antall kombinasjoner å teste: {total_combinations}")
    
    combination_counter = 1
    best_sharpe = -np.inf
    best_combination = None
    best_portfolio_overall = None
    results = []
    
    for pop_size in population_sizes:
        for gen_count in generation_counts:
            for tour_size in tournament_sizes:
                for num_elites in num_elites_list:
                    print(f"Kjører kombinasjon {combination_counter}/{total_combinations}: populasjonsstørrelse={pop_size}, generasjoner={gen_count}, turneringsstørrelse={tour_size}, antall eliter={num_elites}")
                    
                    best_portfolio, sharpe_ratio = advanced_evolutionary_programming(
                        expected_returns, cov_matrix, pop_size, gen_count, risk_free_rate,
                        tournament_size=tour_size, num_elites=num_elites)
                    
                    print(f"Sharpe-ratio for kombinasjon {combination_counter}/{total_combinations}: {sharpe_ratio}")
                    
                    results.append({
                        'kombinasjonsnummer': combination_counter,
                        'populasjonsstørrelse': pop_size,
                        'generasjoner': gen_count,
                        'turneringsstørrelse': tour_size,
                        'antall_eliter': num_elites,
                        'sharpe_ratio': sharpe_ratio
                    })
                    
                    if sharpe_ratio > best_sharpe:
                        best_sharpe = sharpe_ratio
                        best_combination = (pop_size, gen_count, tour_size, num_elites)
                        best_portfolio_overall = best_portfolio
                        best_combination_number = combination_counter
                    
                    combination_counter += 1
    
    # Konverter resultater til en DataFrame
    results_df = pd.DataFrame(results)
    
    # Lagre resultater til en CSV-fil
    results_df.to_csv(results_file, index=False)
    
    # Skriv ut beste kombinasjon
    print("\nBeste kombinasjon funnet:")
    print(f"Kombinasjonsnummer: {best_combination_number}/{total_combinations}")
    print(f"Sharpe-ratio: {best_sharpe}")
    print(f"Populasjonsstørrelse: {best_combination[0]}, Generasjoner: {best_combination[1]}, Turneringsstørrelse: {best_combination[2]}, Antall eliter: {best_combination[3]}")
    print(f"Beste porteføljevekter:\n{best_portfolio_overall}")
    
    # Lagre den beste porteføljen til en CSV-fil
    best_portfolio_df = pd.DataFrame([best_portfolio_overall], columns=returns_df.columns)
    best_portfolio_df.to_csv(os.path.join(script_dir, '../3.prob2_output/3.4aep_best_portfolio.csv'), index=False)
    print(f"Beste portefølje lagret i '3.4aep_best_portfolio.csv'")

# Kjør den avanserte EP-algoritmen
if __name__ == "__main__":
    run_advanced_ep()
