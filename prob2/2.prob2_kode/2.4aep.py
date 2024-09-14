import os           # Bibliotek for fil- og katalogoperasjoner
import numpy as np  # Numerisk bibliotek for matematiske operasjoner og arrays
import pandas as pd # Dataanalysebibliotek for håndtering av data i tabellform

# Valgfritt: Sett en tilfeldig frø for reproduserbarhet av resultater
np.random.seed(42)

# Finn stien til dette skriptet (katalogen hvor denne filen kjører fra)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Definer stier for input- og output-filer basert på skriptets plassering
# Input-filer: månedlige avkastninger og kovariansmatrise
# Output-fil: resultater fra algoritmen
returns_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')  # Fil med månedlige avkastninger
cov_matrix_file = os.path.join(script_dir, '../3.prob2_output/3.2beregn_kovarians_matrix.csv')  # Fil med kovariansmatrise
results_file = os.path.join(script_dir, '../3.prob2_output/3.4aep.csv')  # Fil for å lagre resultatene

# Last inn dataene fra CSV-filen med månedlige avkastninger til en pandas DataFrame
# Bruk 'Date'-kolonnen som indeks og konverter den til datetime-objekter
returns_df = pd.read_csv(returns_file, index_col='Date', parse_dates=True)

# Beregn forventet (gjennomsnittlig) avkastning for hver aksje i porteføljen
expected_returns = returns_df.mean().values  # Hent gjennomsnittlig månedlig avkastning som en numpy array
num_assets = len(expected_returns)  # Antall aksjer i porteføljen

# Last inn kovariansmatrisen fra CSV-filen og konverter den til en numpy array
cov_matrix = pd.read_csv(cov_matrix_file, index_col=0).values

# Definer risikofri rente (for eksempel 2% årlig)
# Konverter den årlige risikofrie renten til månedlig ved å dele på 12
risk_free_rate = 0.02 / 12  # Månedlig risikofri rente

# Funksjon for å beregne porteføljens forventede avkastning og risiko (standardavvik)
def portfolio_performance(weights, expected_returns, cov_matrix):
    # Beregn porteføljens forventede avkastning ved å ta det veide gjennomsnittet av avkastningene
    expected_return = np.dot(weights, expected_returns)
    # Beregn porteføljens varians (risiko) ved å bruke kovariansmatrisen og porteføljevektene
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    # Beregn standardavviket (kvadratroten av variansen)
    portfolio_stddev = np.sqrt(portfolio_variance)
    # Returner forventet avkastning og risiko
    return expected_return, portfolio_stddev

# Fitness-funksjon: Beregn Sharpe-ratio for en gitt portefølje
def fitness_function(weights, expected_returns, cov_matrix, risk_free_rate):
    # Beregn porteføljens forventede avkastning og risiko
    expected_return, portfolio_stddev = portfolio_performance(weights, expected_returns, cov_matrix)
    # Unngå divisjon med null hvis standardavviket er null
    if portfolio_stddev == 0:
        return -np.inf  # Returner negativ uendelig for å indikere dårlig fitness
    # Beregn Sharpe-ratio: (forventet avkastning - risikofri rente) / risiko
    sharpe_ratio = (expected_return - risk_free_rate) / portfolio_stddev
    # Returner Sharpe-ratioen som fitness-verdi
    return sharpe_ratio

# Modifiser individrepresentasjonen til å inkludere mutasjonsstørrelse (sigma)
def generate_portfolio(num_assets):
    # Generer en array med tilfeldige vekter
    weights = np.random.random(num_assets)
    # Normaliser vektene slik at de summerer til 1 (full investert portefølje)
    weights /= np.sum(weights)
    # Initialiser mutasjonsstørrelser (sigma) for hver vekt med verdier mellom 0.05 og 0.2
    sigma = np.random.uniform(0.05, 0.2, num_assets)  # Initiale mutasjonsstørrelser
    # Returner individet som en dictionary med 'weights' og 'sigma'
    return {'weights': weights, 'sigma': sigma}

# Generer en populasjon av porteføljer
def generate_population(pop_size, num_assets):
    # Opprett en liste med 'pop_size' antall porteføljer
    return [generate_portfolio(num_assets) for _ in range(pop_size)]

# Selv-adaptiv mutasjonsfunksjon
def mutate_portfolio(individual):
    weights = individual['weights']
    sigma = individual['sigma']
    num_assets = len(weights)
    
    # Mutasjonsparametere for selv-adaptiv mutasjon
    tau = 1 / np.sqrt(2 * np.sqrt(num_assets))  # Global læringsrate
    tau_prime = 1 / np.sqrt(2 * num_assets)     # Individuell læringsrate
    
    # Oppdater sigma (mutasjonsstørrelser)
    sigma_prime = sigma * np.exp(
        tau_prime * np.random.normal() +       # Global komponent
        tau * np.random.normal(size=num_assets)  # Individuell komponent
    )
    # Sikre at sigma holder seg innenfor grenser
    sigma_prime = np.clip(sigma_prime, 1e-6, 1)  # Unngå null eller negative verdier
    
    # Muter vekter
    weights_prime = weights + sigma_prime * np.random.normal(size=num_assets)
    # Klipp vektene til å være mellom 0 og 1 (ingen negative vekter)
    weights_prime = np.clip(weights_prime, 0, 1)
    # Unngå at alle vekter er null ved å re-initialisere om nødvendig
    if np.sum(weights_prime) == 0:
        weights_prime = generate_portfolio(num_assets)['weights']  # Re-initialiser
    else:
        # Normaliser vektene slik at de summerer til 1
        weights_prime /= np.sum(weights_prime)
    
    # Returner det muterte individet med oppdaterte vekter og sigma
    return {'weights': weights_prime, 'sigma': sigma_prime}

# Turneringseleksjonsfunksjon
def tournament_selection(population, fitness_scores, tournament_size):
    selected = []
    pop_size = len(population)
    # For hver posisjon i populasjonen
    for _ in range(pop_size):
        # Velg individer for turneringen tilfeldig
        participants = np.random.choice(pop_size, tournament_size, replace=False)
        # Hent fitness-score for deltakerne
        participant_fitness = fitness_scores[participants]
        # Finn indeksen til deltakeren med høyest fitness
        winner_idx = participants[np.argmax(participant_fitness)]
        # Legg til vinneren i listen over valgte individer
        selected.append(population[winner_idx])
    # Returner listen over valgte foreldre
    return selected

# Elitisme-funksjon
def get_elites(population, fitness_scores, num_elites):
    # Sorter indeksene basert på fitness-score i stigende rekkefølge
    elite_indices = np.argsort(fitness_scores)[-num_elites:]  # Velg de beste individene
    # Hent de beste individene fra populasjonen
    elites = [population[i] for i in elite_indices]
    # Returner listen over eliteindivider
    return elites

# Avansert Evolutionary Programming-algoritme
def advanced_evolutionary_programming(
    expected_returns, cov_matrix, population_size, num_generations,
    risk_free_rate, tournament_size=3, num_elites=1
):
    num_assets = len(expected_returns)
    # Generer startpopulasjon
    population = generate_population(population_size, num_assets)
    
    # Start evolusjonen over antall generasjoner
    for generation in range(num_generations):
        # Evaluer fitness for hver portefølje
        fitness_scores = np.array([
            fitness_function(ind['weights'], expected_returns, cov_matrix, risk_free_rate)
            for ind in population
        ])
        
        # Elitisme: Hent eliteindivider
        elites = get_elites(population, fitness_scores, num_elites)
        
        # Seleksjon: Turneringseleksjon
        selected_parents = tournament_selection(population, fitness_scores, tournament_size)
        
        # Mutasjon: Opprett avkom
        offspring = [mutate_portfolio(parent) for parent in selected_parents]
        
        # Dann ny populasjon ved å kombinere eliter og avkom
        population = elites + offspring[:population_size - num_elites]
    
    # Endelig evaluering etter siste generasjon
    final_fitness_scores = np.array([
        fitness_function(ind['weights'], expected_returns, cov_matrix, risk_free_rate)
        for ind in population
    ])
    # Finn indeksen til individet med høyest fitness
    best_idx = np.argmax(final_fitness_scores)
    best_individual = population[best_idx]
    
    # Returner den beste porteføljens vekter og Sharpe-ratio
    return best_individual['weights'], final_fitness_scores[best_idx]

# Hovedfunksjon for å kjøre den avanserte EP-algoritmen
def run_advanced_ep():
    # Parameterområder for testing
    population_sizes = [100, 200, 300]     # Populasjonsstørrelser å teste
    generation_counts = [100, 200, 300]    # Antall generasjoner å teste
    tournament_sizes = [2, 3, 5]           # Turneringsstørrelser å teste
    num_elites_list = [1, 2, 5]            # Antall eliter å teste
    
    # Beregn totalt antall kombinasjoner
    total_combinations = (
        len(population_sizes) *
        len(generation_counts) *
        len(tournament_sizes) *
        len(num_elites_list)
    )
    print(f"Totalt antall kombinasjoner å teste: {total_combinations}")
    
    combination_counter = 1  # Teller for å holde styr på hvilken kombinasjon vi er på
    best_sharpe = -np.inf    # Variabel for å holde styr på den beste Sharpe-ratioen
    best_combination = None  # Variabel for å lagre den beste kombinasjonen av parametere
    best_portfolio_overall = None  # Variabel for å lagre den beste porteføljen
    best_combination_number = None  # Variabel for å holde styr på det beste kombinasjonsnummeret
    results = []  # Liste for å samle inn alle resultater
    
    # Test alle kombinasjoner
    for pop_size in population_sizes:
        for gen_count in generation_counts:
            for tour_size in tournament_sizes:
                for num_elites in num_elites_list:
                    print(f"Kjører kombinasjon {combination_counter}/{total_combinations}: "
                          f"populasjonsstørrelse={pop_size}, generasjoner={gen_count}, "
                          f"turneringsstørrelse={tour_size}, antall eliter={num_elites}")
                    
                    # Kjør algoritmen med den nåværende kombinasjonen av parametere
                    best_portfolio, sharpe_ratio = advanced_evolutionary_programming(
                        expected_returns, cov_matrix, pop_size, gen_count, risk_free_rate,
                        tournament_size=tour_size, num_elites=num_elites
                    )
                    
                    print(f"Sharpe-ratio for kombinasjon {combination_counter}/{total_combinations}: {sharpe_ratio}")
                    
                    # Lagre resultatene i listen
                    results.append({
                        'kombinasjonsnummer': combination_counter,
                        'populasjonsstørrelse': pop_size,
                        'generasjoner': gen_count,
                        'turneringsstørrelse': tour_size,
                        'antall_eliter': num_elites,
                        'sharpe_ratio': sharpe_ratio
                    })
                    
                    # Lagre den beste kombinasjonen
                    if sharpe_ratio > best_sharpe:
                        best_sharpe = sharpe_ratio
                        best_combination = (pop_size, gen_count, tour_size, num_elites)
                        best_portfolio_overall = best_portfolio
                        best_combination_number = combination_counter
                        
                    # Oppdater kombinasjonsnummeret
                    combination_counter += 1
    
    # Konverter resultatene til en DataFrame
    results_df = pd.DataFrame(results)
    
    # Lagre resultatene til en CSV-fil
    results_df.to_csv(results_file, index=False)
    
    # Skriv ut den beste kombinasjonen funnet
    print("\nBeste kombinasjon funnet:")
    print(f"Kombinasjonsnummer: {best_combination_number}/{total_combinations}")
    print(f"Sharpe-ratio: {best_sharpe}")
    print(f"Populasjonsstørrelse: {best_combination[0]}, Generasjoner: {best_combination[1]}, "
          f"Turneringsstørrelse: {best_combination[2]}, Antall eliter: {best_combination[3]}")
    print(f"Beste porteføljevekter:\n{best_portfolio_overall}")
    
    # Lagre den beste porteføljen til en CSV-fil
    best_portfolio_df = pd.DataFrame([best_portfolio_overall], columns=returns_df.columns)
    best_portfolio_df.to_csv(os.path.join(script_dir, '../3.prob2_output/3.4aep_best_portfolio.csv'), index=False)
    print(f"Beste portefølje lagret i '3.4aep_best_portfolio.csv'")

# Kjør den avanserte EP-algoritmen
if __name__ == "__main__":
    run_advanced_ep()
