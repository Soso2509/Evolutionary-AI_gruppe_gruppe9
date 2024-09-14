import os          # Bibliotek for fil- og katalogoperasjoner
import numpy as np # Numerisk bibliotek for matematiske operasjoner og arrays
import pandas as pd # Dataanalysebibliotek for håndtering av data i tabellform

# Finn den absolutte stien til dette skriptet (den katalogen hvor denne filen kjører fra)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Definer stier for input- og output-filer basert på skriptets plassering
# Input-filer: månedlige avkastninger og kovariansmatrise
# Output-fil: resultatene fra algoritmen
returns_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')  # Fil med månedlige avkastninger
cov_matrix_file = os.path.join(script_dir, '../3.prob2_output/3.2beregn_kovarians_matrix.csv')  # Fil med kovariansmatrise
results_file = os.path.join(script_dir, '../3.prob2_output/3.3bep.csv')  # Fil for å lagre resultatene

# Les inn de månedlige avkastningene fra CSV-filen til en pandas DataFrame
# Bruk 'Date'-kolonnen som indeks og konverter den til datetime-objekter
returns_df = pd.read_csv(returns_file, index_col='Date', parse_dates=True)

# Beregn forventet (gjennomsnittlig) avkastning for hver aksje i porteføljen
expected_returns = returns_df.mean().values  # Hent gjennomsnittlig månedlig avkastning for hver aksje
num_assets = len(expected_returns)  # Antall aksjer i porteføljen (lengden av expected_returns)

# Les inn kovariansmatrisen fra CSV-filen og konverter den til en numpy array
cov_matrix = pd.read_csv(cov_matrix_file, index_col=0).values  # index_col=0 for å bruke første kolonne som indeks

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
    # Beregn Sharpe-ratio: (forventet avkastning - risikofri rente) / risiko
    sharpe_ratio = (expected_return - risk_free_rate) / portfolio_stddev
    # Returner Sharpe-ratioen som fitness-verdi
    return sharpe_ratio

# Funksjon for å generere en tilfeldig portefølje med vekter som summerer til 1
def generate_portfolio(num_assets):
    # Generer en array med tilfeldige tall for vektene
    weights = np.random.random(num_assets)
    # Normaliser vektene slik at de summerer til 1 (full investert portefølje)
    weights /= np.sum(weights)
    # Returner vektene for porteføljen
    return weights

# Funksjon for å generere en populasjon av tilfeldige porteføljer
def generate_population(pop_size, num_assets):
    population = []
    # Generer 'pop_size' antall porteføljer
    for _ in range(pop_size):
        # Legg til en tilfeldig portefølje i populasjonen
        population.append(generate_portfolio(num_assets))
    # Konverter populasjonen til en numpy array og returner
    return np.array(population)

# Funksjon for å mutere en portefølje for å introdusere variasjon
def mutate_portfolio(portfolio, mutation_rate=0.1):
    # Generer en mutasjon ved å trekke fra en normalfordeling med standardavvik lik mutation_rate
    mutation = np.random.normal(0, mutation_rate, len(portfolio))
    # Legg til mutasjonen til den opprinnelige porteføljen
    mutated_portfolio = portfolio + mutation
    # Klipp vektene til å være mellom 0 og 1 (ingen negative vekter)
    mutated_portfolio = np.clip(mutated_portfolio, 0, 1)
    # Normaliser vektene slik at de summerer til 1
    mutated_portfolio /= np.sum(mutated_portfolio)
    # Returner den muterte porteføljen
    return mutated_portfolio

# Funksjon for å velge de beste porteføljene basert på deres fitness (Sharpe-ratio)
def select_best(population, fitness_scores, num_to_select):
    # Sorter indekser av porteføljene basert på fitness, i stigende rekkefølge
    selected_indices = np.argsort(fitness_scores)[-num_to_select:]  # Hent indeksene til de beste porteføljene
    # Returner de beste porteføljene
    return population[selected_indices]

# Evolutionary Programming Algoritme for porteføljeoptimalisering
def evolutionary_programming(expected_returns, cov_matrix, population_size, num_generations, risk_free_rate, mutation_rate=0.1):
    num_assets = len(expected_returns)
    
    # Generer startpopulasjon av porteføljer
    population = generate_population(population_size, num_assets)
    
    # Start evolusjonen over antall generasjoner
    for generation in range(num_generations):
        # Beregn fitness (Sharpe-ratio) for hver portefølje i populasjonen
        fitness_scores = np.array([fitness_function(p, expected_returns, cov_matrix, risk_free_rate) for p in population])
        
        # Velg de beste porteføljene for å overføre til neste generasjon
        num_to_select = population_size // 2  # Behold halvparten
        best_portfolios = select_best(population, fitness_scores, num_to_select)
        
        # Muter de beste porteføljene for å skape variasjon
        next_generation = []
        for portfolio in best_portfolios:
            next_generation.append(portfolio)  # Behold originalen
            next_generation.append(mutate_portfolio(portfolio, mutation_rate))  # Legg til mutert versjon
        
        # Oppdater populasjonen med neste generasjon
        population = np.array(next_generation)
    
    # Etter alle generasjoner, finn den beste porteføljen i den siste populasjonen
    final_fitness_scores = np.array([fitness_function(p, expected_returns, cov_matrix, risk_free_rate) for p in population])
    best_portfolio = population[np.argmax(final_fitness_scores)]
    
    # Returner den beste porteføljen og dens Sharpe-ratio
    return best_portfolio, np.max(final_fitness_scores)

# Definer parameterområder for testing av algoritmen
population_sizes = [50, 100, 150, 200, 250, 300]  # Ulike populasjonsstørrelser
generation_counts = [50, 100, 150, 200, 250, 300]  # Ulike antall generasjoner
mutation_rates = [0.01, 0.05, 0.1, 0.15, 0.20]  # Ulike mutasjonsrater

# Beregn totalt antall kombinasjoner av parametere som skal testes
total_combinations = len(population_sizes) * len(generation_counts) * len(mutation_rates)

# Skriv ut totalt antall kombinasjoner til konsollen
print(f"Antall kombinasjoner til testing: {total_combinations}")

# Initialiser variabler for å holde styr på den beste kombinasjonen og Sharpe-ratioen
best_sharpe = -np.inf  # Starter med negativ uendelig for å sikre at enhver positiv Sharpe-ratio er bedre
best_combination = None  # Holder styr på parametrene for den beste kombinasjonen
best_combination_number = None  # Holder styr på kombinasjonsnummeret for den beste kombinasjonen

# Initialiser en liste for å samle inn alle resultater fra testene
results = []

# Start testen over alle kombinasjoner av parametere
combination_counter = 1  # Teller for å holde styr på kombinasjonsnummeret

# Loop gjennom alle kombinasjoner av populasjonsstørrelser, generasjonsantall og mutasjonsrater
for pop_size in population_sizes:
    for gen_count in generation_counts:
        for mut_rate in mutation_rates:
            # Skriv ut informasjon om den nåværende kombinasjonen
            print(f"Kjører kombinasjon {combination_counter}/{total_combinations}: pop_size={pop_size}, gen_count={gen_count}, mut_rate={mut_rate}")
            
            # Kjør den evolusjonære algoritmen med de nåværende parameterne
            best_portfolio, sharpe_ratio = evolutionary_programming(expected_returns, cov_matrix, pop_size, gen_count, risk_free_rate, mut_rate)
            
            # Skriv ut Sharpe-ratioen for denne kombinasjonen
            print(f"Sharpe-ratio for kombinasjon {combination_counter}/{total_combinations}: {sharpe_ratio}")
            
            # Lagre resultatene i resultatlisten, inkludert kombinasjonsnummeret
            results.append({
                'combination_number': combination_counter,  # Kombinasjonsnummer
                'pop_size': pop_size,
                'gen_count': gen_count,
                'mut_rate': mut_rate,
                'sharpe_ratio': sharpe_ratio
            })
            
            # Sjekk om denne Sharpe-ratioen er bedre enn den beste så langt
            if sharpe_ratio > best_sharpe:
                # Oppdater den beste Sharpe-ratioen og parameterne
                best_sharpe = sharpe_ratio
                best_combination = (pop_size, gen_count, mut_rate)
                best_combination_number = combination_counter  # Lagre kombinasjonsnummeret
                
            # Oppdater kombinasjonsnummeret
            combination_counter += 1

# Konverter resultatlisten til en pandas DataFrame for enklere lagring og analyse
results_df = pd.DataFrame(results)

# Lagre resultatene til en CSV-fil uten å inkludere DataFrame-indeksen
results_df.to_csv(results_file, index=False)

# Etter at alle kombinasjoner er testet, skriv ut resultatet for den beste kombinasjonen
print("\nBeste kombinasjon funnet")
print(f"Kombinasjonsnummer: {best_combination_number}/{total_combinations}")
print(f"Sharpe-ratio: {best_sharpe}")
print(f"Populasjonsstørrelse: {best_combination[0]}, Antall generasjoner: {best_combination[1]}, Mutasjonsrate: {best_combination[2]}")
print(f"Beste porteføljevekter: {best_portfolio}")

# Lagre den beste porteføljen (vektene) til en CSV-fil for videre analyse
best_portfolio_df = pd.DataFrame([best_portfolio], columns=returns_df.columns)  # Bruk aksjenavnene som kolonnenavn
best_portfolio_df.to_csv(os.path.join(script_dir, '../3.prob2_output/3.3bep_best_portfolio.csv'), index=False)

# Skriv ut melding om at den beste porteføljen er lagret
print(f"Beste portefølje lagret i '3.3bep_best_portfolio.csv'")
