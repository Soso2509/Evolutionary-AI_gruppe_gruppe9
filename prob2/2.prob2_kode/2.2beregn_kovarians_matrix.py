import os
import pandas as pd

# Importerer nødvendige biblioteker:
# - 'os' for fil- og kataloghåndtering
# - 'pandas' (forkortet til 'pd') for dataanalyse og manipulasjon

# Finn den absolutte stien til dette skriptet (der denne filen kjører fra)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Definer relative stier for input- og output-filer basert på skriptets plassering
# Input-fil: CSV-fil med de beregnede månedlige avkastningene
input_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')
# Output-fil: CSV-fil hvor den beregnede kovariansmatrisen skal lagres
output_file = os.path.join(script_dir, '../3.prob2_output/3.2beregn_kovarians_matrix.csv')

# Les inn de månedlige avkastningene fra input-filen til en pandas DataFrame
# 'index_col=0' angir at første kolonne i CSV-filen skal brukes som radindeks (typisk datoer)
returns_df = pd.read_csv(input_file, index_col=0)

# Beregn kovariansmatrisen basert på de månedlige avkastningene
# 'cov()' funksjonen beregner kovarians mellom alle kombinasjoner av aksjer
cov_matrix = returns_df.cov()

# Lagre kovariansmatrisen i en ny CSV-fil for videre analyse eller bruk
cov_matrix.to_csv(output_file)

# Skriv ut en bekreftelsesmelding som informerer om at prosessen er fullført
print(f"Kovariansmatrisen er beregnet og lagret i '{output_file}'.")
