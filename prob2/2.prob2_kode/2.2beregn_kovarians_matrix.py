import os
import pandas as pd

# Finn stien til dette skriptet (hvor filen kjører fra)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Definer relative stier for input- og output-filer basert på skriptets plassering
input_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')
output_file = os.path.join(script_dir, '../3.prob2_output/3.2beregn_kovarians_matrix.csv')

# Les inn de månedlige avkastningene fra CSV-filen
returns_df = pd.read_csv(input_file, index_col=0)

# Beregn kovariansmatrisen
cov_matrix = returns_df.cov()

# Lagre kovariansmatrisen i en ny CSV-fil
cov_matrix.to_csv(output_file)

print(f"Kovariansmatrisen er beregnet og lagret i '{output_file}'.")

