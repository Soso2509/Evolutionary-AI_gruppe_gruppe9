import os
import pandas as pd

# Finn stien til dette skriptet (2.1beregn_mnd_avk.py)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Konstruer den relative stien til 1.prob2_data basert på skriptets plassering
data_folder = os.path.join(script_dir, '../1.prob2_data')

# Funksjon for å beregne månedlig avkastning basert på første og siste dag i måneden
def calculate_monthly_returns(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    monthly_prices_first = df['Adj Close'].resample('ME').first()  # Første dag i måneden
    monthly_prices_last = df['Adj Close'].resample('ME').last()   # Siste dag i måneden
    monthly_returns = (monthly_prices_last / monthly_prices_first) - 1
    return monthly_returns

# Lagre alle resultatene i en dictionary
returns_data = {}

# Loop gjennom alle filer i data-mappen
for filename in os.listdir(data_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(data_folder, filename)
        stock_name = filename.split('.')[0]
        monthly_returns = calculate_monthly_returns(file_path)
        returns_data[stock_name] = monthly_returns

# Kombiner dataene for alle aksjer i én DataFrame
returns_df = pd.DataFrame(returns_data)

# Lagre de månedlige avkastningene i en CSV-fil
output_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')
returns_df.to_csv(output_file)

print(f"Månedlige avkastninger er beregnet og lagret i '{output_file}'.")

