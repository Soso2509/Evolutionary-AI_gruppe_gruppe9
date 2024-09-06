import os
import pandas as pd

# Definer stien til mappen der CSV-filene er lagret
folder_path = 'prob2_data'

# Funksjon for å beregne månedlig avkastning basert på første og siste dag i måneden
def calculate_monthly_returns(file_path):
    # Les inn CSV-filen
    df = pd.read_csv(file_path)
    
    # Konverter 'Date' til datetime-format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sett 'Date' som index
    df.set_index('Date', inplace=True)
    
    # Hent justert sluttkurs for den første og siste dagen i hver måned
    monthly_prices_first = df['Adj Close'].resample('ME').first()  # Første dag i måneden
    monthly_prices_last = df['Adj Close'].resample('ME').last()   # Siste dag i måneden
    
    # Beregn månedlig avkastning: (siste dag / første dag) - 1
    monthly_returns = (monthly_prices_last / monthly_prices_first) - 1
    
    return monthly_returns

# Lagre alle resultatene i en dictionary
returns_data = {}

# Loop gjennom alle filer i mappen
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        stock_name = filename.split('.')[0]  # Bruk filnavnet som aksjeidentifikator
        monthly_returns = calculate_monthly_returns(file_path)
        returns_data[stock_name] = monthly_returns

# Kombiner dataene for alle aksjer i én DataFrame
returns_df = pd.DataFrame(returns_data)

# Lagre resultatene i en CSV-fil
returns_df.to_csv('mnd_avk_aksjer.csv')

print("Månedlige avkastninger er beregnet og lagret i 'mnd_avk_aksjer.csv'.")
