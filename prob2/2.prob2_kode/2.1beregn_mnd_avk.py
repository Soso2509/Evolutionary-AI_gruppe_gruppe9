import os
import pandas as pd

# Finn stien til dette skriptet (2.1beregn_mnd_avk.py) for å kunne bygge relative stier senere
script_dir = os.path.dirname(os.path.abspath(__file__))

# Konstruer den relative stien til data-mappen (../1.prob2_data) basert på skriptets plassering
data_folder = os.path.join(script_dir, '../1.prob2_data')

# Definer en funksjon for å beregne månedlig avkastning for en gitt aksjefil
def calculate_monthly_returns(file_path):
    # Les aksjedata fra CSV-filen inn i en pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Konverter 'Date'-kolonnen til datetime-objekter for tidsseriehåndtering
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sett 'Date'-kolonnen som indeksen i DataFrame for enklere resampling basert på dato
    df.set_index('Date', inplace=True)
    
    # Resample dataene for å få den første justerte lukkekursen ('Adj Close') hver måned
    monthly_prices_first = df['Adj Close'].resample('ME').first()  # 'ME' står for månedlig slutt (Month End)
    
    # Resample dataene for å få den siste justerte lukkekursen hver måned
    monthly_prices_last = df['Adj Close'].resample('ME').last()
    
    # Beregn den månedlige avkastningen som prosentvis endring mellom siste og første kurs hver måned
    monthly_returns = (monthly_prices_last / monthly_prices_first) - 1
    
    # Returner en pandas Series med månedlige avkastninger
    return monthly_returns

# Initialiser en tom dictionary for å lagre avkastningsdataene for hver aksje
returns_data = {}

# Loop gjennom alle filer i data-mappen for å behandle hver aksje individuelt
for filename in os.listdir(data_folder):
    # Sjekk om filen er en CSV-fil
    if filename.endswith(".csv"):
        # Bygg full filsti til CSV-filen
        file_path = os.path.join(data_folder, filename)
        
        # Ekstraher aksjenavnet fra filnavnet (antatt at filnavnet er i formatet 'Aksjenavn.csv')
        stock_name = filename.split('.')[0]
        
        # Beregn månedlig avkastning for den aktuelle aksjen
        monthly_returns = calculate_monthly_returns(file_path)
        
        # Legg til avkastningsdataene i dictionaryen med aksjenavnet som nøkkel
        returns_data[stock_name] = monthly_returns

# Kombiner avkastningsdataene for alle aksjer til én samlet DataFrame
returns_df = pd.DataFrame(returns_data)

# Definer stien til output-filen hvor de månedlige avkastningene skal lagres
output_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')

# Lagre den kombinerte DataFrame til en CSV-fil for videre bruk eller analyse
returns_df.to_csv(output_file)

# Skriv ut en bekreftelsesmelding til konsollen for å informere om at prosessen er fullført
print(f"Månedlige avkastninger er beregnet og lagret i '{output_file}'.")
