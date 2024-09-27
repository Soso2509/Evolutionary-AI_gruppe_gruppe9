import os
import pandas as pd

# Get the directory of the current script (2.1beregn_mnd_avk.py). 
# This is important to construct relative paths later in the script.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the data folder (../1.prob2_data) based on the script's location.
# This allows the script to work across different machines or setups, as long as the directory structure remains consistent.
data_folder = os.path.join(script_dir, '../1.prob2_data')

# Define a function to calculate monthly returns for a given stock data file.
# The file path of the stock data will be passed as an argument to this function.
def calculate_monthly_returns(file_path):
    # Read the stock data from the CSV file into a pandas DataFrame.
    df = pd.read_csv(file_path)
    
    # Convert the 'Date' column to datetime objects to enable handling it as a time series.
    # This step is crucial for resampling data by date.
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set the 'Date' column as the index in the DataFrame.
    # This makes it easier to resample the data based on date intervals (monthly in this case).
    df.set_index('Date', inplace=True)
    
    # Resample the data to get the first adjusted close price ('Adj Close') at the end of each month.
    # 'ME' stands for "Month End", ensuring that data is sampled from the end of each month.
    monthly_prices_first = df['Adj Close'].resample('ME').first()
    
    # Resample the data to get the last adjusted close price at the end of each month.
    # This will allow us to calculate the monthly return by comparing the first and last prices of the month.
    monthly_prices_last = df['Adj Close'].resample('ME').last()
    
    # Calculate the monthly return as the percentage change between the last and first price in each month.
    # The formula (monthly_prices_last / monthly_prices_first) - 1 gives the percentage return.
    monthly_returns = (monthly_prices_last / monthly_prices_first) - 1
    
    # Return a pandas Series object containing the monthly returns for the stock.
    return monthly_returns

# Initialize an empty dictionary to store the return data for each stock.
# The stock name will be the key, and the monthly returns Series will be the value.
returns_data = {}

# Iterate over all files in the data folder to process each stock individually.
for filename in os.listdir(data_folder):
    # Check if the current file is a CSV file.
    # This ensures that we only process stock data files.
    if filename.endswith(".csv"):
        # Build the full file path to the CSV file.
        file_path = os.path.join(data_folder, filename)
        
        # Extract the stock name from the file name (assuming the file name is in the format 'StockName.csv').
        # This will allow us to identify the stock associated with the returns later on.
        stock_name = filename.split('.')[0]
        
        # Calculate the monthly return for the current stock by calling the function.
        monthly_returns = calculate_monthly_returns(file_path)
        
        # Add the return data to the dictionary, with the stock name as the key and the monthly returns as the value.
        returns_data[stock_name] = monthly_returns

# Combine the return data for all stocks into a single DataFrame.
# Each stock's returns will be in a separate column, and the index will be the months.
returns_df = pd.DataFrame(returns_data)

# Define the path to the output file where the combined monthly returns will be saved.
# This will create a CSV file containing all the calculated returns.
output_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')

# Save the combined DataFrame to a CSV file for further use or analysis.
returns_df.to_csv(output_file)

# Print a confirmation message to the console to indicate that the process has completed successfully.
print(f"Monthly returns have been calculated and saved to '{output_file}'.")
