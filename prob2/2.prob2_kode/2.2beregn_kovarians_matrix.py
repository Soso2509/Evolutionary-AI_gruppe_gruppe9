import os
import pandas as pd

# Import the necessary libraries:
# - 'os' is used for file and directory handling (for creating paths and managing files)
# - 'pandas' (aliased as 'pd') is used for data analysis and manipulation, especially for handling tabular data

# Find the absolute path of the directory where this script is located.
# This is useful for constructing paths relative to the script, ensuring the code works regardless of where it's executed.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative paths for the input and output files based on the script's location:
# - 'input_file' points to the file that contains the calculated monthly returns (from previous calculations)
# - 'output_file' is where the calculated covariance matrix will be saved.
input_file = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')
output_file = os.path.join(script_dir, '../3.prob2_output/3.2beregn_kovarians_matrix.csv')

# Read the monthly returns from the input CSV file into a pandas DataFrame.
# The 'index_col=0' argument specifies that the first column (typically the date) should be used as the row index.
# This is crucial because the dates (or periods) are the reference points for the return data.
returns_df = pd.read_csv(input_file, index_col=0)

# Calculate the covariance matrix based on the monthly returns data.
# The 'cov()' function computes the covariance between all pairs of stocks in the DataFrame.
# Covariance is a measure of how two stocks move together. A positive value indicates they move in the same direction, 
# while a negative value means they move in opposite directions.
cov_matrix = returns_df.cov()

# Save the covariance matrix to a new CSV file for further analysis or use in other models.
# This output file will be stored in the output directory alongside other generated data files.
cov_matrix.to_csv(output_file)

# Print a confirmation message to inform the user that the covariance matrix has been successfully calculated and saved.
print(f"The covariance matrix has been calculated and saved to '{output_file}'.")
