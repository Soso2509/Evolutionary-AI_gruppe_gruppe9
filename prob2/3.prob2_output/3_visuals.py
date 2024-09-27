import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the CSV file (update this path as needed)
script_dir = os.path.dirname(os.path.abspath(__file__))
results_file = os.path.join(script_dir, '../3.prob2_output/3.3bep.csv')

# Load the results CSV file into a pandas DataFrame
results_df = pd.read_csv(results_file)

# Check if 'generation' column exists
if 'generation' not in results_df.columns:
    print("Warning: 'generation' column is missing from the CSV file.")
else:
    # Function to plot Sharpe ratios over generations for each combination
    def plot_sharpe_ratios(results_df):
        # Get the unique combinations (based on combination_number)
        unique_combinations = results_df['combination_number'].unique()

        # Plot Sharpe ratios for each combination
        plt.figure(figsize=(10, 6))

        for combination in unique_combinations:
            # Filter data for the current combination
            combination_data = results_df[results_df['combination_number'] == combination]

            # Extract generation and Sharpe ratio values
            generations = combination_data['generation']
            sharpe_ratios = combination_data['sharpe_ratio']

            # Plot the data
            plt.plot(generations, sharpe_ratios, label=f'Combination {combination}')

        plt.xlabel('Generation')
        plt.ylabel('Sharpe Ratio')
        plt.title('Sharpe Ratio over Generations for Different Combinations')
        plt.legend(loc='best', bbox_to_anchor=(1.05, 1), fontsize='small')
        plt.tight_layout()
        plt.grid(True)
        plt.show()

# Function to plot the best combination (if needed)
def plot_best_combination(results_df):
    # Find the combination with the highest Sharpe ratio
    best_combination = results_df.loc[results_df['sharpe_ratio'].idxmax()]

    # Display the best combination info
    print(f"Best Combination:\n{best_combination}")

    # Plot bar chart for this combination's Sharpe ratios over generations
    if 'generation' in results_df.columns:
        best_comb_data = results_df[results_df['combination_number'] == best_combination['combination_number']]
        plt.figure(figsize=(10, 6))
        plt.plot(best_comb_data['generation'], best_comb_data['sharpe_ratio'], label='Best Combination')
        plt.xlabel('Generation')
        plt.ylabel('Sharpe Ratio')
        plt.title('Sharpe Ratio over Generations for Best Combination')
        plt.legend()
        plt.grid(True)
        plt.show()

# Bar plot for the best portfolio weights
def plot_best_portfolio_weights(best_portfolio_df):
    # Assume the CSV has a single row with portfolio weights
    best_portfolio_weights = best_portfolio_df.iloc[0].values

    plt.figure(figsize=(12, 6))
    plt.bar(best_portfolio_df.columns, best_portfolio_weights, color='skyblue')
    plt.title('Best Portfolio Weights')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Call the function to plot the Sharpe ratios over generations
if 'generation' in results_df.columns:
    plot_sharpe_ratios(results_df)

# Plot Sharpe ratios for the best combination
plot_best_combination(results_df)

# Load the best portfolio weights from the CSV file
best_portfolio_file = os.path.join(script_dir, '../3.prob2_output/3.3ep_best_portfolio.csv')
best_portfolio_df = pd.read_csv(best_portfolio_file)

# Plot the best portfolio weights
plot_best_portfolio_weights(best_portfolio_df)
