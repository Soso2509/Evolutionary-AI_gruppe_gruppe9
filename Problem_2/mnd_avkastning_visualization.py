import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Last inn dataene fra CSV-filen
returns_df = pd.read_csv('monthly_returns_all_stocks.csv', index_col='Date', parse_dates=True)

# Funksjon for å plotte linjediagram
def plot_line_chart():
    # Plot linjediagram for månedlige avkastninger
    returns_df.plot(figsize=(12, 8), title="Månedlige Avkastninger for Alle Selskaper")
    plt.xlabel("Dato")
    plt.ylabel("Avkastning")
    plt.legend(loc='best', ncol=2)
    plt.grid(True)
    plt.show()

# Funksjon for å lage heatmap
def plot_heatmap():
    # Lag et heatmap av månedlige avkastninger
    plt.figure(figsize=(12, 8))
    sns.heatmap(returns_df, cmap="RdYlGn", annot=False, cbar=True, linewidths=0.5)
    plt.title("Heatmap av Månedlige Avkastninger for Alle Selskaper")
    plt.xlabel("Selskaper")
    plt.ylabel("Dato")
    plt.show()

# Hoveddel: Kjør visualiseringene
if __name__ == '__main__':
    plot_line_chart()  # Kjør linjediagram
    plot_heatmap()     # Kjør heatmap
