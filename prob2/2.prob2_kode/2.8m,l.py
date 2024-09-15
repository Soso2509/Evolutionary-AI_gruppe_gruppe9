import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

# This is the data
returns_path = os.path.join(script_dir, '../3.prob2_output/3.1beregn_mnd_avk.csv')
cov_matrix_path = os.path.join(script_dir, '../3.prob2_output/3.2beregn_kovarians_matrix.csv')

returns_df = pd.read_csv(returns_path)
returns_df = returns_df.drop(columns=['Date'])
returns = np.array(returns_df.values, dtype=float)

cov_matrix_df = pd.read_csv(cov_matrix_path, index_col=0)
cov_matrix = np.array(cov_matrix_df.values, dtype=float)

# These are the parameters
population_sizes_mu = [20, 50, 100]
population_sizes_lambda = [10, 100, 200]
generation_counts = [50, 100, 200]
mutation_rates = [0.01, 0.05, 0.1]
risk_free_rate = 0.02 / 12

# Method for initializing the original random population
def initialize_population(size, num_assets):
    population = np.random.rand(size, num_assets)
    return population / population.sum(axis=1)[:, np.newaxis]

num_assets = returns.shape[1]
print(f'Num_assets = {num_assets}')

# Method for evaluating the fitness of the current population
def fitness_evaluation(population, returns, cov_matrix):
    fitness = np.zeros(population.shape[0])
    for i, individual in enumerate(population):
        expected_return = np.dot(np.mean(returns, axis=0), individual)
        risk = np.sqrt(np.dot(individual.T, np.dot(cov_matrix, individual)))
        fitness[i] = expected_return - risk
    return fitness

def select_best_children(population, fitness, mu):
    best_indices = np.argsort(fitness)[-mu:]
    return population[best_indices]

def generate_children(parents, lambda_, mutation_rate):
    num_parents, num_assets = parents.shape
    offspring = np.zeros((lambda_, num_assets))
    for i in range(lambda_):
        parent = parents[np.random.randint(num_parents)]
        mutation = np.random.randn(num_assets) * mutation_rate
        child = parent + mutation
        child = np.maximum(child, 0)
        offspring[i] = child / np.sum(child)
    return offspring

def evolution_strategy(returns, cov_matrix, mu, lambda_, num_generations, mutation_rate):
    num_assets = returns.shape[1]
    population = initialize_population(mu, num_assets)
    best_fitness_history = []

    for generation in range(num_generations):
        fitness = fitness_evaluation(population, returns, cov_matrix)
        offspring = generate_children(population, lambda_, mutation_rate)
        offspring_fitness = fitness_evaluation(offspring, returns, cov_matrix)
        combined_population = np.vstack((population, offspring))
        combined_fitness = np.concatenate((fitness, offspring_fitness))
        population = select_best_children(combined_population, combined_fitness, mu)
        best_fitness = np.max(combined_fitness)
        best_fitness_history.append(best_fitness)
        print(f'Generation {generation}: Best fitness = {best_fitness}')

    return population, best_fitness_history

# Initializes a list to store all best portfolios
all_best_portfolios = []

# Initializes a figure for convergence plots
plt.figure(figsize=(16, 10))

for i, mu in enumerate(population_sizes_mu):
    for j, lambda_ in enumerate(population_sizes_lambda):
        for k, num_generations in enumerate(generation_counts):
            for l, mutation_rate in enumerate(mutation_rates):
                print(f'\nRunning ES with mu={mu}, lambda_={lambda_}, generations={num_generations}, mutation_rate={mutation_rate}')
                final_population, best_fitness_history = evolution_strategy(returns, cov_matrix, mu, lambda_, num_generations, mutation_rate)
                best_individual = final_population[np.argmax(fitness_evaluation(final_population, returns, cov_matrix))]
                print(f'Best portfolio weights: {best_individual}')
                
                # Save the best portfolio
                all_best_portfolios.append({
                    'mu': mu,
                    'lambda': lambda_,
                    'generations': num_generations,
                    'mutation_rate': mutation_rate,
                    **{f'Asset_{i+1}': weight for i, weight in enumerate(best_individual)}
                })
                
                # Plot convergence
                plt.plot(best_fitness_history, label=f'μ={mu}, λ={lambda_}, gen={num_generations}, mut={mutation_rate}')

# Saves all best portfolios to CSV
df_best_portfolios = pd.DataFrame(all_best_portfolios)
df_best_portfolios.to_csv(os.path.join(script_dir, '../3.prob2_output/3.8m,l_best_portfolios.csv'), index=False)
print("Best portfolios saved to 3.8m,l_best_portfolios.csv")

# Creates a convergence plot
plt.title('Convergence of Best Fitness Over Generations', fontsize=14)
plt.xlabel('Generation', fontsize=12)
plt.ylabel('Best Fitness', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='x-small', ncol=2) 
plt.tight_layout()
plt.subplots_adjust(right=0.75)  
plt.savefig(os.path.join(script_dir, '../4.prob2_visual/3.8_convergence_plot.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Convergence plot saved as convergence_plot.png")

# Creates a bar plot for the final best portfolio
final_best_portfolio = all_best_portfolios[-1]
assets = [f'Asset_{i+1}' for i in range(num_assets)]
weights = [final_best_portfolio[asset] for asset in assets]

plt.figure(figsize=(12, 6))
plt.bar(assets, weights)
plt.title('Asset Allocation of Best Portfolio')
plt.xlabel('Assets')
plt.ylabel('Weight')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, '../4.prob2_visual/3.8_best_portfolio_allocation.png'))
plt.close()

print("Visualizations saved as convergence_plot.png and best_portfolio_allocation.png")