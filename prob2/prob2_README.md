# Running code from Problem 2
**Make sure you are in the root folder `\Evolutionary-AI_gruppe_gruppe9`**

[Link back to main README](../README.md)

**Dependecies to run code**\
(*If installed `requirements.txt` they are already installed*)
- pandas
- numpy
- matplotlib

## Overview
1. [Calculating monthly returns](#1-calculate-monthly-return)
2. [Calculate the covariance matrix](#2-calculate-the-covariance-matrix)
3. [Basic Evolutionary Programming](#3-runs-basic-evolutionary-programming-bep)
4. [Advanced Evolutionary Programming](#4-runs-advanced-evolutionary-programming-aep)
5. [Basic Evolutionary Strategies](#5-runs-basic-evolutionary-strategies-bes)
6. [Advanced Evolutionary Strategies](#6-runs-advanced-evolutionary-strategies-aes)
7. [(μ + λ) Evolutionary Strategies](#7-runs-μ--λ-evolutionary-strategies)
8. [(μ, λ) Evolutionary Strategies](#8-runs-μ-λ-evolutionary-strategies)
- [HTML file with tables from CSV](#tables-of-the-csv-files)

## 1. Calculate monthly return
Will print the result to a CSV file in `/3.prob2_output` called `3.1beregn_mnd_avk.csv`
```bash
python prob2/2.prob2_kode/2.1beregn_mnd_avk.py
```

## 2. Calculate the covariance matrix
Will print the result to a CSV file in `/3.prob2_output` called `3.2beregn_kovarians_matrix.csv`
```bash
python prob2/2.prob2_kode/2.2beregn_kovarians_matrix.py
```

## 3. Runs Basic Evolutionary Programming (BEP)
Will print the overall result and the best portfolio weights to CSV files in `/3.prob2_output` called `3.3bep.csv` and `3.3bep_best_portfolio`
```bash
python prob2/2.prob2_kode/2.3bep.py
```

## 4. Runs Advanced Evolutionary Programming (AEP)
Will print the overall result and the best portfolio weights to CSV files in `/3.prob2_output` called `3.4aep.csv` and `3.4aep_best_portfolio.csv`
```bash
python prob2/2.prob2_kode/2.4aep.py
```

## 5. Runs Basic Evolutionary Strategies (BES)
Will print the overall result and the best portfolio weights to CSV files in `/3.prob2_output` called `3.5bes.csv` and `3.5bes_best_portfolio.csv`
```bash
python prob2/2.prob2_kode/2.5bes.py
```

## 6. Runs Advanced Evolutionary Strategies (AES)
Will print the overall result and the best portfolio weights to CSV files in `/3.prob2_output` called `3.6aes.csv` and `3.6aes_best_portfolio.csv`
```bash
python prob2/2.prob2_kode/2.6aes.py
```

## 7. Runs (μ + λ) Evolutionary Strategies
Will print the overall result and the best portfolio weights to CSV files in `/3.prob2_output` called `3.7m+l.csv` and `3.7m+l_best_portfolio.csv`
```bash
python prob2/2.prob2_kode/2.7m+l.py
```

## 8. Runs (μ, λ) Evolutionary Strategies
Will print the overall result and the best portfolio weights to CSV files in `/3.prob2_output` called `3.8m,l.csv` and `3.8m,l_best_portfolio.csv`
```bash
python prob2/2.prob2_kode/2.8m,l.py
```

## Tables of the CSV files
### Step 1: Generate HTML file with tables of the CSV files
```bash
python prob2/4.prob2_visual/4.1visual.py
```

### Step 2: Navigate to folder
```bash
cd prob2/4.prob2_visual/
```

### Step 3: Start the server
```bash
python -m http.server 8000
```

### Step 4: Open browser and go to following URL to see the tables
http://localhost:8000/aksje_avkastning_sorterbar.html

### Step 5: "Kill" the server
```bash
Ctrl + C
```
