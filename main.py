import pandas as pd
import numpy as np
from findparams import find_best_initial_guesses
from findparams import get_best_fit_metrics
from findparamsSEIR import get_best_fit_metricsSEIR
from findparamsSEIR import find_best_initial_guessesSEIR
import matplotlib.pyplot as plt

# Load your data
df = pd.read_excel("Influnet_2004_2024_w_age.xlsx")
data_dates = df["Date"][313:337].to_numpy()
data_cases = df["positiveCasi"][313:337].to_numpy()
population = np.average(df["Totale Assistiti"][313:337])

# Define ranges for p and E0
p_range = np.arange(0.1, 3.0, 0.1)  # Example range for p
E0_range = np.arange(1, 100, 1)  # Example range for E0
E0_rangemax = np.arange(1, 3000, 1) # Example range for E0 for fitting SEIR
# Find the best initial guesses
result = find_best_initial_guesses(data_dates, data_cases, population, p_range, E0_range)
resultSEIR = find_best_initial_guessesSEIR(data_dates, data_cases, population, E0_rangemax)
best_metrics = get_best_fit_metrics(result)
best_metricsSEIR = get_best_fit_metricsSEIR(resultSEIR)
# Print the results
print("\nBest SEIR Fit Results:")
print(f"Fitted values: {resultSEIR['best_initial_guess']}")
print(f"Best R-squared Score: {resultSEIR['best_r2_score']:.4f}")
print("Best Metrics:")
for metric, value in resultSEIR['best_metrics'].items():
    print(f"{metric}: {value:.4f}")

# Print the results
print("\nBest SEIR-HET Fit Results:")
print(f"Fitted values: {result['best_initial_guess']}")
print(f"Best R-squared Score: {result['best_r2_score']:.4f}")
print("Best Metrics:")
for metric, value in result['best_metrics'].items():
    print(f"{metric}: {value:.4f}")

# Plot the best fit
    plt.figure(figsize=(10, 6))
    plt.plot(data_dates, data_cases, 'o', label="Observed Data")
    plt.plot(data_dates, result["best_fitted_values"], '-', label="Best SEIR-HET Fit")
    plt.plot(data_dates, resultSEIR["best_fitted_values"], linestyle='dashdot', label="Best SEIR Fit")
    plt.title("Comparison of fitting SEIR and SEIR-HET Model, 2014-15")
    plt.xlabel("Time")
    plt.ylabel("Cases")
    plt.legend()
    plt.savefig('2014-15.jpg')
    plt.show()

