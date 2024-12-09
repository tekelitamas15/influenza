import pandas as pd
from seir_fit import seir_fit
from seirHET_fit import seirhet_fit
import matplotlib.pyplot as plt
from metrics import evaluate_fit

# Observed data (replace with your actual data)


df = pd.read_excel("Influnet_2004_2024_w_age.xlsx")

dataDate2003 = df["Date"][1:29]
dataCases2003 = df["Totale Casi"][1:29]
N = 1176488  # Total population
initial_guess = [4, 3, 2, 67]  # Initial guesses for [beta, gamma, alpha, E0]
initial_guess_het = [4, 3, 2, 40, 67]  # Initial guesses for [beta, gamma, alpha, p, E0]


# Fit the models
fitted_seir, params_seir = seir_fit(dataDate2003, dataCases2003, N, initial_guess)
fitted_seirhet, params_seirhet = seirhet_fit(dataDate2003, dataCases2003, N, initial_guess_het)

print(fitted_seirhet, params_seirhet)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(dataDate2003, dataCases2003, 'o', label='Observed Cases')
plt.plot(dataDate2003, fitted_seir, label='SEIR Fit', linestyle='--')
plt.plot(dataDate2003, fitted_seirhet, label='SEIR-HET Fit', linestyle=':')
plt.title('Comparison of SEIR and SEIR-HET Fits, 2003')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.legend()
plt.savefig("Comparison-initial.png")
plt.show()

# Evaluate the fit

metricsHET = evaluate_fit(dataCases2003, fitted_seirhet)

# Print the results
print("Fitted SEIR-HET parameters:", params_seirhet)
print("Evaluation Metrics:")
for metric, value in metricsHET.items():
    print(f"{metric}: {value:.4f}")

metrics = evaluate_fit(dataCases2003, fitted_seir)

# Print the results
print("Fitted SEIR-HET parameters:", params_seir)
print("Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Print fitted parameters in the form of (name, value)
param_names_seir = ["Beta", "Gamma", "Alpha", "E0"]
print("SEIR Parameters:")
print([(name, value) for name, value in zip(param_names_seir, params_seir)])

param_names_seirhet = ["Beta", "Gamma", "Alpha", "p", "E0"]
print("SEIR-HET Parameters:")
print([(name, value) for name, value in zip(param_names_seirhet, params_seirhet)])

print(len(fitted_seir))
print(len(params_seir))
print(len(fitted_seirhet))
print(len(params_seirhet))