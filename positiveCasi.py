import pandas as pd
import numpy as np
from seir_fit import seir_fit
from seirHET_fit import seirhet_fit
import matplotlib.pyplot as plt
from metrics import evaluate_fit
from SEIRHETmodel import solve_SEIRhet

df = pd.read_excel("Influnet_2004_2024_w_age.xlsx")

dates2014 = df["Date"][313:337]
dataCases2014 = df["Totale Casi"][313:337]
positiveCases2014 = df["positiveCasi"][313:337]
pop = np.average(df["Totale Assistiti"][313:337])

dates2015 = df["Date"][343:365]
dataCases2015 = df["Totale Casi"][343:365]
positiveCases2015 = df["positiveCasi"][343:365]
pop2015 = np.average(df["Totale Assistiti"][343:365])
print(len(positiveCases2015))

initial_guess = [4, 3, 2, 722]  # Initial guesses for [beta, gamma, alpha, E0]
initial_guess_het = [4, 3, 2, 60, 722]  # Initial guesses for [beta, gamma, alpha, p, E0]

# Fit the models
fitted_seir, params_seir = seir_fit(dates2015, positiveCases2015, pop2015, initial_guess)
fitted_seirhet, params_seirhet = seirhet_fit(dates2015, positiveCases2015, pop2015, initial_guess_het)
test = solve_SEIRhet(dates2015, 1.10498456, 6.99534916, 7., 7.34012153, 348.36687539, pop2015)


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(dates2015, positiveCases2015, 'o', label='Observed Cases')
plt.plot(dates2015, fitted_seir, label='SEIR Fit', linestyle='--')
plt.plot(dates2015, fitted_seirhet, label='SEIR-HET Fit', linestyle=':')
plt.plot(dates2015, test, label='SEIR-HET Fit2', linestyle='dashdot')
plt.title('Comparison of SEIR and SEIR-HET Fits, 2015')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.legend()
plt.savefig("Comparison2015.png")
plt.show()

# Evaluate the fit

metricsHET = evaluate_fit(positiveCases2015, fitted_seirhet)

# Print the results
print("Fitted SEIR-HET parameters:", params_seirhet)
print("Evaluation Metrics:")
for metric, value in metricsHET.items():
    print(f"{metric}: {value:.4f}")

metrics = evaluate_fit(positiveCases2015, fitted_seir)

# Print the results
print("Fitted SEIR parameters:", params_seir)
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
