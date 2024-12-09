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

initial_guess_het = []
for i in range(1, 1000):
    list = [4, 3, 2, 100, i]
    initial_guess_het.append(list)  # Initial guesses for [beta, gamma, alpha, p, E0]

print(initial_guess_het)

fitted_seirhet = []
params_seirhet = []
for i in range(1, 100):
    fitted_seirhet.append(seirhet_fit(dataDate2003, dataCases2003, N, initial_guess_het[i])[0])
    params_seirhet.append(seirhet_fit(dataDate2003, dataCases2003, N, initial_guess_het[i])[1])

for i in range(len(fitted_seirhet)):
    print(fitted_seirhet[i])

metrics = []
for i in range(len(fitted_seirhet)):
    metrics.append(evaluate_fit(dataDate2003, dataCases2003, fitted_seirhet[i]))
    print("Fitted Parameters for initial guess E0 = " + str(200 + i) + ":", params_seirhet[i])
    print("Evaluation Metrics:")
    for metric, value in metrics[i].items():
        print(f"{metric}: {value:.4f}")
"""

metrics = []
for i in range(1, 10):
    metrics.append(evaluate_fit(dataDate2003, dataCases2003, fitted_seirhet[i]))
    print("Fitted Parameters:", params_seirhet[i])
    print("Evaluation Metrics:")
    for metric, value in metrics[i].items():
        print(f"{metric}: {value:.4f}")



# Fit the models
fitted_seirhet, params_seirhet = seirhet_fit(dataDate2003, dataCases2003, N, initial_guess_het)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(dataDate2003, dataCases2003, 'o', label='Observed Cases')
plt.plot(dataDate2003, fitted_seirhet, label='SEIR-HET Fit', linestyle=':')
plt.title('SEIR-HET Fit, 2003')
plt.xlabel('Weeks')
plt.ylabel('Cases')
plt.legend()
plt.savefig("Comparison-initial.png")
plt.show()

# Evaluate the fit
metrics = evaluate_fit(dataDate2003, dataCases2003, fitted_seirhet)

# Print the results
print("Fitted Parameters:", params_seirhet)
print("Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")


param_names_seirhet = ["Beta", "Gamma", "Alpha", "p", "E0"]
print("SEIR-HET Parameters:")
print([(name, value) for name, value in zip(param_names_seirhet, params_seirhet)])
print(len(fitted_seirhet))
print(len(params_seirhet))
"""


