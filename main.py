import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SEIRmodel import solve_SEIR
from scipy import optimize
from metrics import evaluate_fit
from SEIR_fit import seir_fit
from SEIRHET_fit import seirhet_fit

# Load and preprocess data
df = pd.read_excel("Influnet_2004_2024_w_age.xlsx")
data_dates = df["Date"][565:589].to_numpy()
data_cases = df["positiveCasi"][565:589].to_numpy()
"""
2014: [313:337]
2015: [343:365]
2016: [369:393]
2017: [397:421]
2018: [425:449]
2019: [453:472]
2021: [509:533]
2022: [537:561]
2023: [565:589]
"""

# Define constants
N = 59_500_000  # Total population size
ur = 3.8  # Scaling factor for cases


# Initial parameter guesses
initial_guess_seir = [5, 4, 4, 50_000, 20_000]  # [beta, gamma, alpha, E0]
initial_guess_seirhet = [5, 4, 4, 1, 50_000]  # [beta, gamma, alpha, p, E0]

# Fit SEIR model
fitted_values_seir, popt_seir = seir_fit(data_dates, ur * data_cases, N, initial_guess_seir)
beta_seir, gamma_seir, alpha_seir, E0_seir, R0_seir = popt_seir

# Fit SEIR-HET model
fitted_values_seirhet, popt_seirhet = seirhet_fit(data_dates, ur * data_cases, N, initial_guess_seirhet)
beta_seirhet, gamma_seirhet, alpha_seirhet, p_seirhet, E0_seirhet = popt_seirhet

# Print fitted parameters
print("SEIR Model Fitted Parameters:")
print(f"Beta: {beta_seir}, Gamma: {gamma_seir}, Alpha: {alpha_seir}, E0: {E0_seir}, R0_frac: {R0_seir/N}, R_0: {beta_seir * (1-R0_seir/N)/gamma_seir}")
print("Model Fit Evaluation:", evaluate_fit(ur * data_cases, fitted_values_seir))

print("\nSEIR-HET Model Fitted Parameters:")
print(f"Beta: {beta_seirhet}, Gamma: {gamma_seirhet}, Alpha: {alpha_seirhet}, p: {p_seirhet}, E0: {E0_seirhet}, R_0: {beta_seirhet/gamma_seirhet}")
print("Model Fit Evaluation:", evaluate_fit(ur * data_cases, fitted_values_seirhet))

# Plot results
plt.figure(figsize=(9, 6))
plt.plot(data_dates, ur * data_cases, 'bo', label='Observed Cases')
plt.plot(data_dates, fitted_values_seirhet, '-', label='Fitted SEIR-HET Model')
plt.plot(data_dates, fitted_values_seir, 'r-', label='Fitted SEIR Model')
plt.xlabel("Time")
plt.ylabel("Cases")
plt.legend()
plt.title("SEIR Models Fit to Data")
plt.savefig('fit23.jpg')
plt.show()




