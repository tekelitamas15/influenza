from scipy import integrate
import numpy as np
from scipy import optimize


def seir_model(y, x, beta, gamma, alpha, N):
    """
    Defines the SEIR model.

    Parameters:
        y (tuple): Current state [S, E, I, R].
        x (float): Time point.
        beta (float): Infection rate.
        gamma (float): Recovery rate.
        alpha (float): Rate of exposed individuals becoming infectious.
        N (int): Total population size.

    Returns:
        tuple: The derivatives of S, E, I, R.
    """
    S = -beta * y[0] * y[2] / N
    E = beta * y[0] * y[2] / N - alpha * y[1]
    I = alpha * y[1] - gamma * y[2]
    R = gamma * y[2]
    return S, E, I, R

def solve_SEIR(x, beta, gamma, alpha, E0, R0, N, ur):
    """
    Solves the SEIR model numerically over time.

    Parameters:
        x (array-like): Time points for integration.
        beta (float): Infection rate.
        gamma (float): Recovery rate.
        alpha (float): Rate of exposed individuals becoming infectious.
        E0 (float): Initial number of exposed individuals.
        N (int): Total population size.

    Returns:
        array: Fitted number of cases (Exposed + Infected) over time.
    """

    I0 = E0
    S0 = N - E0 - I0 - R0
    results = integrate.odeint(seir_model, (S0, E0, I0, R0), x, args=(beta, gamma, alpha, N))

    diff = [ur * E0]
    for i in range(1, len(results[:, 3])):
        diff.append(results[:, 3][i] - results[:, 3][i-1])
    return diff

def seir_fit(data_dates, data_cases, N, initial_guess):
    """
    Perform curve fitting for the SEIR model to the data.

    Parameters:
        data_dates (array-like): Time points.
        data_cases (array-like): Observed cases data.
        N (int): Total population size.
        initial_guess (list): Initial guesses for [beta, gamma, alpha, E0].

    Returns:
        dict: Fitted parameters (beta, gamma, alpha, E0).
    """

    # Define a function for SEIR fitting
    def wrapped_model(x, beta, gamma, alpha, E0):
        return solve_SEIR(x, beta, gamma, alpha, E0, N, ur)

    popt, _ = optimize.curve_fit(
            wrapped_model, data_dates, data_cases, p0=initial_guess,
            bounds=((0, 3.9, 3.9, 0), (np.inf, 4.1, 4.1, N))
        )
    fitted_values = wrapped_model(data_dates, *popt)
    return fitted_values, popt


















