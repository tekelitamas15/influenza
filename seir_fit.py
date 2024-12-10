import numpy as np
from scipy import optimize
from SEIRmodel import solve_SEIR


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
        return solve_SEIR(x, beta, gamma, alpha, E0, N)

    popt, _ = optimize.curve_fit(
            wrapped_model, data_dates, data_cases, p0=initial_guess,
            bounds=((0, 0, 0, 0), (np.inf, 7, 7, N))
        )
    fitted_values = wrapped_model(data_dates, *popt)
    return fitted_values, popt



