import numpy as np
from scipy import optimize
from SEIRHETmodel import solve_SEIRhet


def seirhet_fit(data_dates, data_cases, N, initial_guess):
    """
    Fits the SEIR-HET model to the given data using curve fitting.

    Parameters:
        data_dates (array-like): Time points of the observed data.
        data_cases (array-like): Observed cases corresponding to the time points.
        N (int): Total population size.
        initial_guess (list): Initial guesses for the parameters [beta, gamma, alpha, p, E0].

    Returns:
        dict: Fitted parameters and their values.
    """

    # Create a wrapper for the solve_SEIRhet function
    def wrapped_model(x, beta, gamma, alpha, p, E0):
        return solve_SEIRhet(x, beta, gamma, alpha, p, E0, N)

    popt, _ = optimize.curve_fit(wrapped_model, data_dates, data_cases, p0=initial_guess, bounds=((0, 0, 0, 0, 0), (np.inf, np.inf, np.inf, np.inf, N)))
    fitted_values = wrapped_model(data_dates, *popt)
    return fitted_values, popt
