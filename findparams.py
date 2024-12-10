import numpy as np
from seirHET_fit import seirhet_fit
from metrics import evaluate_fit  # Assume this calculates R-squared and other metrics


def find_best_initial_guesses(data_dates, data_cases, population, p_range, E0_range):
    """
    Finds the best initial guesses for p and E0 parameters in the SEIR-HET model.

    Parameters:
        data_dates (array-like): Time points of the observed data.
        data_cases (array-like): Observed cases corresponding to the time points.
        population (int): Total population size.
        p_range (array-like): Range of p values to try.
        E0_range (array-like): Range of E0 values to try.

    Returns:
        dict: Best parameters, R-squared score, and fitted values.
    """
    best_r2_score = -np.inf
    best_initial_guess = None
    best_fitted_values = None
    best_metrics = None

    # Iterate through all combinations of p and E0
    for p in p_range:
        for E0 in E0_range:
            # Initial guesses for other parameters
            initial_guess = [4, 3, 2, p, E0]

            try:
                # Fit the model
                fitted_values, params = seirhet_fit(data_dates, data_cases, population, initial_guess)

                # Evaluate the fit
                metrics = evaluate_fit(data_cases, fitted_values)
                r2_score = metrics.get("R-squared", None)

                # Update the best fit if the current fit is better
                if r2_score is not None and r2_score > best_r2_score:
                    best_r2_score = r2_score
                    best_initial_guess = params
                    best_fitted_values = fitted_values
                    best_metrics = metrics
            except Exception as e:
                print(f"Error fitting with p={p}, E0={E0}: {e}")
                continue

    # Return the best results
    return {
        "best_initial_guess": best_initial_guess,
        "best_r2_score": best_r2_score,
        "best_fitted_values": best_fitted_values,
        "best_metrics": best_metrics,
    }


def get_best_fit_metrics(result):
    """
    Extracts and optionally prints the best fit metrics from the result.

    Parameters:
        result (dict): The dictionary containing the best fit details.

    Returns:
        dict: The best fit metrics.
    """
    best_metrics = result.get("best_metrics", {})

    # Print metrics for inspection (optional)
    print("\nBest Fit Metrics:")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value:.4f}")

    return best_metrics