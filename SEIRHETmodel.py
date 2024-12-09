from scipy import integrate


def seirhet_model(y, x, beta, gamma, alpha, p, N):
    """
    SEIR-HET model function that calculates the derivatives for a heterogeneous SEIR model.

    Parameters:
        y (list or ndarray): Current state of the compartments [S, E, I, R].
        x (float): Time (not used explicitly in this model but required for compatibility with odeint).
        beta (float): Transmission rate parameter.
        gamma (float): Recovery rate parameter.
        alpha (float): Incubation rate parameter.
        p (float): Heterogeneity parameter that modifies the transmission dynamics.
        N (int): Total population size.

    Returns:
        tuple: Derivatives of the compartments (dS/dt, dE/dt, dI/dt, dR/dt).
    """
    N = sum(y)  # Ensure the total population size remains consistent.
    S = -beta * pow(y[0], 1 + (1 / p)) * y[2] / N
    E = beta * pow(y[0], 1 + (1 / p)) * y[2] / N - alpha * y[1]
    I = alpha * y[1] - gamma * y[2]
    R = gamma * y[2]
    return S, E, I, R


def solve_SEIRhet(x, beta, gamma, alpha, p, E0, N):
    """
    Solves the SEIR-HET model using numerical integration.

    Parameters:
        x (array-like): Time points at which to solve the model.
        beta (float): Transmission rate parameter.
        gamma (float): Recovery rate parameter.
        alpha (float): Incubation rate parameter.
        p (float): Heterogeneity parameter that modifies the transmission dynamics.
        E0 (float): Initial number of exposed individuals.
        N (int): Total population size.

    Returns:
        ndarray: Sum of the Exposed (E) and Infectious (I) compartments at each time point.
    """
    S0 = N - E0  # Initial number of susceptible individuals.
    I0 = 0       # Initial number of infectious individuals.
    R0 = 0       # Initial number of recovered individuals.

    # Solve the ODE system for the E and I compartments
    results = integrate.odeint(seirhet_model, (S0, E0, I0, R0), x, args=(beta, gamma, alpha, p, N), atol=1e-3, rtol=1e-3)
    return results[:, 1] + results[:, 2]  # Return the sum of E and I compartments.
