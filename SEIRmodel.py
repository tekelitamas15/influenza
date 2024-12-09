from scipy import integrate

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

def solve_SEIR(x, beta, gamma, alpha, E0, N):
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
    S0 = N - E0
    I0 = 0
    R0 = 0
    results = integrate.odeint(seir_model, (S0, E0, I0, R0), x, args=(beta, gamma, alpha, N))
    return results[:, 1] + results[:, 2]









