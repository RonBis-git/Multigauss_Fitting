import numpy as np
from scipy.special import erfc

def cross_section_model(weight, Vg, Wg, Rg, E_cm):
    # calculating Z
    Z = (E_cm - Vg) / (np.sqrt(2.0) * Wg)

    # calculating terms for the cross-section
    bracket_term1 = np.sqrt(np.pi) * Z * erfc(-Z)
    bracket_term2 = np.exp(-Z * Z)
    outside_bracket = 10 * np.pi * Rg**2 * Wg / (np.sqrt(2 * np.pi) * E_cm)

    return weight * outside_bracket * (bracket_term1 + bracket_term2)


def get_BD_from_xs(E, sigma, sigma_err):
    Esigma = E * sigma

    fbd = np.empty(len(E) - 2)
    fbd_err = np.empty((len(E) - 2))
    Ebd = E[1:-1]

    for i in range(1, len(E) - 1):
        fbd[i - 1] = (
            2
            / (E[i + 1] - E[i - 1])
            * (
                (Esigma[i + 1] - Esigma[i]) / (E[i + 1] - E[i])
                - (Esigma[i] - Esigma[i - 1]) / (E[i] - E[i - 1])
            )
        )
        fbd_err[i - 1] = (
            Ebd[i - 1]
            / (E[i + 1] - E[i - 1]) ** 2
            * np.sqrt(
                sigma_err[i + 1] ** 2 + 4 * sigma_err[i] ** 2 + sigma_err[i - 1] ** 2
            )
        )

    return Ebd, fbd, fbd_err


def barrier_distribution_model(weight, Vg, Wg, E_bd):

    Z = (E_bd - Vg) / (np.sqrt(2.0) * Wg)

    # calculating terms for the barrier distribution
    bracket_term = np.exp(-Z * Z)
    coeff = 1.0 / (np.sqrt(2 * np.pi) * Wg)

    return weight * coeff * bracket_term
