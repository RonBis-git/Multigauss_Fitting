import numpy as np
from scipy.special import erfc

from dataclasses import dataclass


@dataclass
class ExcitationFunction:
    E_cm: np.ndarray
    sigma: np.ndarray
    sigma_error: np.ndarray


@dataclass
class ExcitationFunctionModel:
    weight: float
    Vg: float
    Wg: float
    Rg: float

    def expression_xs(self, E_cm):
        # calculating Z
        Z = (E_cm - self.Vg) / (np.sqrt(2.0) * self.Wg)

        # calculating terms for the cross-section
        bracket_term1 = np.sqrt(np.pi) * Z * erfc(-Z)
        bracket_term2 = np.exp(-Z * Z)
        outside_bracket = np.pi * self.Rg**2 * self.Wg / (np.sqrt(2 * np.pi) * E_cm)

        return self.weight * outside_bracket * (bracket_term1 + bracket_term2)


@dataclass
class BarrierDistribution:
    E_bd: np.ndarray
    D_fus: np.ndarray
    D_fus_err: np.ndarray

    @staticmethod
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
                    sigma_err[i + 1] ** 2
                    + 4 * sigma_err[i] ** 2
                    + sigma_err[i - 1] ** 2
                )
            )

        return Ebd, fbd, fbd_err


@dataclass
class BarrierDistributionModel:
    weight: float
    Vg: float
    Wg: float

    def expression_bd(self, E_bd):

        Z = (E_bd - self.Vg) / (np.sqrt(2.0) * self.Wg)

        # calculating terms for the barrier distribution
        bracket_term = np.exp(-Z * Z)
        #   print(bracket_term)
        coeff = 1.0 / (np.sqrt(2 * np.pi) * self.Wg)

        return self.weight * coeff * bracket_term
