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
    experimental_data: ExcitationFunction
    weight: float
    Vg: float
    Wg: float
    Rg: float

    def expression_xs(self):
        # calculating Z
        Z = (self.experimental_data.E_cm - self.Vg) / (np.sqrt(2.0) * self.Wg)

        # calculating terms for the cross-section
        bracket_term1 = np.sqrt(np.pi) * Z * erfc(-Z)
        bracket_term2 = np.exp(-Z * Z)
        outside_bracket = (
            np.pi
            * self.Rg**2
            * self.Wg
            / (np.sqrt(2 * np.pi) * self.experimental_data.E_cm)
        )

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
    experimental_data: BarrierDistribution
    weight: float
    Vg: float
    Wg: float

    def expression_bd(self):

        Z = (self.experimental_data.E_bd - self.Vg) / (np.sqrt(2.0) * self.Wg)

        # calculating terms for the barrier distribution
        bracket_term = np.exp(-Z * Z)
        #   print(bracket_term)
        coeff = 1.0 / (np.sqrt(2 * np.pi) * self.Wg)

        return self.weight * coeff * bracket_term


# def get_chi2(x, y_expt, y_err, y_model):
#     return np.sum((y_expt - y_model) ** 2 / y_err**2) / len(x)


# def get_y_model(E_cm: np.ndarray, *params: np.ndarray) -> np.ndarray:
#     y_model = np.zeros_like(E_cm)

#     num_gauss = int(len(params) / (3))
#     weight = np.asarray(params[:num_gauss-1])
#     Vg = np.asarray(params[num_gauss-1:2*num_gauss-1])
#     Wg = np.asarray(params[2*num_gauss-1:3*num_gauss-1])
#     R0 = np.asarray(params[-1])
#     np.append(weight, 1 - np.sum(weight[:-1]))  #  constraint on weights

#     for i in range(num_gauss):
#         y_model += (
#             10
#             * np.pi
#             * R0**2
#             * expression_xs(E_cm, weight=weight[i], Vg=Vg[i], Wg=Wg[i])
#         )
#     return y_model

# def get_y_model_bd(E_cm: np.ndarray, *params: np.ndarray) -> np.ndarray:
#     y_model = np.zeros_like(E_cm)

#     num_gauss = int(len(params) / 3)
#     weight = np.asarray(params[:num_gauss])
#     Vg = np.asarray(params[num_gauss:2*num_gauss])
#     Wg = np.asarray(params[2*num_gauss:3*num_gauss])
#     A = np.asarray(params[-1])
#     weight[-1] = 1 - np.sum(weight[:-1])  #  constraint on weights

#     for i in range(num_gauss):
#         y_model += expression_bd(E_cm, weight=weight[i], Vg=Vg[i], Wg=Wg[i])
#         # print(y_model)
#     return A*y_model


# def get_init_params(N):
#     if N == 1:
#         weight = 1.0
#     else:
#         weight = 1 / N * np.ones(N-1)

#     Vg = np.random.default_rng().uniform(0, 100, size=N)
#     Wg = np.random.default_rng().uniform(0, 10, size=N)
#     R0 = np.random.default_rng().uniform(0, 20, size=1)

#     return np.hstack((weight, Vg, Wg, R0))

# def get_init_params_bd(N):
#     if N == 1:
#         weight = 1.0
#     else:
#         weight = 1 / N * np.ones(N)

#     Vg = np.random.default_rng().uniform(0, 100, size=N)
#     Wg = np.random.default_rng().uniform(0, 6, size=N)

#     return np.hstack((weight, Vg, Wg,[1000]))

# def optimize_bd(x, y, y_err, max_num_gauss=5):

#     num_gauss = np.arange(2,max_num_gauss+1)
#     chi2 = np.empty(max_num_gauss, dtype=float)

#     lower_bounds = 0

#     for ng in num_gauss:
#         weight_upper_bound = np.ones(ng-1)
#         Vg_upper_bound = 100*np.ones(ng)
#         Wg_upper_bound = 10*np.ones(ng)
#         A_upper_bound = 50000
#         upper_bounds = np.hstack((weight_upper_bound,Vg_upper_bound,Wg_upper_bound, A_upper_bound))

#         init_params = get_init_params_bd(ng)

#         popt, pcov = curve_fit(get_y_model_bd, x, y, p0=init_params, sigma=y_err, bounds = (lower_bounds, upper_bounds))
#         print(popt, pcov)
#         chi2[ng-1] = get_chi2(x,y,y_err,get_y_model_bd(x,*popt))
#         print(get_y_model_bd(x,*popt))
#         print(chi2[ng-1])

#     print(num_gauss, chi2)
#     plt.plot(num_gauss, chi2)
#     plt.show()

# def optimize_xs(x, y, y_err, max_num_gauss=5):

#     num_gauss = np.arange(1,max_num_gauss+1)
#     chi2 = np.empty(max_num_gauss, dtype=float)

#     lower_bounds = 0

#     for ng in num_gauss:
#         weight_upper_bound = np.ones(ng-1)
#         Vg_upper_bound = 100*np.ones(ng)
#         Wg_upper_bound = 10*np.ones(ng)
#         R0_upper_bound = np.asarray([50])
#         upper_bounds = np.hstack((weight_upper_bound,Vg_upper_bound,Wg_upper_bound, R0_upper_bound))

#         init_params = get_init_params(ng)

#         popt, pcov = curve_fit(get_y_model, x, y, p0=init_params, sigma=y_err, bounds = (lower_bounds, upper_bounds))
#         print(popt)
#         chi2[ng-1] = get_chi2(x,y,y_err,get_y_model(x,*popt))
#         print(f'y_model = {get_y_model(x,*popt)}')
#         print(chi2[ng-1])

#     print(num_gauss, chi2)
#     chii2 = len(y)*chi2/(len(y)-3*num_gauss)
#     plt.plot(num_gauss, chi2)
#     plt.plot(num_gauss, chii2)
#     plt.show()
