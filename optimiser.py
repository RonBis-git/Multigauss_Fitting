import numpy as np
from scipy.optimize import curve_fit

import pandas as pd

import matplotlib.pyplot as plt

import models


def get_chi0_2(x, y_expt, y_err, y_model):
    """Calculates the chi_0_sq values"""
    return np.sum((y_expt - y_model) ** 2 / y_err**2) / len(x)


def get_init_params_cross_section(N):
    """Function to randomly generate initial guesses of parameters within a given range for N gaussians"""

    Vg = np.random.default_rng().uniform(20, 100, size=N)
    Wg = np.random.default_rng().uniform(1, 5, size=N)
    R0 = np.random.default_rng().uniform(7, 15, size=1)
    weight = 1 / N * np.ones(N)

    return np.hstack((weight, Vg, Wg, R0))  # Stack(concatenate) into a single array


def unpack_params_cross_section(params):

    # Converting from tuple to numpy array
    params = np.asarray(params)

    num_gauss = int(len(params) / (3))  # Get number of gaussians to fit

    weight = params[:num_gauss]  # Extract raw weight
    # Normalise weight according to constraint
    constrained_weight = weight / np.sum(weight)

    Vg = params[num_gauss : 2 * num_gauss]  # Extract barrier height parameter
    Wg = params[2 * num_gauss : 3 * num_gauss]  # Extract width parameter
    R0 = params[-1]  # Extract normalisation parameter

    return num_gauss, constrained_weight, Vg, Wg, R0


def get_parameter_bounds_cross_section(num_gauss: int):
    """
    Function to return lower and upper bounds of optimisation parameters.
    Remark: The return of bounds can only be used in arguements of scipy.optimise.
            It is designed in only this way.
    """

    lower_bounds = 0  # Lower bounds for all the parameters

    # Upper bounds.
    # Change the bounds according to your needs.

    weight_upper_bound = np.ones(num_gauss)  # N*1
    Vg_upper_bound = 200 * np.ones(num_gauss)  # N * 200
    Wg_upper_bound = 10 * np.ones(num_gauss)  # N * 10
    R0_upper_bound = np.asarray([20])  # 1*20

    upper_bounds = np.hstack(
        (weight_upper_bound, Vg_upper_bound, Wg_upper_bound, R0_upper_bound)
    )  # stacking them into a single array

    return lower_bounds, upper_bounds


def get_y_model_cross_section(E_cm: np.ndarray, *params: np.ndarray) -> np.ndarray:
    # Initialising y_model array with zeros
    # It will be the sum of N gaussian models
    y_model = np.zeros_like(E_cm)

    # Unpacking paramters for optimisation
    num_gauss, constrained_weight, Vg, Wg, R0 = unpack_params_cross_section(params)

    # Running the loop for N gaussians to get y_model
    for i in range(num_gauss):
        y_model += models.cross_section_model(
            weight=constrained_weight[i], Vg=Vg[i], Wg=Wg[i], Rg=R0, E_cm=E_cm
        )

    return y_model


# def get_y_model_bd(E_bd: np.ndarray, *params: np.ndarray) -> np.ndarray:
#     y_model = np.zeros_like(E_bd)

#     num_gauss = int(len(params) / 3)
#     weight = np.asarray(params[:num_gauss])
#     Vg = np.asarray(params[num_gauss : 2 * num_gauss])
#     Wg = np.asarray(params[2 * num_gauss : 3 * num_gauss])
#     constrained_weight = weight / np.sum(weight)  #  constraint on weights

#     for i in range(num_gauss):
#         bd_model_obj = BarrierDistributionModel(
#             weight=constrained_weight[i], Vg=Vg[i], Wg=Wg[i]
#         )
#         y_model += bd_model_obj.expression_bd(E_bd)
#         # print(y_model)
#     return y_model


# def get_init_params_bd(N):

#     weight = 1 / N * np.ones(N)

#     Vg = np.random.default_rng().uniform(20, 100, size=N)
#     Wg = np.random.default_rng().uniform(0, 5, size=N)

#     return np.hstack((weight, Vg, Wg))


# def optimize_bd(x, y, y_err, max_num_gauss=5):

#     num_gauss = np.arange(1, max_num_gauss + 1)
#     chi2 = np.empty(max_num_gauss, dtype=float)

#     lower_bounds = 0

#     for ng in num_gauss:
#         weight_upper_bound = np.ones(ng)
#         Vg_upper_bound = 200 * np.ones(ng)
#         Wg_upper_bound = 10 * np.ones(ng)
#         upper_bounds = np.hstack((weight_upper_bound, Vg_upper_bound, Wg_upper_bound))

#         init_params = get_init_params_bd(ng)

#         popt, pcov = curve_fit(
#             get_y_model_bd,
#             x,
#             y,
#             p0=init_params,
#             sigma=y_err,
#             bounds=(lower_bounds, upper_bounds),
#         )
#         print(popt)
#         chi2[ng - 1] = get_chi0_2(x, y, y_err, get_y_model_bd(x, *popt))
#         print(get_y_model_bd(x, *popt))
#         print(chi2[ng - 1])

#     print(num_gauss, chi2)
#     plt.plot(num_gauss, chi2)
#     plt.show()


def optimize_cross_section(x, y, y_err, max_num_gauss=5):
    """Function to run the optimisation for fusion cross section."""

    full_result = {}

    # Array conataining number of gaussians in the model.
    # Currently num_gauss = [1,2,3,4,5]
    num_gauss = np.arange(1, max_num_gauss + 1)

    # Array to contain chi_0_sq values for each N gaussians
    chi_0_2 = np.empty(max_num_gauss, dtype=float)
    chi_2 = np.empty(max_num_gauss, dtype=float)

    for ng in num_gauss:

        lower_bounds, upper_bounds = get_parameter_bounds_cross_section(ng)
        init_params = get_init_params_cross_section(ng)

        popt, pcov = curve_fit(
            get_y_model_cross_section,
            x,
            y,
            p0=init_params,  # Converting numpy array to python list
            sigma=y_err,
            bounds=(lower_bounds, upper_bounds),
        )
        # print(popt)

        y_model = get_y_model_cross_section(x, *popt)
        _, weight_opt, Vg_opt, Wg_opt, R0_opt = unpack_params_cross_section(popt)

        chi_0_2[ng - 1] = get_chi0_2(x, y, y_err, y_model)
        chi_2[ng - 1] = len(y) * chi_0_2[ng - 1] / (len(y) - (3 * ng + 1))

        full_result[ng] = {
            "chi_0_2": chi_0_2[ng - 1],
            "chi_2": chi_2[ng - 1],
            "weight": weight_opt,
            "Vg": Vg_opt,
            "Wg": Wg_opt,
            "R0": R0_opt,
            "y_model": y_model,
        }

        # print(f"y_model = {get_y_model(x,*popt)}")
        # print(chi_0_2[ng - 1])

    print(num_gauss, chi_0_2)
    plt.plot(num_gauss, chi_0_2, label="chi_0_sq")
    plt.plot(num_gauss, chi_2, label="chi_sq")
    plt.legend()
    plt.xlabel("No. of gaussians")
    plt.show()

    df = pd.DataFrame.from_dict(full_result, orient="index")

    # Save the DataFrame as an Excel file
    df.to_csv("results.csv", index_label="N")

    opt_num_gauss = num_gauss[np.argmin(chi_2)]
    print(f"Optimum number of gaussians = {opt_num_gauss}")
    print("Optimum parameter values:")

    for key, value in full_result[opt_num_gauss].items():
        if key != "y_model":
            print(f"{key}={value}")
        else:
            np.savetxt(
                "cs_data.dat",
                np.column_stack((x, value)),
                fmt=("%.3f", "%.3e"),
                delimiter="\t",
            )
            print('The model cross sections are stored in cs_data.dat')

    
