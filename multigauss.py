import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc


def read_file():
    fname = "/home/rohan/sn/32S_89Y.dat"
    return np.loadtxt(fname, unpack=True)


def get_Z(E_cm: np.ndarray, Vg: float, Wg: float) -> np.ndarray:
    return (E_cm - Vg) / np.sqrt(2.0) / Wg


def expression_xs(E_cm, weight, Vg, Wg):
    # calculating Z
    Z = get_Z(E_cm, Vg, Wg)

    # calculating terms for the cross-section
    bracket_term1 = np.sqrt(np.pi) * Z * erfc(-Z)
    bracket_term2 = np.exp(-Z * Z)
    outside_bracket = weight * Wg / np.sqrt(2 * np.pi) / E_cm

    return outside_bracket * (bracket_term1 + bracket_term2)


def chi2(x, y_expt, y_err, y_model):
    return np.sum((y_expt - y_model) ** 2 / y_err**2) / len(x)


def get_y_model(
    E_cm: np.ndarray,
    num_gauss: int,
    weight: np.ndarray,
    Vg: np.ndarray,
    Wg: np.ndarray,
    R0,
) -> np.ndarray:
    y_model = np.zeros_like(E_cm)
    for i in range(num_gauss):
        y_model += (
            10
            * np.pi
            * R0**2
            * expression_xs(E_cm, weight=weight[i], Vg=Vg[i], Wg=Wg[i])
        )
    return y_model


def main():
    E_cm, f_xs, err_f_xs = read_file()


if __name__ == "__main__":
    main()
