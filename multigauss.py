import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc

from optimiser import optimize_xs, optimize_bd
from models import BarrierDistribution


def read_file():
    fname = "/home/rohan/sn/32S_89Y.dat"
    return np.loadtxt(fname, unpack=True)


def main():
    E_cm, f_xs, err_f_xs = read_file()
    optimize_xs(E_cm,f_xs, err_f_xs)
    E_bd, fbd, fbderr = BarrierDistribution.get_BD_from_xs(E_cm,f_xs,err_f_xs)
    optimize_bd(E_bd,fbd, fbderr)


if __name__ == "__main__":
    main()
