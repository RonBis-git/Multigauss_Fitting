import numpy as np
from scipy.optimize import curve_fit

def read_file():
    fname = '/home/rohan/sn/32S_89Y.dat'
    return np.loadtxt(fname, unpack=True)

def main():
    E_cm, f_xs, err_f_xs = read_file()

if __name__ == '__main__':
    main()