import numpy as np
from scipy import integrate

def talbot_path(theta, r, s):
    return s + r * complex(theta / np.tan(theta), theta)

def talbot_jac(theta, r):
    sigma = theta + (theta / np.tan(theta) - 1) / np.tan(theta)
    return r * complex(1, sigma)

def quad_ker(theta, func, x, nf, r, s):
    n = talbot_path(theta, r, s)
    jac = talbot_jac(theta, r)
    return np.real(np.exp(- np.log(x) * n) * func(n, nf) * jac)

def inverse_mellin(func, x, nf, r, s):
    return integrate.quad(quad_ker, 0.0, np.pi, args=(func, x, nf, r, s))[0] / np.pi