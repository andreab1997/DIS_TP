import numpy as np
from scipy import integrate


def talbot_path(theta, r, s):
    return s + r * complex(theta * np.cos(theta) / np.sin(theta), theta)


def check_path(r, s):
    if r + s < 1.0:
        raise ValueError(
            "r + s must be greater than 1 (i.e. than the real part of the rightmost pole)"
        )


def talbot_jac(theta, r):
    cot = np.cos(theta) / np.sin(theta)
    sigma = theta + (theta * cot - 1) * cot
    return r * complex(1, sigma)


def quad_ker_talbot(theta, func, x, nf, r, s):
    n = talbot_path(theta, r, s)
    jac = talbot_jac(theta, r)
    return np.real(np.exp(-np.log(x) * n) * func(n, nf) * jac)


def inverse_mellin_talbot(func, x, nf, r, s):
    if r is None:
        r = 0.4 * 16.0 / (1.0 - np.log(x))
    if s is None:
        s = 1.0
    check_path(r, s)
    return (
        integrate.quad(quad_ker_talbot, 0.0, np.pi, args=(func, x, nf, r, s))[0] / np.pi
    )


# Implement also the linear path for benchmarking the Talbot path


def path_linear(t, r, s):
    return complex(s - r * t, t)


def jac_linear(r):
    return complex(-r, 1)


def quad_ker_linear(t, func, x, nf, r, s):
    n = path_linear(t, r, s)
    jac = jac_linear(r)
    return np.imag(np.exp(-np.log(x) * n) * func(n, nf) * jac)


def inverse_mellin_linear(func, x, nf, r, s):
    if r is None:
        r = 1.0
    if s is None:
        s = 1.5
    check_path(0, s)
    return (
        integrate.quad(quad_ker_linear, 0.0, np.inf, args=(func, x, nf, r, s))[0]
        / np.pi
    )


def inverse_mellin(func, x, nf, r, s, path):
    if path == "talbot":
        return inverse_mellin_talbot(func, x, nf, r, s)
    elif path == "linear":
        return inverse_mellin_linear(func, x, nf, r, s)
    else:
        raise NotImplementedError(
            "Selected path is not implemented: choose either 'talbot' or 'linear'"
        )
