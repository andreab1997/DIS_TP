# this contains the FO massive coefficients functions.

import LeProHQ
import numpy as np
from eko.constants import TR

from . import Initialize, parameters


# F2
def Cg_1_m_reg(z, Q, p, nf):
    Q2 = Q * Q
    m_b = p[0]
    eps = m_b * m_b / Q2
    thre = 4.0 * eps * z / (1 - z)
    e_h = p[-1]
    if thre > 1.0:
        return 0
    v = np.sqrt(1 - thre)
    return (
        4
        * TR
        * e_h
        * e_h
        * (
            v * (8 * z * (1 - z) - 1 - 4 * z * (1 - z) * eps)
            + np.log((1 + v) / (1 - v))
            * (z * z + (1 - z) ** 2 + 4 * z * eps * (1 - 3 * z) - 8 * z * z * eps * eps)
        )
    )


def Cg_2_m_reg(z, Q, p, nf):
    Q2 = Q * Q
    m_b = p[0]
    eps = m_b**2 / Q2
    thre = 4.0 * eps * z / (1 - z)
    if thre > 1.0:
        return 0
    e_h = p[-1]
    xi = 1 / eps
    eta = xi / 4.0 * (1.0 / z - 1.0) - 1.0
    FHprefactor = Q2 / (np.pi * m_b**2) * e_h**2
    if xi > 2499.9999999999995:
        # FH grids are not defined above this
        return 0.0
    return (
        FHprefactor
        / z
        * (4.0 * np.pi) ** 2
        * (
            LeProHQ.cg1("F2", "VV", xi, eta)
            + LeProHQ.cgBar1("F2", "VV", xi, eta) * np.log(xi)
        )
    )


def Cg_3_m_reg(z, Q, p, nf):
    Q2 = Q * Q
    m_b = p[0]
    eps = m_b * m_b / Q2
    thre = 4.0 * eps * z / (1 - z)
    if thre > 1.0:
        return 0.0
    return Initialize.Cg3m(z, Q)[0]


def Cq_2_m_reg(z, Q, p, nf):
    Q2 = Q * Q
    m_b = p[0]
    eps = m_b * m_b / Q2
    thre = 4.0 * eps * z / (1 - z)
    if thre > 1.0:
        return 0
    e_h = p[-1]
    xi = 1 / eps
    eta = xi / 4.0 * (1.0 / z - 1.0) - 1.0
    FHprefactor = Q2 / (np.pi * m_b**2) * e_h**2
    return (
        FHprefactor
        / z
        * (4.0 * np.pi) ** 2
        * (
            LeProHQ.cq1("F2", "VV", xi, eta)
            + LeProHQ.cqBarF1("F2", "VV", xi, eta) * np.log(xi)
        )
    )


def Cq_3_m_reg(z, Q, p, nf):
    Q2 = Q * Q
    m_b = p[0]
    eps = m_b * m_b / Q2
    thre = 4.0 * eps * z / (1 - z)
    if thre > 1.0:
        return 0.0
    return Initialize.Cq3m(z, Q)[0]


# FL
def CLg_1_m_reg(z, Q, p, nf):
    Q2 = Q * Q
    m_b = p[0]
    eps = m_b * m_b / Q2
    z2 = z * z
    thre = 4.0 * eps * z / (1 - z)
    v = np.sqrt(1 - thre)
    e_h = p[-1]
    if thre > 1.0:
        return 0
    return (
        4
        * TR
        * e_h
        * e_h
        * (-8 * eps * z2 * np.log((1 + v) / (1 - v)) + 4 * v * z * (1 - z))
    )


def CLg_2_m_reg(z, Q, p, nf):
    Q2 = Q * Q
    m_b = p[0]
    eps = m_b * m_b / Q2
    thre = 4.0 * eps * z / (1 - z)
    if thre > 1.0:
        return 0
    e_h = p[-1]
    xi = 1 / eps
    eta = xi / 4.0 * (1.0 / z - 1.0) - 1.0
    FHprefactor = Q2 / (np.pi * m_b**2) * e_h**2
    if xi > 2499.9999999999995:
        # FH grids are not defined above this
        return 0.0
    return (
        FHprefactor
        / z
        * (4.0 * np.pi) ** 2
        * (
            LeProHQ.cg1("FL", "VV", xi, eta)
            + LeProHQ.cgBar1("FL", "VV", xi, eta) * np.log(xi)
        )
    )


def CLg_3_m_reg(z, Q, p, nf):
    Q2 = Q * Q
    m_b = p[0]
    eps = m_b * m_b / Q2
    thre = 4.0 * eps * z / (1 - z)
    if thre > 1.0:
        return 0
    return Initialize.CLg3m(z, Q)[0]


def CLq_2_m_reg(z, Q, p, nf):
    Q2 = Q * Q
    m_b = p[0]
    eps = m_b * m_b / Q2
    thre = 4.0 * eps * z / (1 - z)
    if thre > 1.0:
        return 0
    e_h = p[-1]
    xi = 1 / eps
    eta = xi / 4.0 * (1.0 / z - 1.0) - 1.0
    FHprefactor = Q2 / (np.pi * m_b**2) * e_h**2
    return (
        FHprefactor
        / z
        * (4.0 * np.pi) ** 2
        * (
            LeProHQ.cq1("FL", "VV", xi, eta)
            + LeProHQ.cqBarF1("FL", "VV", xi, eta) * np.log(xi)
        )
    )


def CLq_3_m_reg(z, Q, p, nf):
    Q2 = Q * Q
    m_b = p[0]
    eps = m_b * m_b / Q2
    thre = 4.0 * eps * z / (1 - z)
    if thre > 1.0:
        return 0.0
    return Initialize.CLq3m(z, Q)[0]
