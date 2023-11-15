# this contains the FO massive coefficients functions.

import LeProHQ
import numpy as np
from eko.constants import TR
from scipy.integrate import quad

from . import Initialize, scale_variations


# F2
def Cb_2_m_reg(z, Q, p, _nf, mur_ratio=1.0):
    Q2 = Q * Q
    m_b = p[0]
    eps = m_b * m_b / Q2
    thre = 4.0 * eps * z / (1 - z)
    if thre > 1.0:
        return 0
    e_h = p[-1]
    xi = 1 / eps
    eta = xi / 4.0 * (1.0 / z - 1.0) - 1.0
    eta = min(eta, 1e5)
    FHprefactor = Q2 / (np.pi * m_b**2) * e_h**2
    bare_res = FHprefactor / z * (4.0 * np.pi) ** 2 * LeProHQ.dq1("F2", "VV", xi, eta)
    return scale_variations.apply_rensv_kernel(0, 2, [bare_res], mur_ratio, _nf)


def Cb_2_m_loc(_z, Q, p, _nf, mur_ratio=1.0):
    l = quad(
        lambda x: Cb_2_m_reg(x, Q, p, _nf),
        0.0,
        1.0,
        points=(0.0, 1.0),
    )
    bare_res = -l[0]
    return scale_variations.apply_rensv_kernel(0, 2, [bare_res], mur_ratio, _nf)


def Cg_1_m_reg(z, Q, p, _nf, mur_ratio=1.0):
    Q2 = Q * Q
    m_b = p[0]
    eps = m_b * m_b / Q2
    thre = 4.0 * eps * z / (1 - z)
    e_h = p[-1]
    if thre > 1.0:
        return 0
    v = np.sqrt(1 - thre)
    bare_res = (
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
    return scale_variations.apply_rensv_kernel(0, 1, [bare_res], mur_ratio, _nf)


def Cg_2_m_reg(z, Q, p, _nf, mur_ratio=1.0):
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
    bare_res = (
        FHprefactor
        / z
        * (4.0 * np.pi) ** 2
        * (
            LeProHQ.cg1("F2", "VV", xi, eta)
            + LeProHQ.cgBar1("F2", "VV", xi, eta) * np.log(xi)
        )
    )
    return scale_variations.apply_rensv_kernel(
        1, 1, [bare_res, Cg_1_m_reg(z, Q, p, _nf, mur_ratio=1.0)], mur_ratio, _nf
    )


def Cg_3_m_reg(z, Q, p, nf, mur_ratio=1.0):
    e_h = p[-1]
    Q2 = Q * Q
    m_b = p[0]
    eps = m_b * m_b / Q2
    thre = 4.0 * eps * z / (1 - z)
    if thre > 1.0:
        return 0.0
    bare_res = e_h**2 * Initialize.Cg3m[nf - 4](z, Q)[0]
    return scale_variations.apply_rensv_kernel(
        2,
        1,
        [
            bare_res,
            Cg_2_m_reg(z, Q, p, nf, mur_ratio=1.0),
            Cg_1_m_reg(z, Q, p, nf, mur_ratio=1.0),
        ],
        mur_ratio,
    )


def Cq_2_m_reg(z, Q, p, _nf, mur_ratio=1.0):
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
    bare_res = (
        FHprefactor
        / z
        * (4.0 * np.pi) ** 2
        * (
            LeProHQ.cq1("F2", "VV", xi, eta)
            + LeProHQ.cqBarF1("F2", "VV", xi, eta) * np.log(xi)
        )
    )
    return scale_variations.apply_rensv_kernel(0, 2, [bare_res], mur_ratio, _nf)


def Cq_3_m_reg(z, Q, p, nf, mur_ratio=1.0):
    e_h = p[-1]
    Q2 = Q * Q
    m_b = p[0]
    eps = m_b * m_b / Q2
    thre = 4.0 * eps * z / (1 - z)
    if thre > 1.0:
        return 0.0
    bare_res = e_h**2 * Initialize.Cq3m[nf - 4](z, Q)[0]
    return scale_variations.apply_rensv_kernel(
        1, 2, [bare_res, Cq_2_m_reg(z, Q, p, nf, mur_ratio=1.0)], mur_ratio, nf
    )


# FL
def CLb_2_m_reg(z, Q, p, _nf, mur_ratio=1.0):
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
    bare_res = FHprefactor / z * (4.0 * np.pi) ** 2 * LeProHQ.dq1("FL", "VV", xi, eta)
    return scale_variations.apply_rensv_kernel(0, 2, [bare_res], mur_ratio, _nf)


def CLg_1_m_reg(z, Q, p, _nf, mur_ratio=1.0):
    Q2 = Q * Q
    m_b = p[0]
    eps = m_b * m_b / Q2
    z2 = z * z
    thre = 4.0 * eps * z / (1 - z)
    e_h = p[-1]
    if thre > 1.0:
        return 0
    v = np.sqrt(1 - thre)
    bare_res = (
        4
        * TR
        * e_h
        * e_h
        * (-8 * eps * z2 * np.log((1 + v) / (1 - v)) + 4 * v * z * (1 - z))
    )
    return scale_variations.apply_rensv_kernel(0, 1, [bare_res], mur_ratio, _nf)


def CLg_2_m_reg(z, Q, p, _nf, mur_ratio=1.0):
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
    bare_res = (
        FHprefactor
        / z
        * (4.0 * np.pi) ** 2
        * (
            LeProHQ.cg1("FL", "VV", xi, eta)
            + LeProHQ.cgBar1("FL", "VV", xi, eta) * np.log(xi)
        )
    )
    return scale_variations.apply_rensv_kernel(
        1, 1, [bare_res, CLg_1_m_reg(z, Q, p, _nf, mur_ratio=1.0)], mur_ratio, _nf
    )


def CLg_3_m_reg(z, Q, p, nf, mur_ratio=1.0):
    e_h = p[-1]
    Q2 = Q * Q
    m_b = p[0]
    eps = m_b * m_b / Q2
    thre = 4.0 * eps * z / (1 - z)
    if thre > 1.0:
        return 0
    bare_res = e_h**2 * Initialize.CLg3m[nf - 4](z, Q)[0]
    return scale_variations.apply_rensv_kernel(
        2,
        1,
        [
            bare_res,
            CLg_2_m_reg(z, Q, p, nf, mur_ratio=1.0),
            CLg_1_m_reg(z, Q, p, nf, mur_ratio=1.0),
        ],
        mur_ratio,
        nf,
    )


def CLq_2_m_reg(z, Q, p, _nf, mur_ratio=1.0):
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
    bare_res = (
        FHprefactor
        / z
        * (4.0 * np.pi) ** 2
        * (
            LeProHQ.cq1("FL", "VV", xi, eta)
            + LeProHQ.cqBarF1("FL", "VV", xi, eta) * np.log(xi)
        )
    )
    return scale_variations.apply_rensv_kernel(0, 2, [bare_res], mur_ratio, _nf)


def CLq_3_m_reg(z, Q, p, nf, mur_ratio=1.0):
    e_h = p[-1]
    Q2 = Q * Q
    m_b = p[0]
    eps = m_b * m_b / Q2
    thre = 4.0 * eps * z / (1 - z)
    if thre > 1.0:
        return 0.0
    bare_res = e_h**2 * Initialize.CLq3m[nf - 4](z, Q)[0]
    return scale_variations.apply_rensv_kernel(
        1, 2, [bare_res, CLq_2_m_reg(z, Q, p, nf, mur_ratio=1.0)], mur_ratio, nf
    )
