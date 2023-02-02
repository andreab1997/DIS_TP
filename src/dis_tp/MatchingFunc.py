# This contains the matching conditions needed for the evaluation of the tilde coefficients functions.
# notation: p[1] is Q^2 while p[0] is m_b^2
import numpy as np
from eko.constants import CA, CF, TR
from eko.harmonics.constants import zeta2, zeta3, zeta4, zeta5
from eko.matching_conditions import as2, as3

from . import Harmonics as harm
from . import Initialize as Ini
from . import parameters
from .InverseMellin import inverse_mellin


def Mbg_1(z, p, nf):
    return 2 * TR * np.log((p[1] ** 2) / (p[0] ** 2)) * (z * z + (1 - z) * (1 - z))


# NOTE: here there is a different convention on the use of
#  matching conditions A_Hq and A_hg wrt EKO
def Mbg_2(z, p, nf, r=None, s=None, path="talbot"):
    if parameters.grids:
        return Ini.Mbg2(z, p[1])[0]
    L = np.log((p[1] ** 2) / (p[0] ** 2))
    return 0.5 * inverse_mellin(as2.A_hg, z, nf, r, s, path, L)


def Mbg_3_reg(z, p, nf):
    if parameters.grids:
        return Ini.Mbg3(z, p[1])[0]


def Mgg_1_loc(z, p, nf):
    return -(4.0 / 3.0) * TR * np.log((p[1] ** 2) / (p[0] ** 2))


def Mgg_2_reg(z, p, nf):
    L = np.log((p[1] ** 2) / (p[0] ** 2))
    z2 = z**2
    lnz = np.log(z)
    lnz2 = lnz**2
    A01 = (
        4.0 * (1.0 + z) * lnz**3 / 3.0
        + (6.0 + 10.0 * z) * lnz2
        + (32.0 + 48.0 * z) * lnz
        - 8.0 / z
        + 80.0
        - 48.0 * z
        - 24 * z2
    )
    A02 = (
        4.0 * (1.0 + z) * lnz2 / 3.0
        + (52.0 + 88.0 * z) * lnz / 9.0
        - 4.0 * z * np.log(1.0 - z) / 3.0
        + (556.0 / z - 628.0 + 548.0 * z - 700.0 * z2) / 27.0
    )
    AS2ggH_R = TR * (CF * A01 + CA * A02)
    omeL1 = -(
        CF
        * TR
        * (
            8.0 * (1.0 + z) * lnz2
            + (24.0 + 40.0 * z) * lnz
            - 16.0 / 3.0 / z
            + 64.0
            - 32.0 * z
            - 80.0 * z2 / 3.0
        )
        + CA
        * TR
        * (
            16.0 * (1.0 + z) * lnz / 3.0
            + 184.0 / 9.0 / z
            - 232.0 / 9.0
            + 152.0 * z / 9.0
            - 184.0 * z2 / 9.0
        )
    )
    omeL2 = CF * TR * (
        8.0 * (1.0 + z) * lnz + 16.0 / 3.0 / z + 4.0 - 4.0 * z - 16.0 * z2 / 3.0
    ) + CA * TR * (8.0 / 3.0 / z - 16.0 / 3.0 + 8.0 * z / 3.0 - 8.0 * z2 / 3.0)
    return AS2ggH_R + L * omeL1 + L**2 * omeL2


def Mgg_2_loc(z, p, nf):
    L = np.log((p[1] ** 2) / (p[0] ** 2))
    AS2ggH_L = TR * (-CF * 15.0 + CA * 10.0 / 9.0)
    omeL1 = -TR * (CF * 4.0 + CA * 16.0 / 3.0)
    omeL2 = TR**2 * 16.0 / 9.0
    return AS2ggH_L + L * omeL1 + L**2 * omeL2


def Mgg_2_sing(z, p, nf):
    L = np.log((p[1] ** 2) / (p[0] ** 2))
    z1 = 1 - z
    AS2ggH_S = 224.0 / 27.0 / z1
    omeL1 = -80.0 / 9.0 / z1
    omeL2 = 8.0 / 3.0 / z1
    return CA * TR * (AS2ggH_S + L**2 * omeL2 + L * omeL1)


def Mbq_2(z, p, nf, r=None, s=None, path="talbot"):
    if parameters.grids:
        return Ini.Mbq2(z, p[1])[0]
    L = np.log((p[1] ** 2) / (p[0] ** 2))
    return 0.5 * inverse_mellin(as2.A_hq_ps, z, nf, r, s, path, L)


def Mbq_3_reg(z, p, nf):
    if parameters.grids:
        return Ini.Mbq3(z, p[1])[0]


def Mgq_2_reg(z, p, nf):
    L = np.log((p[1] ** 2) / (p[0] ** 2))
    z1 = 1 - z
    L1 = np.log(z1)
    return (
        CF
        * TR
        * (
            ((16.0 / (3 * z)) - (16.0 / 3.0) + z * (8.0 / 3.0)) * (L**2)
            - (
                (160.0 / (9 * z))
                - (160.0 / 9.0)
                + z * (128.0 / 9.0)
                + L1 * ((32.0 / (3 * z)) - (32.0 / 3.0) + z * (16.0 / 3.0))
            )
            * L
            + (4.0 / 3.0) * ((2.0 / z) - 2 + z) * (L1**2)
            + (8.0 / 9.0) * ((10.0 / z) - 10 + 8 * z) * L1
            + (1.0 / 27.0) * ((448.0 / z) - 448 + 344 * z)
        )
    )


def Mbg_3_reg_inv(x, p, nf, r=None, s=None, path="talbot"):
    if parameters.grids:
        return Ini.Mbg3(x, p[1])[0]
    L = np.log((p[1] ** 2) / (p[0] ** 2))
    return 0.5 * inverse_mellin(as3.A_Hg, x, nf, r, s, path, L)


def Mbq_3_reg_inv(x, p, nf, r=None, s=None, path="talbot"):
    if parameters.grids:
        return Ini.Mbq3(x, p[1])[0]
    L = np.log((p[1] ** 2) / (p[0] ** 2))
    return 0.5 * inverse_mellin(as3.A_Hq, x, nf, r, s, path, L)


def P1(p, nf):
    return -Mgg_1_loc(0, p, nf)


def P2(p):
    fact = np.log((p[1] ** 2) / (p[0] ** 2))
    return (2.0 / 9.0) * (2 * (fact**2) + 19 * 3 * fact + 7 * 3)  # Thanks EKO
