# This contains the tilde coefficents functions for the matched scheme.

import numpy as np
import scipy.special as special

from eko.constants import TR, CF
from . import Initialize as Ini
from .MassiveCoeffFunc import (
    Cg_1_m_reg,
    Cg_2_m_reg,
    Cg_3_m_reg,
    CLg_1_m_reg,
    CLg_2_m_reg,
    CLg_3_m_reg,
    CLq_2_m_reg,
    CLq_3_m_reg,
    Cq_2_m_reg,
    Cq_3_m_reg,
)
from .MasslessCoeffFunc import (
    Cb_0_loc,
    Cb_1_loc,
    Cb_1_reg,
    Cb_1_sing,
    Cb_2_loc,
    Cb_2_reg,
    Cb_2_sing,
    CLb_1_reg,
    CLb_2_loc,
    CLb_2_reg,
)
from .MatchingFunc import (
    P1,
    P2,
    Mbg_1,
    Mbg_2,
    Mbg_3_reg,
    Mbq_2,
    Mbq_3_reg,
    Mgg_1_loc,
    Mgg_2_loc,
    Mgg_2_reg,
    Mgg_2_sing,
    Mgq_2_reg,
)
from .tools import (
    Convolute,
    Convolute_matching,
    Convolute_plus_coeff,
    Convolute_plus_matching,
    Convolute_plus_matching_per_matching,
)


def Cb1_Mbg1(z, p, nf):
    e_h = p[-1]
    return (
        4
        * CF
        * TR
        * pow(e_h, 2)
        * (
            -(5.0 / 2.0)
            + 2 * z * (3 - 4 * z)
            + (pow(np.pi, 2) / 6.0) * (-1 + 2 * z - 4 * pow(z, 2))
            + pow(np.log(1 - z), 2) * (1 - 2 * z * (1 - z))
            - (1.0 / 2.0)
            * np.log(1 - z)
            * (7 + 4 * z * (3 * z - 4) + (4 - 8 * z * (1 - z)) * np.log(z))
            + (1.0 / 2.0)
            * np.log(z)
            * (-1 + 4 * z * (3 * z - 2) + (1 - 2 * z + 4 * pow(z, 2)) * np.log(z))
            + (2 * z - 1) * special.spence(z)
        )
    )


def CLb1_Mbg1(z, p, nf):
    e_h = p[-1]
    return 8 * CF * TR * pow(e_h, 2) * (1 + z - 2 * pow(z, 2) + 2 * z * np.log(z))


# F2
def Cg_1_til_reg(z, Q, p, nf):
    return Cg_1_m_reg(z, Q, p, nf) - 2 * Cb_0_loc(z, Q, p, nf) * Mbg_1(z, p, nf)


def Cg_2_til_reg(z, Q, p, nf, grids=True):
    if grids:
        return Ini.Cg2_til(z, Q)[0]
    return (
        Cg_2_m_reg(z, Q, p, nf)
        - 2
        * Cb_0_loc(z, Q, p, nf)
        * (Mbg_2(z, p, nf) - Mbg_1(z, p, nf) * Mgg_1_loc(z, p, nf))
        - 2 * np.log((Q**2) / (p[0] ** 2)) * Cb1_Mbg1(z, p, nf)
    )


def Cg_3_til_reg(z, Q, p, nf, grids=False):
    if grids:
        return Ini.Cg3_til(z, Q)[0]
    return (
        Cg_3_m_reg(z, Q, p, nf)
        + Cg_2_m_reg(z, Q, p, nf) * Mgg_1_loc(z, p, nf)
        + P2(p) * Cg_1_m_reg(z, Q, p, nf)
        - (
            Cg_1_m_reg(z, Q, p, nf) * Mgg_2_loc(z, p, nf)
            + Convolute(Cg_1_m_reg, Mgg_2_reg, z, Q, p, nf)
            + Convolute_plus_matching(Cg_1_m_reg, Mgg_2_sing, z, Q, p, nf)
        )
        - 2
        * Cb_0_loc(z, Q, p, nf)
        * (
            Mbg_3_reg(z, p, nf)
            - Mgg_1_loc(z, p, nf) * Mbg_2(z, p, nf)
            + Mbg_1(z, p, nf) * Mgg_1_loc(z, p, nf) * Mgg_1_loc(z, p, nf)
            - (
                Mbg_1(z, p, nf) * Mgg_2_loc(z, p, nf)
                + Convolute_matching(Mbg_1, Mgg_2_reg, z, Q, p, nf)
                + Convolute_plus_matching_per_matching(Mgg_2_sing, Mbg_1, z, Q, p, nf)
            )
        )
        - 2
        * (
            Cb_1_loc(z, Q, p, nf) * Mbg_2(z, p, nf)
            + Convolute(Cb_1_reg, Mbg_2, z, Q, p, nf)
            + Convolute_plus_coeff(Cb_1_sing, Mbg_2, z, Q, p, nf)
            - Cb1_Mbg1(z, p, nf) * Mgg_1_loc(z, p, nf)
        )
        - 2
        * (
            Mbg_1(z, p, nf) * Cb_2_loc(z, Q, p, nf)
            + Convolute(Cb_2_reg, Mbg_1, z, Q, p, nf)
            + Convolute_plus_coeff(Cb_2_sing, Mbg_1, z, Q, p, nf)
        )
    )


def Cq_2_til_reg(z, Q, p, nf, grids=True):
    if grids:
        return Ini.Cq2_til(z, Q)[0]
    return Cq_2_m_reg(z, Q, p, nf) - 2 * Cb_0_loc(z, Q, p, nf) * Mbq_2(z, p, nf)


def Cq_3_til_reg(z, Q, p, nf, grids=False):
    q = [p[0], Q]
    if grids:
        return Ini.Cq3_til(z, Q)[0]
    return (
        Cq_3_m_reg(z, Q, p, nf)
        + 2 * Cq_2_m_reg(z, Q, p, nf) * Mgg_1_loc(z, p, nf)
        - Convolute(Cg_1_m_reg, Mgq_2_reg, z, Q, p, nf)
        - 2
        * (
            Cb_1_loc(z, Q, p, nf) * Mbq_2(z, p, nf)
            + Convolute(Cb_1_reg, Mbq_2, z, Q, p, nf)
            + Convolute_plus_coeff(Cb_1_sing, Mbq_2, z, Q, p, nf)
        )
        - 2 * (Cb_0_loc(z, Q, p, nf) * Mbq_3_reg(z, p, nf))
    )


# FL
def CLg_1_til_reg(z, Q, p, nf):
    return CLg_1_m_reg(z, Q, p, nf)


def CLg_2_til_reg(z, Q, p, nf, grids=True):
    if grids:
        return Ini.CLg2_til(z, Q)[0]
    return CLg_2_m_reg(z, Q, p, nf) - 2 * np.log((Q**2) / (p[0] ** 2)) * CLb1_Mbg1(
        z, p, nf
    )


def CLg_3_til_reg(z, Q, p, nf, grids=False):
    if grids:
        return Ini.CLg3_til(z, Q)[0]
    return (
        CLg_3_m_reg(z, Q, p, nf)
        + CLg_2_m_reg(z, Q, p, nf) * Mgg_1_loc(z, p, nf)
        + P2(p) * CLg_1_m_reg(z, Q, p, nf)
        - (
            CLg_1_m_reg(z, Q, p, nf) * Mgg_2_loc(z, p, nf)
            + Convolute(CLg_1_m_reg, Mgg_2_reg, z, Q, p, nf)
            + Convolute_plus_matching(CLg_1_m_reg, Mgg_2_sing, z, Q, p, nf)
        )
        - 2
        * (
            Convolute(CLb_1_reg, Mbg_2, z, Q, p, nf)
            - CLb1_Mbg1(z, p, nf) * Mgg_1_loc(z, p, nf)
        )
        - 2
        * (
            CLb_2_loc(z, Q, p, nf) * Mbg_1(z, p, nf)
            + Convolute(CLb_2_reg, Mbg_1, z, Q, p, nf)
        )
    )


def CLq_2_til_reg(z, Q, p, nf, grids=True):
    if grids:
        return Ini.CLq2_til(z, Q)[0]
    return CLq_2_m_reg(z, Q, p, nf)


def CLq_3_til_reg(z, Q, p, nf, grids=False):
    if grids:
        return Ini.CLq3_til(z, Q)[0]
    return (
        CLq_3_m_reg(z, Q, p, nf)
        + 2 * CLq_2_m_reg(z, Q, p, nf) * Mgg_1_loc(z, p, nf)
        - Convolute(CLg_1_m_reg, Mgq_2_reg, z, Q, p, nf)
        - 2 * Convolute(CLb_1_reg, Mbq_2, z, Q, p, nf)
    )
