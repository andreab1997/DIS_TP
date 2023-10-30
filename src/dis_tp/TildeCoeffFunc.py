# This contains the tilde coefficents functions for the matched scheme.

import numpy as np
import scipy.special as special
from eko.constants import CA, CF, TR

from . import Initialize as Ini
from . import parameters, scale_variations
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
from .structure_functions.heavy_tools import (
    Convolute,
    Convolute_matching,
    Convolute_plus_coeff,
    Convolute_plus_matching,
)


# convolutions
def Cb1_Mbg1(z, p, _nf):
    e_h = p[-1]
    res = (
        4
        * CF
        * TR
        * e_h**2
        * (
            -(5.0 / 2.0)
            + 2 * z * (3 - 4 * z)
            + (np.pi**2 / 6.0) * (-1 + 2 * z - 4 * z**2)
            + np.log(1 - z) ** 2 * (1 - 2 * z * (1 - z))
            - (1.0 / 2.0)
            * np.log(1 - z)
            * (7 + 4 * z * (3 * z - 4) + (4 - 8 * z * (1 - z)) * np.log(z))
            + (1.0 / 2.0)
            * np.log(z)
            * (-1 + 4 * z * (3 * z - 2) + (1 - 2 * z + 4 * pow(z, 2)) * np.log(z))
            + (2 * z - 1) * special.spence(complex(1 - z))
        )
    )
    return np.real(res)


def CLb1_Mbg1(z, p, _nf):
    e_h = p[-1]
    return 8 * CF * TR * pow(e_h, 2) * (1 + z - 2 * pow(z, 2) + 2 * z * np.log(z))


def Mbg1_Mgg2_sing(x, p, _nf):
    L = np.log((p[1] ** 2) / (p[0] ** 2))
    return (
        16
        / 27
        * CA
        * L
        * (28 - 30 * L + 9 * L**2)
        * TR**2
        * (-1 + (4 - 3 * x) * x + (-1 - 2 * (-1 + x) * x) * np.log(x))
    )


# F2
def Cg_1_til_reg(z, Q, p, _nf, mur_ratio=1.0, muf_ratio=1.0):
    bare_res = Cg_1_m_reg(z, Q, p, _nf - 1) - 2 * Cb_0_loc(z, Q, p, _nf) * Mbg_1(
        z, p, _nf
    )
    return scale_variations.apply_sv_kernel(
        order=0,
        m=1,
        ingredients=[bare_res],
        z=z,
        Q=Q,
        p=p,
        nf=_nf,
        mur_ratio=mur_ratio,
        muf_ratio=muf_ratio,
    )


def Cg_2_til_reg(z, Q, p, _nf, mur_ratio=1.0, muf_ratio=1.0):
    bare_res = (
        Cg_2_m_reg(z, Q, p, _nf - 1)
        - 2
        * Cb_0_loc(z, Q, p, _nf)
        * (Mbg_2(z, p, _nf) - Mbg_1(z, p, _nf) * Mgg_1_loc(z, p, _nf))
        - 2 * np.log((Q**2) / (p[0] ** 2)) * Cb1_Mbg1(z, p, _nf)
    )
    return scale_variations.apply_sv_kernel(
        order=1,
        m=1,
        ingredients=[bare_res, Cg_1_til_reg],
        z=z,
        Q=Q,
        p=p,
        nf=_nf,
        mur_ratio=mur_ratio,
        muf_ratio=muf_ratio,
    )


def Cg_3_til_reg(z, Q, p, nf, use_analytic=False, mur_ratio=1.0, muf_ratio=1.0):
    if parameters.grids and not use_analytic:
        bare_res = Ini.Cg3_til[nf - 4](z, Q)[0]
        return scale_variations.apply_sv_kernel(
            order=2,
            m=1,
            ingredients=[bare_res, Cg_1_til_reg, Cg_2_til_reg],
            z=z,
            Q=Q,
            p=p,
            nf=nf,
            mur_ratio=mur_ratio,
            muf_ratio=muf_ratio,
        )
    bare_res = (
        Cg_3_m_reg(z, Q, p, nf)
        + Cg_2_m_reg(z, Q, p, nf - 1) * Mgg_1_loc(z, p, nf - 1)
        + P2(p) * Cg_1_m_reg(z, Q, p, nf - 1)
        - (
            Cg_1_m_reg(z, Q, p, nf - 1) * Mgg_2_loc(z, p, nf - 1)
            + Convolute(Cg_1_m_reg, Mgg_2_reg, z, Q, p, nf - 1, nf - 1)
            + Convolute_plus_matching(Cg_1_m_reg, Mgg_2_sing, z, Q, p, nf - 1, nf - 1)
        )
        - 2
        * Cb_0_loc(z, Q, p, nf)
        * (
            Mbg_3_reg(
                z, p, nf
            )  # This is called with nf instead of nf-1 but the grid is computed correctly with nf-1
            - Mgg_1_loc(z, p, nf - 1) * Mbg_2(z, p, nf - 1)
            + Mbg_1(z, p, nf - 1) * Mgg_1_loc(z, p, nf - 1) * Mgg_1_loc(z, p, nf - 1)
            - (
                Mbg_1(z, p, nf - 1) * Mgg_2_loc(z, p, nf - 1)
                + Convolute_matching(Mbg_1, Mgg_2_reg, z, Q, p, nf - 1)
                + Mbg1_Mgg2_sing(z, p, nf - 1)
            )
        )
        - 2
        * (
            Cb_1_loc(z, Q, p, nf) * Mbg_2(z, p, nf - 1)
            + Convolute(Cb_1_reg, Mbg_2, z, Q, p, nf - 1)
            + Convolute_plus_coeff(Cb_1_sing, Mbg_2, z, Q, p, nf - 1)
            - Cb1_Mbg1(z, p, nf - 1) * Mgg_1_loc(z, p, nf - 1)
        )
        - 2
        * (
            Mbg_1(z, p, nf - 1) * Cb_2_loc(z, Q, p, nf - 1)
            + Convolute(Cb_2_reg, Mbg_1, z, Q, p, nf - 1)
            + Convolute_plus_coeff(Cb_2_sing, Mbg_1, z, Q, p, nf - 1)
        )
    )
    return scale_variations.apply_sv_kernel(
        order=2,
        m=1,
        ingredients=[bare_res, Cg_1_til_reg, Cg_2_til_reg],
        z=z,
        Q=Q,
        p=p,
        nf=nf,
        mur_ratio=mur_ratio,
        muf_ratio=muf_ratio,
    )


def Cq_2_til_reg(z, Q, p, _nf, mur_ratio=1.0, muf_ratio=1.0):
    bare_res = Cq_2_m_reg(z, Q, p, _nf - 1) - 2 * Cb_0_loc(z, Q, p, _nf) * Mbq_2(
        z, p, _nf
    )
    return scale_variations.apply_sv_kernel(
        order=0,
        m=2,
        ingredients=[bare_res],
        z=z,
        Q=Q,
        p=p,
        nf=_nf,
        mur_ratio=mur_ratio,
        muf_ratio=muf_ratio,
    )


def Cq_3_til_reg(z, Q, p, nf, use_analytic=False, mur_ratio=1.0, muf_ratio=1.0):
    if parameters.grids and not use_analytic:
        bare_res = Ini.Cq3_til[nf - 4](z, Q)[0]
        return scale_variations.apply_sv_kernel(
            order=1,
            m=2,
            ingredients=[bare_res, Cq_2_til_reg],
            z=z,
            Q=Q,
            p=p,
            nf=nf,
            mur_ratio=mur_ratio,
            muf_ratio=muf_ratio,
        )
    bare_res = (
        Cq_3_m_reg(z, Q, p, nf)
        + 2 * Cq_2_m_reg(z, Q, p, nf - 1) * Mgg_1_loc(z, p, nf - 1)
        - Convolute(Cg_1_m_reg, Mgq_2_reg, z, Q, p, nf - 1, nf - 1)
        - 2
        * (
            Cb_1_loc(z, Q, p, nf) * Mbq_2(z, p, nf - 1)
            + Convolute(Cb_1_reg, Mbq_2, z, Q, p, nf - 1)
            + Convolute_plus_coeff(Cb_1_sing, Mbq_2, z, Q, p, nf - 1)
        )
        - 2
        * (
            Cb_0_loc(z, Q, p, nf) * Mbq_3_reg(z, p, nf)
        )  # This is called with nf instead of nf-1 but the grid is computed correctly with nf-1
    )
    return scale_variations.apply_sv_kernel(
        order=1,
        m=2,
        ingredients=[bare_res, Cq_2_til_reg],
        z=z,
        Q=Q,
        p=p,
        nf=nf,
        mur_ratio=mur_ratio,
        muf_ratio=muf_ratio,
    )


# FL
def CLg_1_til_reg(z, Q, p, _nf, mur_ratio=1.0, muf_ratio=1.0):
    bare_res = CLg_1_m_reg(z, Q, p, _nf - 1)
    return scale_variations.apply_sv_kernel(
        order=0,
        m=1,
        ingredients=[bare_res],
        z=z,
        Q=Q,
        p=p,
        nf=_nf,
        mur_ratio=mur_ratio,
        muf_ratio=muf_ratio,
    )


def CLg_2_til_reg(z, Q, p, _nf, mur_ratio=1.0, muf_ratio=1.0):
    bare_res = CLg_2_m_reg(z, Q, p, _nf - 1) - 2 * np.log(
        (Q**2) / (p[0] ** 2)
    ) * CLb1_Mbg1(z, p, _nf)
    return scale_variations.apply_sv_kernel(
        order=1,
        m=1,
        ingredients=[bare_res, CLg_1_til_reg],
        z=z,
        Q=Q,
        p=p,
        nf=_nf,
        mur_ratio=mur_ratio,
        muf_ratio=muf_ratio,
    )


def CLg_3_til_reg(z, Q, p, nf, use_analytic=False, mur_ratio=1.0, muf_ratio=1.0):
    if parameters.grids and not use_analytic:
        bare_res = Ini.CLg3_til[nf - 4](z, Q)[0]
        return scale_variations.apply_sv_kernel(
            order=2,
            m=1,
            ingredients=[bare_res, CLg_1_til_reg, CLg_2_til_reg],
            z=z,
            Q=Q,
            p=p,
            nf=nf,
            mur_ratio=mur_ratio,
            muf_ratio=muf_ratio,
        )
    bare_res = (
        CLg_3_m_reg(z, Q, p, nf)
        + CLg_2_m_reg(z, Q, p, nf - 1) * Mgg_1_loc(z, p, nf - 1)
        + P2(p) * CLg_1_m_reg(z, Q, p, nf - 1)
        - (
            CLg_1_m_reg(z, Q, p, nf - 1) * Mgg_2_loc(z, p, nf - 1)
            + Convolute(CLg_1_m_reg, Mgg_2_reg, z, Q, p, nf - 1, nf - 1)
            + Convolute_plus_matching(CLg_1_m_reg, Mgg_2_sing, z, Q, p, nf - 1, nf - 1)
        )
        - 2
        * (
            Convolute(CLb_1_reg, Mbg_2, z, Q, p, nf - 1)
            - CLb1_Mbg1(z, p, nf - 1) * Mgg_1_loc(z, p, nf - 1)
        )
        - 2
        * (
            CLb_2_loc(z, Q, p, nf) * Mbg_1(z, p, nf - 1)
            + Convolute(CLb_2_reg, Mbg_1, z, Q, p, nf - 1)
        )
    )
    return scale_variations.apply_sv_kernel(
        order=2,
        m=1,
        ingredients=[bare_res, CLg_1_til_reg, CLg_2_til_reg],
        z=z,
        Q=Q,
        p=p,
        nf=nf,
        mur_ratio=mur_ratio,
        muf_ratio=muf_ratio,
    )


def CLq_2_til_reg(z, Q, p, _nf, mur_ratio=1.0, muf_ratio=1.0):
    bare_res = CLq_2_m_reg(z, Q, p, _nf - 1)
    return scale_variations.apply_sv_kernel(
        order=0,
        m=2,
        ingredients=[bare_res],
        z=z,
        Q=Q,
        p=p,
        nf=_nf,
        mur_ratio=mur_ratio,
        muf_ratio=muf_ratio,
    )


def CLq_3_til_reg(z, Q, p, nf, use_analytic=False, mur_ratio=1.0, muf_ratio=1.0):
    if parameters.grids and not use_analytic:
        bare_res = Ini.CLq3_til[nf - 4](z, Q)[0]
        return scale_variations.apply_sv_kernel(
            order=1,
            m=2,
            ingredients=[bare_res, CLq_2_til_reg],
            z=z,
            Q=Q,
            p=p,
            nf=nf,
            mur_ratio=mur_ratio,
            muf_ratio=muf_ratio,
        )
    bare_res = (
        CLq_3_m_reg(z, Q, p, nf)
        + 2 * CLq_2_m_reg(z, Q, p, nf - 1) * Mgg_1_loc(z, p, nf - 1)
        - Convolute(CLg_1_m_reg, Mgq_2_reg, z, Q, p, nf - 1, nf - 1)
        - 2 * Convolute(CLb_1_reg, Mbq_2, z, Q, p, nf - 1)
    )
    return scale_variations.apply_sv_kernel(
        order=1,
        m=2,
        ingredients=[bare_res, CLq_2_til_reg],
        z=z,
        Q=Q,
        p=p,
        nf=nf,
        mur_ratio=mur_ratio,
        muf_ratio=muf_ratio,
    )
