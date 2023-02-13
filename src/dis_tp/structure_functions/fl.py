"""FL structure function"""
import numpy as np

from .. import MassiveCoeffFunc, MasslessCoeffFunc, TildeCoeffFunc
from ..parameters import (
    charges,
    masses,
    number_active_flavors,
    number_light_flavors,
    pids,
    alpha_s,
)
from ..tools import PDFConvolute
from .tools import PDFConvolute_light, mkPDF, non_singlet_pdf

g_id = pids["g"]


def FL_FO(order, pdf, x, Q, h_id, meth=None, muF_ratio=1, muR_ratio=1):
    """
    Compute the FO results for the structure function FL

    Parameters:
        order : int
            requested perturbative order (0 == LO, 1 == NLO,...)
        pdf : str or list(str)
            pdf(s) to be used
        x : float
            x-value
        Q : float
            Q-value
        h_id : int
            heavy quark id
        muF_ratio : float
            ratio to Q of the factorization scale
        muR_ratio : float
            ratio to Q of the renormalization scale
    Returns:
            : float
            result
    """
    Mypdf = mkPDF(pdf, order)
    muR = muR_ratio * Q
    p = [masses(h_id), Q, charges(h_id)]
    nf = number_active_flavors(h_id)
    a_s = alpha_s(muR**2, Q**2)
    if order >= 0:
        res = 0.0
    if order >= 1:
        res += a_s * PDFConvolute(
            MassiveCoeffFunc.CLg_1_m_reg, Mypdf, x, Q, p, nf, g_id
        )
    if order >= 2:
        res += a_s**2 * (
            PDFConvolute(MassiveCoeffFunc.CLg_2_m_reg, Mypdf, x, Q, p, nf, g_id)
            + PDFConvolute(MassiveCoeffFunc.CLq_2_m_reg, Mypdf, x, Q, p, nf)
        )
    if order >= 3:
        res += a_s**3 * (
            PDFConvolute(MassiveCoeffFunc.CLg_3_m_reg, Mypdf, x, Q, p, nf, g_id)
            + PDFConvolute(MassiveCoeffFunc.CLq_3_m_reg, Mypdf, x, Q, p, nf)
        )
    return res


def FL_R(order, pdf, x, Q, h_id, meth=None, muF_ratio=1, muR_ratio=1):
    """
    Compute the R result for the structure function FL

    Parameters:
        order : int
            requested perturbative order (0 == LO, 1 == NLO,...)
        pdf : str or list(str)
            pdf(s) to be used
        x : float
            x-value
        Q : float
            Q-value
        h_id : int
            heavy quark id
        muF_ratio : float
            ratio to Q of the factorization scale
        muR_ratio : float
            ratio to Q of the renormalization scale
    Returns:
            : float
            result
    """
    Mypdf = mkPDF(pdf, order)
    muR = muR_ratio * Q
    p = [masses(h_id), Q, charges(h_id)]
    nf = number_active_flavors(h_id)
    a_s = alpha_s(muR**2, Q**2)
    res = 0.0
    if order >= 0:
        res = 0.0
    if order >= 1:
        res += a_s * PDFConvolute(MasslessCoeffFunc.CLg_1_reg, Mypdf, x, Q, p, nf, g_id)
    if order >= 2:
        nnll_reg = a_s * (
            a_s
            * (
                PDFConvolute(MasslessCoeffFunc.CLg_2_reg, Mypdf, x, Q, p, nf, g_id)
                + PDFConvolute(MasslessCoeffFunc.CLq_2_reg, Mypdf, x, Q, p, nf)
            )
            + PDFConvolute(MasslessCoeffFunc.CLb_1_reg, Mypdf, x, Q, p, nf, h_id)
        )
        res += nnll_reg
    if order >= 3:
        n3ll_reg = a_s**2 * (
            a_s
            * (
                PDFConvolute(MasslessCoeffFunc.CLg_3_reg, Mypdf, x, Q, p, nf, g_id)
                + PDFConvolute(MasslessCoeffFunc.CLq_3_reg, Mypdf, x, Q, p, nf)
            )
            + PDFConvolute(MasslessCoeffFunc.CLb_2_reg, Mypdf, x, Q, p, nf, h_id)
        )
        n3ll_loc = a_s**2 * (
            MasslessCoeffFunc.CLb_2_loc(x, Q, p, nf)
            * (Mypdf.xfxQ2(h_id, x, Q * Q) + Mypdf.xfxQ2(-h_id, x, Q * Q))
        )
        res += n3ll_reg + n3ll_loc
    return res


def FL_M(order, pdf, x, Q, h_id, meth, muF_ratio=1, muR_ratio=1):
    """
    Compute the M result for the structure function FL

    Parameters:
        order : int
            requested perturbative order (0 == LO, 1 == NLO,...)
        meth : str
            method to be used (our, fonll)
        pdf : str or list(str)
            pdf(s) to be used
        x : float
            x-value
        Q : float
            Q-value
        h_id : int
            heavy quark id
        muF_ratio : float
            ratio to Q of the factorization scale
        muR_ratio : float
            ratio to Q of the renormalization scale
    Returns:
            : float
            result
    """
    Mypdf = mkPDF(pdf, order)
    muR = muR_ratio * Q
    nf = number_active_flavors(h_id)
    p = [masses(h_id), Q, charges(h_id)]
    a_s = alpha_s(muR**2, Q**2)
    if meth == "our":
        if order >= 0:
            res = 0.0
        if order >= 1:
            res += a_s * PDFConvolute(
                TildeCoeffFunc.CLg_1_til_reg, Mypdf, x, Q, p, nf, g_id
            )
        if order >= 2:
            nnlo_nnll_reg = a_s * (
                a_s
                * (
                    PDFConvolute(TildeCoeffFunc.CLg_2_til_reg, Mypdf, x, Q, p, nf, g_id)
                    + PDFConvolute(TildeCoeffFunc.CLq_2_til_reg, Mypdf, x, Q, p, nf)
                )
                + PDFConvolute(MasslessCoeffFunc.CLb_1_reg, Mypdf, x, Q, p, nf, h_id)
            )
            res += nnlo_nnll_reg
        if order >= 3:
            n3lo_n3ll_reg = a_s**2 * (
                a_s
                * (
                    PDFConvolute(TildeCoeffFunc.CLg_3_til_reg, Mypdf, x, Q, p, nf, g_id)
                    + PDFConvolute(TildeCoeffFunc.CLq_3_til_reg, Mypdf, x, Q, p, nf)
                )
                + PDFConvolute(MasslessCoeffFunc.CLb_2_reg, Mypdf, x, Q, p, nf, h_id)
            )
            n3lo_n3ll_loc = (
                a_s
                * a_s
                * MasslessCoeffFunc.CLb_2_loc(x, Q, p, nf)
                * (Mypdf.xfxQ2(h_id, x, Q * Q) + Mypdf.xfxQ2(-h_id, x, Q * Q))
            )
            res += n3lo_n3ll_loc + n3lo_n3ll_reg
    if meth == "fonll":
        if order >= 0:
            res = 0.0
        if order >= 1:
            res += a_s * (
                PDFConvolute(TildeCoeffFunc.CLg_1_til_reg, Mypdf, x, Q, p, nf, g_id)
                + PDFConvolute(MasslessCoeffFunc.CLb_1_reg, Mypdf, x, Q, p, nf, h_id)
            )
        if order >= 2:
            nnlo_nnll_reg = a_s**2 * (
                PDFConvolute(TildeCoeffFunc.CLg_2_til_reg, Mypdf, x, Q, p, nf, g_id)
                + PDFConvolute(TildeCoeffFunc.CLq_2_til_reg, Mypdf, x, Q, p, nf)
                + PDFConvolute(MasslessCoeffFunc.CLb_2_reg, Mypdf, x, Q, p, nf, h_id)
            )
            nnlo_nnll_loc = (
                a_s**2
                * MasslessCoeffFunc.CLb_2_loc(x, Q, p, nf)
                * (Mypdf.xfxQ2(h_id, x, Q * Q) + Mypdf.xfxQ2(-h_id, x, Q * Q))
            )
            res += nnlo_nnll_reg + nnlo_nnll_loc
        if order >= 3:
            n3lo_n3ll_reg = a_s**3 * (
                PDFConvolute(TildeCoeffFunc.CLg_3_til_reg, Mypdf, x, Q, p, nf, g_id)
                + PDFConvolute(TildeCoeffFunc.CLq_3_til_reg, Mypdf, x, Q, p, nf)
                + PDFConvolute(MasslessCoeffFunc.CLb_3_reg, Mypdf, x, Q, p, nf, h_id)
            )
            n3lo_n3ll_loc = (
                a_s**3
                * MasslessCoeffFunc.CLb_3_loc(x, Q, p, nf)
                * (Mypdf.xfxQ2(h_id, x, Q * Q) + Mypdf.xfxQ2(-h_id, x, Q * Q))
            )
            res += n3lo_n3ll_reg + n3lo_n3ll_loc
    return res


def FL_Light(order, pdf, x, Q, h_id=None, meth=None, muR_ratio=1):
    """
    Compute the light contribution for the structure function FL

    Parameters:
        order : int
            requested perturbative order (0 == LO, 1 == NLO,...)
        pdf : str or list(str)
            pdf(s) to be used
        x : float
            x-value
        Q : float
            Q-value
        h_id : int
            heavy quark id
        muR_ratio : float
            ratio to Q of the renormalization scale
    Returns:
            : float
            result
    """
    Mypdf = mkPDF(pdf, order)
    muR = muR_ratio * Q
    p = [0, Q, 1]
    nl = number_light_flavors(Q)
    a_s = alpha_s(muR**2, Q**2)
    meansq_e = np.mean([charges(nl) ** 2 for nl in range(1, nl + 1)])
    if order >= 0:
        res = 0.0
    if order >= 1:
        reg = PDFConvolute_light(
            MasslessCoeffFunc.CLb_1_reg, Mypdf, x, Q, p, nl
        ) + nl * meansq_e * PDFConvolute(
            MasslessCoeffFunc.CLg_1_reg, Mypdf, x, Q, p, nl, g_id
        )
        res += a_s * reg
    if order >= 2:
        reg = PDFConvolute_light(
            MasslessCoeffFunc.CLb_2_reg, Mypdf, x, Q, p, nl
        ) + nl * meansq_e * (
            PDFConvolute(MasslessCoeffFunc.CLg_2_reg, Mypdf, x, Q, p, nl, g_id)
            + PDFConvolute(MasslessCoeffFunc.CLq_2_reg, Mypdf, x, Q, p, nl+1)
        )
        loc = MasslessCoeffFunc.CLb_2_loc(x, Q, p, nl) * non_singlet_pdf(
            Mypdf, x, Q, nl
        )
        res += a_s**2 * (reg + loc)
    if order >= 3:
        reg = PDFConvolute_light(
            MasslessCoeffFunc.CLb_3_reg, Mypdf, x, Q, p, nl
        ) + nl * meansq_e * (
            PDFConvolute(MasslessCoeffFunc.CLg_3_reg, Mypdf, x, Q, p, nl, g_id)
            + PDFConvolute(MasslessCoeffFunc.CLq_3_reg, Mypdf, x, Q, p, nl+1)
        )
        loc = MasslessCoeffFunc.CLb_3_loc(x, Q, p, nl) * non_singlet_pdf(
            Mypdf, x, Q, nl
        )
        res += a_s**3 * (reg + loc)
    return res


def FL_ZM(order, pdf, x, Q, h_id, meth=None, muR_ratio=1):
    """
    Compute the ZM heavy contribution to structure function FL

    Parameters:
        order : int
            requested perturbative order (0 == LO, 1 == NLO,...)
        pdf : str or list(str)
            pdf(s) to be used
        x : float
            x-value
        Q : float
            Q-value
        h_id : int
            heavy quark id
        muR_ratio : float
            ratio to Q of the renormalization scale
    Returns:
            : float
            result
    """
    Mypdf = mkPDF(pdf, order)
    muR = muR_ratio * Q
    nl = 1
    p = [masses(h_id), Q, charges(h_id)]
    a_s = alpha_s(muR**2, Q**2)
    pdfxfx = Mypdf.xfxQ2(h_id, x, Q**2) + Mypdf.xfxQ2(-h_id, x, Q**2)
    if order >= 0:
        res = 0
    if order >= 1:
        reg = PDFConvolute(
            MasslessCoeffFunc.CLb_1_reg, Mypdf, x, Q, p, nl, h_id
        ) + PDFConvolute(MasslessCoeffFunc.CLg_1_reg, Mypdf, x, Q, p, nl, g_id)
        res += a_s * reg
    if order >= 2:
        reg = (
            PDFConvolute(MasslessCoeffFunc.CLb_2_reg, Mypdf, x, Q, p, nl, h_id)
            + PDFConvolute(MasslessCoeffFunc.CLg_2_reg, Mypdf, x, Q, p, nl, g_id)
            + PDFConvolute(MasslessCoeffFunc.CLq_2_reg, Mypdf, x, Q, p, nl, h_id)
        )
        loc = MasslessCoeffFunc.CLb_2_loc(x, Q, p, nl) * pdfxfx
        res += a_s**2 * (reg + loc)
    if order >= 3:
        reg = (
            PDFConvolute(MasslessCoeffFunc.CLb_3_reg, Mypdf, x, Q, p, nl, h_id)
            + PDFConvolute(MasslessCoeffFunc.CLg_3_reg, Mypdf, x, Q, p, nl, g_id)
            + PDFConvolute(MasslessCoeffFunc.CLq_3_reg, Mypdf, x, Q, p, nl, h_id)
        )
        loc = MasslessCoeffFunc.CLb_3_loc(x, Q, p, nl) * pdfxfx
        res += a_s**3 * (reg + loc)
    return res


def FL_FONLL(order, pdf, x, Q, h_id, meth, muR_ratio=1):
    """
    Compute the Yadism like FONLL structure function FL

    Parameters:
        order : int
            requested perturbative order (0 == LO, 1 == NLO,...)
        pdf : str or list(str)
            pdf(s) to be used
        x : float
            x-value
        Q : float
            Q-value
        h_id : int
            heavy quark id
        muR_ratio : float
            ratio to Q of the renormalization scale
    Returns:
            : float
            result
    """
    mh = masses(h_id)
    mhp1 = masses(h_id + 1)
    if Q < mh:
        return FL_FO(order, pdf, x, Q, h_id, muR_ratio=muR_ratio)
    elif Q < mhp1:
        return FL_M(order, pdf, x, Q, h_id, meth, muR_ratio=muR_ratio)
    elif Q >= mhp1:
        return FL_ZM(order, pdf, x, Q, h_id, muR_ratio=muR_ratio)

def FL_Total(order, pdf, x, Q, h_id, meth, muR_ratio=1):
    """
    Compute the total structure function FL.

    Parameters:
        order : int
            requested perturbative order (0 == LO, 1 == NLO,...)
        meth : str
            method to be used (our, fonll)
        pdf : str or list(str)
            pdf(s) to be used
        x : float
            x-value
        Q : float
            Q-value
        muR_ratio : float
            ratio to Q of the renormalization scale
    Returns:
        : float
            result
    """
    if Q < masses(5):
        res = (
            FL_Light(order, pdf, x, Q, 3, muR_ratio)
            + FL_FONLL(order, pdf, x, Q, 4, meth, muR_ratio=muR_ratio)
        )
    if Q >= masses(5):
        res = (
            FL_Light(order, pdf, x, Q, 4, muR_ratio)
             + FL_FONLL(order, pdf, x, Q, 5, meth, muR_ratio=muR_ratio)
        )
    return res