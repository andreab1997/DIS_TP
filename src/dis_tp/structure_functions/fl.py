"""FL structure function"""
import lhapdf
import numpy as np

from .. import MassiveCoeffFunc, MasslessCoeffFunc, TildeCoeffFunc
from ..parameters import charges, masses, number_active_flavors, pids
from ..tools import PDFConvolute, PDFConvolute_plus

g_id = pids["g"]

def FL_FO(order, pdf, x, Q, h_id, muF_ratio=1, muR_ratio=1):
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
    lhapdf.setVerbosity(0)
    Mypdf = None
    if isinstance(pdf, list):
        Mypdf = lhapdf.mkPDF(pdf[order - 1], 0)
    elif isinstance(pdf, str):
        Mypdf = lhapdf.mkPDF(pdf, 0)
    muF = muF_ratio * Q
    muR = muR_ratio * Q
    p = [masses(h_id), Q, charges(h_id)]
    nf = number_active_flavors(h_id)
    res = 0.0
    if order >= 0:
        res += 0.0
    if order >= 1:
        res += (
            (1 / (4 * np.pi))
            * Mypdf.alphasQ(muR)
            * PDFConvolute(MassiveCoeffFunc.CLg_1_m_reg, Mypdf, x, Q, p, nf, g_id)
        )
    if order >= 2:
        res += pow((1 / (4 * np.pi)) * Mypdf.alphasQ(muR), 2) * (
            PDFConvolute(MassiveCoeffFunc.CLg_2_m_reg, Mypdf, x, Q, p, nf, g_id)
            + PDFConvolute(MassiveCoeffFunc.CLq_2_m_reg, Mypdf, x, Q, p, nf)
        )
    if order >= 3:
        res += pow((1 / (4 * np.pi)) * Mypdf.alphasQ(muR), 3) * (
            PDFConvolute(MassiveCoeffFunc.CLg_3_m_reg, Mypdf, x, Q, p, nf, g_id)
            + PDFConvolute(MassiveCoeffFunc.CLq_3_m_reg, Mypdf, x, Q, p, nf)
        )
    return res


def FL_R(order, pdf, x, Q, h_id, muF_ratio=1, muR_ratio=1):
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
    lhapdf.setVerbosity(0)
    Mypdf = None
    if isinstance(pdf, list):
        Mypdf = lhapdf.mkPDF(pdf[order - 1], 0)
    elif isinstance(pdf, str):
        Mypdf = lhapdf.mkPDF(pdf, 0)
    muF = muF_ratio * Q
    muR = muR_ratio * Q
    p = [masses(h_id), Q, charges(h_id)]
    nf = number_active_flavors(h_id)
    res = 0.0
    if order >= 0:
        res += 0.0
    if order >= 1:
        res += (
            (1 / (4 * np.pi))
            * Mypdf.alphasQ(muR)
            * PDFConvolute(MasslessCoeffFunc.CLg_1_reg, Mypdf, x, Q, p, nf, g_id)
        )
    if order >= 2:
        nnll_reg = (
            (1 / (4 * np.pi))
            * Mypdf.alphasQ(muR)
            * (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (
                    PDFConvolute(MasslessCoeffFunc.CLg_2_reg, Mypdf, x, Q, p, nf, g_id)
                    + PDFConvolute(MasslessCoeffFunc.CLq_2_reg, Mypdf, x, Q, p, nf)
                )
                + PDFConvolute(MasslessCoeffFunc.CLb_1_reg, Mypdf, x, Q, p, nf, h_id)
            )
        )
        res += nnll_reg
    if order >= 3:
        n3ll_reg = (((1 / (4 * np.pi)) * Mypdf.alphasQ(muR)) ** 2) * (
            (1 / (4 * np.pi))
            * Mypdf.alphasQ(muR)
            * (
                PDFConvolute(MasslessCoeffFunc.CLg_3_reg, Mypdf, x, Q, p, nf, g_id)
                + PDFConvolute(MasslessCoeffFunc.CLq_3_reg, Mypdf, x, Q, p, nf)
            )
            + PDFConvolute(MasslessCoeffFunc.CLb_2_reg, Mypdf, x, Q, p, nf, h_id)
        )
        n3ll_loc = (((1 / (4 * np.pi)) * Mypdf.alphasQ(muR)) ** 2) * (
            MasslessCoeffFunc.CLb_2_loc(x, Q, p, nf)
            * (Mypdf.xfxQ2(h_id, x, Q * Q) + Mypdf.xfxQ2(-h_id, x, Q * Q))
        )
        res += n3ll_reg + n3ll_loc
    return res


def FL_M(order, meth, pdf, x, Q, h_id, muF_ratio=1, muR_ratio=1):
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
    lhapdf.setVerbosity(0)
    Mypdf = None
    if isinstance(pdf, list):
        Mypdf = lhapdf.mkPDF(pdf[order - 1], 0)
    elif isinstance(pdf, str):
        Mypdf = lhapdf.mkPDF(pdf, 0)
    muF = muF_ratio * Q
    muR = muR_ratio * Q
    nf = number_active_flavors(h_id)
    p = [masses(h_id), Q, charges(h_id)]
    res = 0.0
    if meth == "our":
        if order >= 0:
            res += 0.0
        if order >= 1:
            res += (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * PDFConvolute(TildeCoeffFunc.CLg_1_til_reg, Mypdf, x, Q, p, nf, g_id)
            )
        if order >= 2:
            nnlo_nnll_reg = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (
                    (1 / (4 * np.pi))
                    * Mypdf.alphasQ(muR)
                    * (
                        PDFConvolute(
                            TildeCoeffFunc.CLg_2_til_reg, Mypdf, x, Q, p, nf, g_id
                        )
                        + PDFConvolute(TildeCoeffFunc.CLq_2_til_reg, Mypdf, x, Q, p, nf)
                    )
                    + PDFConvolute(
                        MasslessCoeffFunc.CLb_1_reg, Mypdf, x, Q, p, nf, h_id
                    )
                )
            )
            res += nnlo_nnll_reg
        if order >= 3:
            n3lo_n3ll_reg = (((1 / (4 * np.pi)) * Mypdf.alphasQ(muR)) ** 2) * (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (
                    PDFConvolute(TildeCoeffFunc.CLg_3_til_reg, Mypdf, x, Q, p, nf, g_id)
                    + PDFConvolute(TildeCoeffFunc.CLq_3_til_reg, Mypdf, x, Q, p, nf)
                )
                + PDFConvolute(MasslessCoeffFunc.CLb_2_reg, Mypdf, x, Q, p, nf, h_id)
            )
            n3lo_n3ll_loc = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * MasslessCoeffFunc.CLb_2_loc(x, Q, p, nf)
                * (Mypdf.xfxQ2(h_id, x, Q * Q) + Mypdf.xfxQ2(-h_id, x, Q * Q))
            )
            res += n3lo_n3ll_loc + n3lo_n3ll_reg
    if meth == "fonll":
        if order >= 0:
            res += 0.0
        if order >= 1:
            res += (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (
                    PDFConvolute(TildeCoeffFunc.CLg_1_til_reg, Mypdf, x, Q, p, nf, g_id)
                    + PDFConvolute(
                        MasslessCoeffFunc.CLb_1_reg, Mypdf, x, Q, p, nf, h_id
                    )
                )
            )
        if order >= 2:
            nnlo_nnll_reg = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (
                    PDFConvolute(TildeCoeffFunc.CLg_2_til_reg, Mypdf, x, Q, p, nf, g_id)
                    + PDFConvolute(TildeCoeffFunc.CLq_2_til_reg, Mypdf, x, Q, p, nf)
                    + PDFConvolute(
                        MasslessCoeffFunc.CLb_2_reg, Mypdf, x, Q, p, nf, h_id
                    )
                )
            )
            nnlo_nnll_loc = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * MasslessCoeffFunc.CLb_2_loc(x, Q, p, nf)
                * (Mypdf.xfxQ2(h_id, x, Q * Q) + Mypdf.xfxQ2(-h_id, x, Q * Q))
            )
            res += nnlo_nnll_reg + nnlo_nnll_loc
        if order >= 3:
            n3lo_n3ll_reg = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (
                    PDFConvolute(TildeCoeffFunc.CLg_3_til_reg, Mypdf, x, Q, p, nf, g_id)
                    + PDFConvolute(TildeCoeffFunc.CLq_3_til_reg, Mypdf, x, Q, p, nf)
                    + PDFConvolute(
                        MasslessCoeffFunc.CLb_3_reg, Mypdf, x, Q, p, nf, h_id
                    )
                )
            )
            n3lo_n3ll_loc = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * MasslessCoeffFunc.CLb_3_loc(x, Q, p, nf)
                * (Mypdf.xfxQ2(h_id, x, Q * Q) + Mypdf.xfxQ2(-h_id, x, Q * Q))
            )
            res += n3lo_n3ll_reg + n3lo_n3ll_loc
    return res
