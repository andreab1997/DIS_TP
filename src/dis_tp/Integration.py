# This is the actual code for computing structure functions at fixed order, in zero-mass scheme and in a matched massive scheme
import lhapdf
import numpy as np

from . import Initialize as Ini
from . import MassiveCoeffFunc, MasslessCoeffFunc, TildeCoeffFunc
from .parameters import (
    charges,
    masses,
    number_active_flavors,
    number_light_flavors,
    pids,
)
from .tools import PDFConvolute, PDFConvolute_plus

g_id = pids["g"]


def Initialize_all(nf):
    """
    Initialize all the needed global lists
    """
    Ini.InitializeQX()
    # Ini.InitializeCg2_m(nf)
    # Ini.InitializeCq2_m(nf)
    # Ini.InitializeCLg2_m(nf)
    # Ini.InitializeCLq2_m(nf)
    # Ini.InitializeMbg2(nf)
    # Ini.InitializeMbq2(nf)
    # Ini.InitializeHPL()
    Ini.InitializeMbg_3(nf)
    Ini.InitializeMbq_3(nf)
    # Ini.InitializeCg2_til(nf)
    # Ini.InitializeCq2_til(nf)
    # Ini.InitializeCLg2_til(nf)
    # Ini.InitializeCLq2_til(nf)
    # Ini.InitializeCq3_m(nf)
    # Ini.InitializeCLq3_m(nf)
    # Ini.InitializeCg3_m(nf)
    # Ini.InitializeCLg3_m(nf)
    # Ini.InitializeCLq3_til(nf)
    # Ini.InitializeCLg3_til(nf)
    # Ini.InitializeCq3_til(nf)
    # Ini.InitializeCg3_til(nf)


def F2_FO(order, pdf, x, Q, h_id, muF_ratio=1, muR_ratio=1):
    """
    Compute the FO result for the structure function F2

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
            * PDFConvolute(MassiveCoeffFunc.Cg_1_m_reg, Mypdf, x, Q, p, nf, g_id)
        )
    if order >= 2:
        res += pow((1 / (4 * np.pi)) * Mypdf.alphasQ(muR), 2) * (
            PDFConvolute(MassiveCoeffFunc.Cg_2_m_reg, Mypdf, x, Q, p, nf, g_id)
            + PDFConvolute(MassiveCoeffFunc.Cq_2_m_reg, Mypdf, x, Q, p, nf)
        )
    if order >= 3:
        res += pow((1 / (4 * np.pi)) * Mypdf.alphasQ(muR), 3) * (
            PDFConvolute(MassiveCoeffFunc.Cg_3_m_reg, Mypdf, x, Q, p, nf, g_id)
            + PDFConvolute(MassiveCoeffFunc.Cq_3_m_reg, Mypdf, x, Q, p, nf)
        )
    return res


def F2_R(order, pdf, x, Q, h_id, muF_ratio=1, muR_ratio=1):
    """
    Compute the R result for the structure function F2

    Parameters:
        order : int
            requested perturbative order (0 == LO, 1 == NLO,...)
        pdf : str or list(str)
            pdf(s) to be used
        x : float
            x-value
        Q : float
            Q-value
        h_id: int
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
    if order >= 0:
        res += 0.0
    if order >= 1:
        nll_reg = (
            (1 / (4 * np.pi))
            * Mypdf.alphasQ(muR)
            * PDFConvolute(MasslessCoeffFunc.Cg_1_reg, Mypdf, x, Q, p, nf, g_id)
        )
        nll_local = MasslessCoeffFunc.Cb_0_loc(x, Q, p, nf) * (
            Mypdf.xfxQ2(h_id, x, Q * Q) + Mypdf.xfxQ2(-h_id, x, Q * Q)
        )
        res += nll_reg + nll_local
    if order >= 2:
        nnll_reg = (
            (1 / (4 * np.pi))
            * Mypdf.alphasQ(muR)
            * (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (
                    PDFConvolute(MasslessCoeffFunc.Cg_2_reg, Mypdf, x, Q, p, nf, g_id)
                    + PDFConvolute(MasslessCoeffFunc.Cq_2_reg, Mypdf, x, Q, p, nf)
                )
                + PDFConvolute(MasslessCoeffFunc.Cb_1_reg, Mypdf, x, Q, p, nf, h_id)
            )
        )
        nnll_local = (
            (1 / (4 * np.pi))
            * Mypdf.alphasQ(muR)
            * MasslessCoeffFunc.Cb_1_loc(x, Q, p, nf)
            * (Mypdf.xfxQ2(h_id, x, Q * Q) + Mypdf.xfxQ2(-h_id, x, Q * Q))
        )
        nnll_sing = (
            (1 / (4 * np.pi))
            * Mypdf.alphasQ(muR)
            * PDFConvolute_plus(MasslessCoeffFunc.Cb_1_sing, Mypdf, x, Q, p, nf, h_id)
        )
        res += nnll_reg + nnll_local + nnll_sing
    if order >= 3:
        n3ll_reg = (((1 / (4 * np.pi)) * Mypdf.alphasQ(muR)) ** 2) * (
            (1 / (4 * np.pi))
            * Mypdf.alphasQ(muR)
            * (
                PDFConvolute(MasslessCoeffFunc.Cg_3_reg, Mypdf, x, Q, p, nf, g_id)
                + PDFConvolute(MasslessCoeffFunc.Cq_3_reg, Mypdf, x, Q, p, nf)
            )
            + PDFConvolute(MasslessCoeffFunc.Cb_2_reg, Mypdf, x, Q, p, nf, h_id)
        )
        n3ll_local = (((1 / (4 * np.pi)) * Mypdf.alphasQ(muR)) ** 2) * (
            MasslessCoeffFunc.Cb_2_loc(x, Q, p, nf)
            * (Mypdf.xfxQ2(h_id, x, Q * Q) + Mypdf.xfxQ2(-h_id, x, Q * Q))
            + (1 / (4 * np.pi))
            * Mypdf.alphasQ(muR)
            * (
                MasslessCoeffFunc.Cg_3_loc(x, Q, p, nf) * Mypdf.xfxQ2(h_id, x, Q * Q)
                + MasslessCoeffFunc.Cq_3_loc(x, Q, p, nf)
                * (
                    Mypdf.xfxQ2(1, x, Q * Q)
                    + Mypdf.xfxQ2(-1, x, Q * Q)
                    + Mypdf.xfxQ2(2, x, Q * Q)
                    + Mypdf.xfxQ2(-2, x, Q * Q)
                    + Mypdf.xfxQ2(3, x, Q * Q)
                    + Mypdf.xfxQ2(-3, x, Q * Q)
                    + Mypdf.xfxQ2(4, x, Q * Q)
                    + Mypdf.xfxQ2(-4, x, Q * Q)
                )
            )
        )
        n3ll_sing = (
            (1 / (4 * np.pi))
            * Mypdf.alphasQ(muR)
            * (1 / (4 * np.pi))
            * Mypdf.alphasQ(muR)
            * PDFConvolute_plus(MasslessCoeffFunc.Cb_2_sing, Mypdf, x, Q, p, nf, h_id)
        )
        res += n3ll_reg + n3ll_local + n3ll_sing
    return res


def F2_M(order, meth, pdf, x, Q, h_id, muF_ratio=1, muR_ratio=1):
    """
    Compute the M result for the structure function F2

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
            nlo_nll_reg = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * PDFConvolute(TildeCoeffFunc.Cg_1_til_reg, Mypdf, x, Q, p, nf, g_id)
            )
            nlo_nll_local = MasslessCoeffFunc.Cb_0_loc(x, Q, p, nf) * (
                Mypdf.xfxQ2(h_id, x, Q * Q) + Mypdf.xfxQ2(-h_id, x, Q * Q)
            )
            res += nlo_nll_reg + nlo_nll_local
        if order >= 2:
            nnlo_nnll_reg = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (
                    (1 / (4 * np.pi))
                    * Mypdf.alphasQ(muR)
                    * (
                        PDFConvolute(
                            TildeCoeffFunc.Cg_2_til_reg, Mypdf, x, Q, p, nf, g_id
                        )
                        + PDFConvolute(TildeCoeffFunc.Cq_2_til_reg, Mypdf, x, Q, p, nf)
                    )
                    + PDFConvolute(MasslessCoeffFunc.Cb_1_reg, Mypdf, x, Q, p, nf, h_id)
                )
            )
            nnlo_nnll_local = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * MasslessCoeffFunc.Cb_1_loc(x, Q, p, nf)
                * (Mypdf.xfxQ2(h_id, x, Q * Q) + Mypdf.xfxQ2(-h_id, x, Q * Q))
            )
            nnlo_nnll_sing = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * PDFConvolute_plus(
                    MasslessCoeffFunc.Cb_1_sing, Mypdf, x, Q, p, nf, h_id
                )
            )
            res += nnlo_nnll_reg + nnlo_nnll_local + nnlo_nnll_sing
        if order >= 3:
            n3lo_n3ll_reg = (((1 / (4 * np.pi)) * Mypdf.alphasQ(muR)) ** 2) * (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (
                    PDFConvolute(TildeCoeffFunc.Cg_3_til_reg, Mypdf, x, Q, p, nf, g_id)
                    + PDFConvolute(TildeCoeffFunc.Cq_3_til_reg, Mypdf, x, Q, p, nf)
                )
                + PDFConvolute(MasslessCoeffFunc.Cb_2_reg, Mypdf, x, Q, p, nf, h_id)
            )
            n3lo_n3ll_local = (
                (((1 / (4 * np.pi)) * Mypdf.alphasQ(muR)) ** 2)
                * MasslessCoeffFunc.Cb_2_loc(x, Q, p, nf)
                * (Mypdf.xfxQ2(h_id, x, Q * Q) + Mypdf.xfxQ2(-h_id, x, Q * Q))
            )
            n3lo_n3ll_sing = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * PDFConvolute_plus(
                    MasslessCoeffFunc.Cb_2_sing, Mypdf, x, Q, p, nf, h_id
                )
            )
            res += n3lo_n3ll_reg + n3lo_n3ll_local + n3lo_n3ll_sing
    if meth == "fonll":
        if order >= 0:
            res += MasslessCoeffFunc.Cb_0_loc(x, Q, p, nf) * (
                Mypdf.xfxQ2(h_id, x, Q * Q) + Mypdf.xfxQ2(-h_id, x, Q * Q)
            )
        if order >= 1:
            nlo_nll_reg = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (
                    PDFConvolute(TildeCoeffFunc.Cg_1_til_reg, Mypdf, x, Q, p, nf, g_id)
                    + PDFConvolute(MasslessCoeffFunc.Cb_1_reg, Mypdf, x, Q, p, nf, h_id)
                )
            )
            nlo_nll_local = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * MasslessCoeffFunc.Cb_1_loc(x, Q, p, nf)
                * (Mypdf.xfxQ2(h_id, x, Q * Q) + Mypdf.xfxQ2(-h_id, x, Q * Q))
            )
            nlo_nll_singular = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * PDFConvolute_plus(
                    MasslessCoeffFunc.Cb_1_sing, Mypdf, x, Q, p, nf, h_id
                )
            )
            res += nlo_nll_reg + nlo_nll_local + nlo_nll_singular
        if order >= 2:

            nnlo_nnll_reg = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (
                    PDFConvolute(TildeCoeffFunc.Cg_2_til_reg, Mypdf, x, Q, p, nf, g_id)
                    + PDFConvolute(TildeCoeffFunc.Cq_2_til_reg, Mypdf, x, Q, p, nf)
                    + PDFConvolute(MasslessCoeffFunc.Cb_2_reg, Mypdf, x, Q, p, nf, h_id)
                )
            )
            nnlo_nnll_local = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * MasslessCoeffFunc.Cb_2_loc(x, Q, p, nf)
                * (Mypdf.xfxQ2(h_id, x, Q * Q) + Mypdf.xfxQ2(-h_id, x, Q * Q))
            )
            nnlo_nnll_sing = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * PDFConvolute_plus(
                    MasslessCoeffFunc.Cb_2_sing, Mypdf, x, Q, p, nf, h_id
                )
            )
            res += nnlo_nnll_reg + nnlo_nnll_local + nnlo_nnll_sing
        if order >= 3:
            n3lo_n3ll_reg = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (
                    PDFConvolute(TildeCoeffFunc.Cg_3_til_reg, Mypdf, x, Q, p, nf, g_id)
                    + PDFConvolute(TildeCoeffFunc.Cq_3_til_reg, Mypdf, x, Q, p, nf)
                    + PDFConvolute(MasslessCoeffFunc.Cb_3_reg, Mypdf, x, Q, p, nf, h_id)
                )
            )
            n3lo_n3ll_local = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * MasslessCoeffFunc.Cb_3_loc(x, Q, p, nf)
                * (Mypdf.xfxQ2(h_id, x, Q * Q) + Mypdf.xfxQ2(-h_id, x, Q * Q))
            )
            n3lo_n3ll_sing = (
                (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * (1 / (4 * np.pi))
                * Mypdf.alphasQ(muR)
                * PDFConvolute_plus(
                    MasslessCoeffFunc.Cb_3_sing, Mypdf, x, Q, p, nf, h_id
                )
            )
            res += n3lo_n3ll_reg + n3lo_n3ll_local + n3lo_n3ll_sing
    return res


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
            MasslessCoeffFunc.CLb_2_loc(x, p, Q, nf)
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
    nl = number_light_flavors(h_id)
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
                    + PDFConvolute(MasslessCoeffFunc.CLb_2_reg, Mypdf, x, Q, p, h_id)
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
