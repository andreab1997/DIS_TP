"""F2 structure function"""
import lhapdf
import numpy as np

from .. import MassiveCoeffFunc, MasslessCoeffFunc, TildeCoeffFunc
from ..parameters import (
    charges,
    masses,
    number_active_flavors,
    number_light_flavors,
    pids,
    default_masses,
)
from ..tools import PDFConvolute, PDFConvolute_plus
from .tools import PDFConvolute_light, PDFConvolute_light_plus, mkPDF, non_singlet_pdf

g_id = pids["g"]


def F2_FO(order, pdf, x, Q, h_id, meth=None, muF_ratio=1, muR_ratio=1):
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
    Mypdf = mkPDF(pdf, order)
    muR = muR_ratio * Q
    p = [masses(h_id), Q, charges(h_id)]
    nf = number_active_flavors(h_id)
    if order >= 0:
        res = 0.0
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


def F2_R(order, pdf, x, Q, h_id, meth=None, muF_ratio=1, muR_ratio=1):
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
    Mypdf = mkPDF(pdf, order)
    muR = muR_ratio * Q
    nf = number_active_flavors(h_id)
    p = [masses(h_id), Q, charges(h_id)]
    if order >= 0:
        res = 0.0
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


def F2_M(order, pdf, x, Q, h_id, meth, muF_ratio=1, muR_ratio=1):
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
    Mypdf = mkPDF(pdf, order)
    muR = muR_ratio * Q
    nf = number_active_flavors(h_id)
    p = [masses(h_id), Q, charges(h_id)]
    if meth == "our":
        if order >= 0:
            res = 0.0
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
            res = MasslessCoeffFunc.Cb_0_loc(x, Q, p, nf) * (
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


def F2_Light(order, pdf, x, Q, h_id, meth=None, muR_ratio=1):
    """
    Compute the light contribution for the structure function F2

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
    # TODO: here we fake charge of 1 and add it later...
    # the proper fix would be to remove it from the cf definition
    p = [0, Q, 1]
    nl = number_light_flavors(h_id)
    alphas = 1 / (4 * np.pi) * Mypdf.alphasQ(muR)
    meansq_e = np.mean([charges(nl) ** 2 for nl in range(1, nl + 1)])
    if order >= 0:
        res = MasslessCoeffFunc.Cb_0_loc(x, Q, p, nl) * non_singlet_pdf(Mypdf, x, Q, nl)
    if order >= 1:
        reg = PDFConvolute_light(
            MasslessCoeffFunc.Cb_1_reg, Mypdf, x, Q, p, nl
        ) + nl * meansq_e * PDFConvolute(
            MasslessCoeffFunc.Cg_1_reg, Mypdf, x, Q, p, nl, g_id
        )
        loc = MasslessCoeffFunc.Cb_1_loc(x, Q, p, nl) * non_singlet_pdf(Mypdf, x, Q, nl)
        sing = PDFConvolute_light_plus(MasslessCoeffFunc.Cb_1_sing, Mypdf, x, Q, p, nl)
        res += alphas * (reg + loc + sing)
    if order >= 2:
        reg = PDFConvolute_light(
            MasslessCoeffFunc.Cb_2_reg, Mypdf, x, Q, p, nl
        ) + nl * meansq_e * (
            PDFConvolute(MasslessCoeffFunc.Cg_2_reg, Mypdf, x, Q, p, nl, g_id)
            + PDFConvolute(MasslessCoeffFunc.Cq_2_reg, Mypdf, x, Q, p, nl)
        )
        loc = MasslessCoeffFunc.Cb_2_loc(x, Q, p, nl) * non_singlet_pdf(Mypdf, x, Q, nl)
        sing = PDFConvolute_light_plus(MasslessCoeffFunc.Cb_2_sing, Mypdf, x, Q, p, nl)
        res += alphas**2 * (reg + loc + sing)
    if order >= 3:
        reg = PDFConvolute_light(
            MasslessCoeffFunc.Cb_3_reg, Mypdf, x, Q, p, nl
        ) + nl * meansq_e * (
            PDFConvolute(MasslessCoeffFunc.Cg_3_reg, Mypdf, x, Q, p, nl, g_id)
            + PDFConvolute(MasslessCoeffFunc.Cq_3_reg, Mypdf, x, Q, p, nl)
        )
        loc = MasslessCoeffFunc.Cb_3_loc(x, Q, p, nl) * non_singlet_pdf(
            Mypdf, x, Q, nl
        ) + nl * meansq_e * (
            MasslessCoeffFunc.Cg_3_loc(x, Q, p, nl) * Mypdf.xfxQ2(g_id, x, Q**2)
            + MasslessCoeffFunc.Cq_3_loc(x, Q, p, nl) * non_singlet_pdf(Mypdf, x, Q, nl)
        )
        sing = PDFConvolute_light_plus(MasslessCoeffFunc.Cb_3_sing, Mypdf, x, Q, p, nl)
        res += alphas**3 * (reg + loc + sing)
    return res


def F2_ZM(order, pdf, x, Q, h_id, meth=None, muR_ratio=1):
    """
    Compute the ZM heavy contribution to structure function F2

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
    alphas = 1 / (4 * np.pi) * Mypdf.alphasQ(muR)
    pdfxfx = Mypdf.xfxQ2(h_id, x, Q**2) + Mypdf.xfxQ2(-h_id, x, Q**2)
    if order >= 0:
        res = MasslessCoeffFunc.Cb_0_loc(x, Q, p, nl) * pdfxfx
    if order >= 1:
        reg = PDFConvolute(
            MasslessCoeffFunc.Cb_1_reg, Mypdf, x, Q, p, nl, h_id
        ) + PDFConvolute(MasslessCoeffFunc.Cg_1_reg, Mypdf, x, Q, p, nl, g_id)
        loc = MasslessCoeffFunc.Cb_1_loc(x, Q, p, nl) * pdfxfx
        sing = PDFConvolute_plus(MasslessCoeffFunc.Cb_1_sing, Mypdf, x, Q, p, nl, h_id)
        res += alphas * (reg + loc + sing)
    if order >= 2:
        reg = (
            PDFConvolute(MasslessCoeffFunc.Cb_2_reg, Mypdf, x, Q, p, nl, h_id)
            + PDFConvolute(MasslessCoeffFunc.Cg_2_reg, Mypdf, x, Q, p, nl, g_id)
            + PDFConvolute(MasslessCoeffFunc.Cq_2_reg, Mypdf, x, Q, p, nl, h_id)
        )
        loc = MasslessCoeffFunc.Cb_2_loc(x, Q, p, nl) * pdfxfx
        sing = PDFConvolute_plus(MasslessCoeffFunc.Cb_2_sing, Mypdf, x, Q, p, nl, h_id)
        res += alphas**2 * (reg + loc + sing)
    if order >= 3:
        reg = (
            PDFConvolute(MasslessCoeffFunc.Cb_3_reg, Mypdf, x, Q, p, nl, h_id)
            + PDFConvolute(MasslessCoeffFunc.Cg_3_reg, Mypdf, x, Q, p, nl, g_id)
            + PDFConvolute(MasslessCoeffFunc.Cq_3_reg, Mypdf, x, Q, p, nl, h_id)
        )
        loc = (
            MasslessCoeffFunc.Cb_3_loc(x, Q, p, nl) * pdfxfx
            + MasslessCoeffFunc.Cg_3_loc(x, Q, p, nl) * Mypdf.xfxQ2(g_id, x, Q**2)
            + MasslessCoeffFunc.Cq_3_loc(x, Q, p, nl) * pdfxfx
        )
        sing = PDFConvolute_plus(MasslessCoeffFunc.Cb_3_sing, Mypdf, x, Q, p, nl, h_id)
        res += alphas**3 * (reg + loc + sing)
    return res


def F2_FONLL(order, pdf, x, Q, h_id, meth, muR_ratio=1):
    """
    Compute the Yadism like FONLL structure function F2

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
    # TODO: add a DUMPING option ??
    mhp1 = masses(h_id + 1)
    if Q < mh:
        return F2_FO(order, pdf, x, Q, h_id, muR_ratio=muR_ratio)
    elif Q < mhp1:
        return F2_M(order, pdf, x, Q, h_id, meth, muR_ratio=muR_ratio)
    elif Q >= mhp1:
        return F2_ZM(order, pdf, x, Q, h_id, muR_ratio=muR_ratio)


def F2_Total(order, pdf, x, Q, h_id, meth, muR_ratio=1):
    """
    Compute the total structure function F2.

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
        muR_ratio : float
            ratio to Q of the renormalization scale
    Returns:
        : float
            result
    """
    # TODO: need to add the missing diagrams
    return (
        F2_Light(order, pdf, x, Q, 3, muR_ratio)
        + F2_FONLL(order, meth, pdf, x, Q, 4, muR_ratio=muR_ratio)
        + F2_FONLL(order, meth, pdf, x, Q, 5, muR_ratio=muR_ratio)
        # + F2_FONLL(order, meth, pdf, x, Q, 6, muR_ratio=muR_ratio)
    )
