import lhapdf
import numpy as np
from scipy import integrate

from ..parameters import charges


def mkPDF(pdf_name, order):
    lhapdf.setVerbosity(0)
    Mypdf = None
    if isinstance(pdf_name, list):
        Mypdf = lhapdf.mkPDF(pdf_name[order - 1], 0)
    elif isinstance(pdf_name, str):
        Mypdf = lhapdf.mkPDF(pdf_name, 0)
    return Mypdf


def non_singlet_pdf(pdf, x, Q, nf):
    """Return the `NonSinglet` flavor combination."""
    light_f = [1, 2, 3]
    if nf >= 4:
        light_f.append(4)
    if nf >= 5:
        light_f.append(5)
    return np.sum(
        [
            charges(nl) ** 2 * (pdf.xfxQ2(nl, x, Q * Q) + pdf.xfxQ2(-nl, x, Q * Q))
            for nl in light_f
        ]
    )


def singlet_pdf(pdf, x, Q, nf):
    """Return the `Singlet` flavor combination."""
    light_f = [1, 2, 3]
    if nf >= 4:
        light_f.append(4)
    if nf >= 5:
        light_f.append(5)
    return np.sum(
        [(pdf.xfxQ2(nl, x, Q * Q) + pdf.xfxQ2(-nl, x, Q * Q)) for nl in light_f]
    )


def PDFConvolute_light(func1, pdf, x, Q, p1, nf):
    result, _ = integrate.quad(
        lambda z: func1(z, Q, p1, nf) * non_singlet_pdf(pdf, x / z, Q, nf),
        x,
        1.0,
        epsabs=1e-12,
        epsrel=1e-6,
        limit=200,
        points=(x, 1.0),
    )
    return result


def PDFConvolute_light_singlet(func1, pdf, x, Q, p1, nf):
    result, _ = integrate.quad(
        lambda z: func1(z, Q, p1, nf) * singlet_pdf(pdf, x / z, Q, nf),
        x,
        1.0,
        epsabs=1e-12,
        epsrel=1e-6,
        limit=200,
        points=(x, 1.0),
    )
    return result


def PDFConvolute_light_plus(func1, pdf, x, Q, p1, nf):
    result, _ = integrate.quad(
        lambda z: func1(z, Q, p1, nf)
        * (non_singlet_pdf(pdf, x / z, Q, nf) - non_singlet_pdf(pdf, x, Q, nf)),
        x,
        1.0,
        epsabs=1e-12,
        epsrel=1e-6,
        limit=200,
        points=(x, 1.0),
    )
    return result
