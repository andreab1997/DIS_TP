# tool functions to compute convolutions between PDFs and coefficients functions but also between matching functions and
# coefficient functions and between matching functions alone

import numpy as np
import scipy.integrate as integrate


def PDFConvolute(func1, pdf, x, Q, p1, nf, pid=None):
    np.seterr(invalid="ignore")
    lower = x
    upper = 1.0
    if pid == 21:
        result, error = integrate.quad(
            lambda z: func1(z, Q, p1, nf) * pdf.xfxQ2(pid, x * (1.0 / z), Q * Q),
            lower,
            upper,
            epsrel=1.0e-02,
            points=(x, 1.0),
        )
    if pid in [4, 5, 6]:
        result, error = integrate.quad(
            lambda z: func1(z, Q, p1, nf)
            * (
                pdf.xfxQ2(pid, x * (1.0 / z), Q * Q)
                + pdf.xfxQ2(-pid, x * (1.0 / z), Q * Q)
            ),
            lower,
            upper,
            epsrel=1.0e-02,
            points=(x, 1.0),
        )
    else:

        def light_pdfs(z, Q):
            light_f = np.sum(
                [pdf.xfxQ2(nl, x * (1.0 / z), Q * Q) for nl in [-1, -2, -3, 1, 2, 3]]
            )
            if nf > 4:
                light_f += np.sum(
                    [pdf.xfxQ2(nl, x * (1.0 / z), Q * Q) for nl in [-4, 4]]
                )
            if nf > 5:
                light_f += np.sum(
                    [pdf.xfxQ2(nl, x * (1.0 / z), Q * Q) for nl in [-5, 5]]
                )
            return light_f

        result, error = integrate.quad(
            lambda z: func1(z, Q, p1, nf) * light_pdfs(z, Q),
            lower,
            upper,
            epsrel=1.0e-02,
            points=(x, 1.0),
        )
    return result


def Convolute(func1, matching, x, Q, p1, nf):
    np.seterr(invalid="ignore")
    lower = x
    upper = 1.0
    q = [p1[0], Q]
    result, error = integrate.quad(
        lambda z: (1.0 / z) * func1(z, Q, p1, nf) * matching(x * (1.0 / z), q, nf),
        lower,
        upper,
        epsrel=1.0e-02,
        points=(x, 1.0),
    )
    return result


def Convolute_matching(matching1, matching2, x, Q, p1, nf):
    np.seterr(invalid="ignore")
    lower = x
    upper = 1.0
    q = [p1[0], Q]
    result, error = integrate.quad(
        lambda z: (1.0 / z) * matching1(z, q, nf) * matching2(x * (1.0 / z), q, nf),
        lower,
        upper,
        epsrel=1.0e-02,
        points=(x, 1.0),
    )
    return result


def PDFConvolute_plus(func1, pdf, x, Q, p1, nf, pid=None):
    if pid == 21:
        plus1, error1 = integrate.quad(
            lambda z: func1(z, Q, p1, nf)
            * (pdf.xfxQ2(pid, x * (1.0 / z), Q * Q) - pdf.xfxQ2(pid, x, Q * Q)),
            x,
            1.0,
            epsrel=1.0e-02,
            points=(x, 1.0),
        )
        plus2, error2 = integrate.quad(
            lambda z: func1(z, Q, p1, nf) * pdf.xfxQ2(pid, x, Q * Q),
            0.0,
            x,
            epsrel=1.0e-02,
            points=(0.0, x),
        )
    if pid in [4, 5, 6]:
        plus1, error1 = integrate.quad(
            lambda z: func1(z, Q, p1, nf)
            * (
                pdf.xfxQ2(pid, x * (1.0 / z), Q * Q)
                + pdf.xfxQ2(-pid, x * (1.0 / z), Q * Q)
                - pdf.xfxQ2(pid, x, Q * Q)
                - pdf.xfxQ2(-pid, x, Q * Q)
            ),
            x,
            1.0,
            epsrel=1.0e-02,
            points=(x, 1.0),
        )
        plus2, error2 = integrate.quad(
            lambda z: func1(z, Q, p1, nf)
            * (pdf.xfxQ2(pid, x, Q * Q) + pdf.xfxQ2(-pid, x, Q * Q)),
            0.0,
            x,
            epsrel=1.0e-02,
            points=(0.0, x),
        )
    else:

        def light_pdfs(z, Q):
            light_f = np.sum(
                [
                    pdf.xfxQ2(nl, x * (1.0 / z), Q * Q) - pdf.xfxQ2(nl, x, Q * Q)
                    for nl in [-1, -2, -3, 1, 2, 3]
                ]
            )
            if nf > 4:
                light_f += np.sum(
                    [
                        pdf.xfxQ2(nl, x * (1.0 / z), Q * Q) - pdf.xfxQ2(nl, x, Q * Q)
                        for nl in [-4, 4]
                    ]
                )
            if nf > 5:
                light_f += np.sum(
                    [
                        pdf.xfxQ2(nl, x * (1.0 / z), Q * Q) - pdf.xfxQ2(nl, x, Q * Q)
                        for nl in [-5, 5]
                    ]
                )
            return light_f

        plus1, error1 = integrate.quad(
            lambda z: func1(z, Q, p1, nf) * light_pdfs,
            x,
            1.0,
            epsrel=1.0e-02,
            points=(x, 1.0),
        )
        plus2, error2 = integrate.quad(
            lambda z: func1(z, Q, p1, nf)
            * (
                pdf.xfxQ2(1, x, Q * Q)
                + pdf.xfxQ2(2, x, Q * Q)
                + pdf.xfxQ2(3, x, Q * Q)
                + pdf.xfxQ2(4, x, Q * Q)
                + pdf.xfxQ2(-1, x, Q * Q)
                + pdf.xfxQ2(-2, x, Q * Q)
                + pdf.xfxQ2(-3, x, Q * Q)
                + pdf.xfxQ2(-4, x, Q * Q)
            ),
            0.0,
            x,
            epsrel=1.0e-02,
            points=(0.0, x),
        )
    return plus1 - plus2


def Convolute_plus_coeff(func1, matching, x, Q, p1, nf):
    np.seterr(invalid="ignore")
    q = [p1[0], Q]
    plus1, error1 = integrate.quad(
        lambda z: func1(z, Q, p1, nf)
        * ((1.0 / z) * matching(x * (1.0 / z), q, nf) - matching(x, q, nf)),
        x,
        1.0,
        epsrel=1.0e-02,
        points=(x, 1.0),
    )
    plus2, error2 = integrate.quad(
        lambda z: func1(z, Q, p1, nf) * matching(x, q, nf),
        0.0,
        x,
        epsrel=1.0e-02,
        points=(0.0, x),
    )
    return plus1 - plus2


def Convolute_plus_matching(func1, matching, x, Q, p1, nf):
    np.seterr(invalid="ignore")
    q = [p1[0], Q]
    plus1, error1 = integrate.quad(
        lambda z: matching(z, q, nf)
        * ((1.0 / z) * func1(x * (1.0 / z), Q, p1, nf) - func1(x, Q, p1, nf)),
        x,
        1.0,
        epsrel=1.0e-02,
        points=(x, 1.0),
    )
    plus2, error2 = integrate.quad(
        lambda z: matching(z, q, nf) * func1(x, Q, p1, nf),
        0.0,
        x,
        epsrel=1.0e-02,
        points=(0.0, x),
    )
    return plus1 - plus2


def Convolute_plus_matching_per_matching(matchingplus, matching2, x, Q, p1, nf):
    np.seterr(invalid="ignore")
    q = [p1[0], Q]
    plus1, error1 = integrate.quad(
        lambda z: matchingplus(z, q, nf)
        * ((1.0 / z) * matching2(x * (1.0 / z), q, nf) - matching2(x, q, nf)),
        x,
        1.0,
        epsrel=1.0e-02,
        points=(x, 1.0),
    )
    plus2, error2 = integrate.quad(
        lambda z: matchingplus(z, q, nf) * matching2(x, q, nf),
        0.0,
        x,
        epsrel=1.0e-02,
        points=(0.0, x),
    )
    return plus1 - plus2
