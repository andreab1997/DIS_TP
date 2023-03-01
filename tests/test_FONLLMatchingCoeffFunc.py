import numpy as np
import pathlib
from numpy.testing import assert_allclose
from yadism.coefficient_functions.fonll.partonic_channel import (
    PdfMatchingNNLLNonSinglet,
    PdfMatchingNLLNonSinglet,
    PdfMatchingLLNonSinglet,
)

from dis_tp import MatchingFunc as mf
from dis_tp.Initialize import Initialize_all
from dis_tp.parameters import charges, default_masses, initialize_theory

from test_MasslessCoeffFunc import MockESF
from dis_tp import MatchingFunc as mf

from scipy.integrate import quad

nf = 4
mhq = default_masses(nf)
initialize_theory(use_grids=True)
e_h = charges(nf)
p = np.array([mhq, e_h])
Initialize_all(nf)

here = pathlib.Path(__file__).parent


class Test_Kqq:
    xs = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.456, 0.7]
    Q = 10

    def test_nnlo(self):
        p = np.array([mhq, self.Q, e_h])

        def f(x):
            return x * (1 - x)

        def my_convolute(x):
            reg = quad(
                lambda z: mf.Mqq_2_reg(z, p, nf) * f(x / z) / z,
                x,
                1,
            )[0]
            loc = mf.Mqq_2_loc(x, p, nf) * f(x)
            sing = quad(
                lambda z: mf.Mqq_2_sing(z, p, nf) * (f(x / z) / z - f(x)),
                x,
                1,
            )[0]
            return reg + sing + loc

        def yad_convolute(x, q):
            esf = MockESF(x, q**2)
            matchNNLL = PdfMatchingNNLLNonSinglet(esf, nf, m2hq=mhq**2).NNLO()
            matchNLL = PdfMatchingNLLNonSinglet(esf, nf, m2hq=mhq**2).NNLO()
            matchLL = PdfMatchingLLNonSinglet(esf, nf, m2hq=mhq**2).NNLO()
            loc = (
                matchNNLL.loc(x, matchNNLL.args["loc"])
                + matchNLL.loc(x, matchNLL.args["loc"])
                + matchLL.loc(x, matchLL.args["loc"])
            ) * f(x)
            sing = quad(
                lambda z: (
                    matchNNLL.sing(z, matchNNLL.args["sing"])
                    + matchNLL.sing(z, matchNLL.args["sing"])
                    + matchLL.sing(z, matchLL.args["sing"])
                )
                * (f(x / z) / z - f(x)),
                x,
                1,
            )[0]
            reg = quad(
                lambda z: matchLL.reg(z, matchLL.args["reg"]) * (f(x / z) / z),
                x,
                1,
            )[0]
            return sing + loc + reg

        my = []
        yad = []
        for x in self.xs:
            my.append(my_convolute(x))
            yad.append(yad_convolute(x, self.Q))
        assert_allclose(my, yad)
