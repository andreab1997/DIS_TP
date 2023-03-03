import pathlib

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from test_MasslessCoeffFunc import MockESF
from yadism.coefficient_functions.fonll.partonic_channel import (
    PdfMatchingLLNonSinglet,
    PdfMatchingNLLNonSinglet,
    PdfMatchingNNLLNonSinglet,
)

from dis_tp import MatchingFunc as mf
from dis_tp.Initialize import Initialize_all
from dis_tp.parameters import charges, default_masses, initialize_theory
from dis_tp import MasslessCoeffFunc as mlf
from dis_tp import TildeCoeffFunc_light as tf

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



def test_CLb_2_reg():
    xs = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.456, 0.7]
    Q = 100
    my = []
    ref = []
    nf = 3
    p = [mhq, Q, 1]
    for x in xs:
        my.append( mlf.CLb_2_reg(x, Q, p, nf) - mf.P1(p, nf) * mlf.CLb_1_reg(x,Q,p,nf))
        ref.append(tf.CLb_2_til_reg(x,Q,p,nf))
    assert_allclose(my, ref)
