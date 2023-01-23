from numpy.testing import assert_allclose
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
from yadism.coefficient_functions.heavy import f2_nc, fl_nc
from dis_tp import MassiveCoeffFunc as cf
from dis_tp.parameters import charges, masses
from dis_tp.Integration import Initialize_all

from test_MasslessCoeffFunc import MockESF

h_id = 5
NF = h_id
e_h = charges(h_id)
mhq = masses(h_id)
p = np.array([mhq, e_h])

Initialize_all(h_id)

class Test_F2:
    xs = [0.0001, 0.0123, 0.456]
    Q = 10

    def test_nlo(self):
        for x in self.xs:
            esf = MockESF(x, self.Q**2)
            f2_g = f2_nc.GluonVV(esf, NF, m2hq=mhq**2).NLO()
            # g reg
            yad = f2_g.reg(x, f2_g.args["reg"])
            # TODO: here it seems there is no NF, as it was in the massless,
            # is this consistent?
            my = cf.Cg_1_m_reg(x, self.Q, p, NF) / e_h**2
            assert_allclose(my, yad)

    def test_nnlo(self):
        for x in self.xs:
            esf = MockESF(x, self.Q**2)
            f2_g = f2_nc.GluonVV(esf, NF, m2hq=mhq**2).NNLO()
            # g reg
            yad = f2_g.reg(x, f2_g.args["reg"])
            my = cf.Cg_2_m_reg(x, self.Q, p, NF) / e_h**2
            assert_allclose(my, yad, rtol=6e-2)

            f2_s = f2_nc.SingletVV(esf, NF, m2hq=mhq**2).NNLO()
            # singlet reg
            yad = f2_s.reg(x, f2_s.args["reg"])
            my = cf.Cq_2_m_reg(x, self.Q, p, NF) / e_h**2
            assert_allclose(my, yad, rtol=6e-2)




class Test_FL:
    xs = [0.0001, 0.0123, 0.456]
    Q = 10

    def test_nlo(self):
        for x in self.xs:
            esf = MockESF(x, self.Q**2)
            fl_g = fl_nc.GluonVV(esf, NF, m2hq=mhq**2).NLO()
            # g reg
            yad = fl_g.reg(x, fl_g.args["reg"])
            my = cf.CLg_1_m_reg(x, self.Q, p, NF) / e_h**2
            assert_allclose(my, yad)

    def test_nnlo(self):
        for x in self.xs:
            esf = MockESF(x, self.Q**2)
            fl_g = fl_nc.GluonVV(esf, NF, m2hq=mhq**2).NNLO()
            # g reg
            yad = fl_g.reg(x, fl_g.args["reg"])
            my = cf.CLg_2_m_reg(x, self.Q, p, NF) / e_h**2
            assert_allclose(my, yad, rtol=6e-2)

            fl_s = fl_nc.SingletVV(esf, NF, m2hq=mhq**2).NNLO()
            # singlet reg
            yad = fl_s.reg(x, fl_s.args["reg"])
            my = cf.CLq_2_m_reg(x, self.Q, p, NF) / e_h**2
            assert_allclose(my, yad, rtol=6e-2)

