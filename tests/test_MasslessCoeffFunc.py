import numpy as np
from numpy.testing import assert_allclose

from yadism.coefficient_functions.light import f2_nc, fl_nc
from dis_tp import MasslessCoeffFunc as cf

NF = 4
e_b = -1 / 3


class MockSF:
    pass


class MockESF:
    def __init__(self, x, q2):
        self.sf = MockSF()
        self.x = x
        self.Q2 = q2


class Test_F2:
    xs = [0.0001, 0.0123, 0.456]
    Q = 10

    def test_nlo(self):
        for x in self.xs:
            esf = MockESF(x, self.Q**2)
            f2_ns = f2_nc.NonSinglet(esf, NF).NLO()
            # ns reg
            yad = f2_ns.reg(x, f2_ns.args["reg"])
            my = cf.Cb_1_reg(x, self.Q, None) / e_b**2
            assert_allclose(my, yad)
            # ns loc
            yad = f2_ns.loc(x, f2_ns.args["loc"])
            my = cf.Cb_1_loc(x, self.Q) / e_b**2
            # TODO: not passing?
            # assert_allclose(my, yad, rtol=3e-3)
            # ns sing
            yad = f2_ns.sing(x, f2_ns.args["sing"])
            my = cf.Cb_1_sing(x, self.Q, None) / e_b**2
            assert_allclose(my, yad)

            f2_g = f2_nc.Gluon(esf, NF).NLO()
            # g reg
            yad = f2_g.reg(x, f2_g.args["reg"])
            # TODO: is this NF correct?
            my = NF * cf.Cg_1_reg(x, self.Q, None) / e_b**2
            assert_allclose(my, yad)

    def test_nnlo(self):
        for x in self.xs:
            esf = MockESF(x, self.Q**2)
            f2_ns = f2_nc.NonSinglet(esf, NF).NNLO()
            # ns reg
            yad = f2_ns.reg(x, f2_ns.args["reg"])
            my = cf.Cb_2_reg(x, self.Q, None) / e_b**2
            # TODO: check this ???
            # assert_allclose(my, yad)
            # ns loc
            yad = f2_ns.loc(x, f2_ns.args["loc"])
            my = cf.Cb_2_loc(x, self.Q) / e_b**2
            # TODO: not passing?
            # assert_allclose(my, yad)
            # ns sing
            yad = f2_ns.sing(x, f2_ns.args["sing"])
            my = cf.Cb_2_sing(x, self.Q, None) / e_b**2
            # TODO: not passing?
            # assert_allclose(my, yad)

            f2_g = f2_nc.Gluon(esf, NF).NNLO()
            # g reg
            yad = f2_g.reg(x, f2_g.args["reg"])
            my = NF * cf.Cg_2_reg(x, self.Q, None) / e_b**2
            assert_allclose(my, yad, rtol=1e-2)

            f2_s = f2_nc.Singlet(esf, NF).NNLO()
            # singlet reg
            yad = f2_s.reg(x, f2_s.args["reg"])
            my = NF * cf.Cq_2_reg(x, self.Q, None) / e_b**2
            assert_allclose(my, yad, rtol=5e-3)


class Test_FL:
    xs = [0.0001, 0.0123, 0.456]
    Q = 10

    def test_nlo(self):
        for x in self.xs:
            esf = MockESF(x, self.Q**2)
            fl_ns = fl_nc.NonSinglet(esf, NF).NLO()
            # ns reg
            yad = fl_ns.reg(x, fl_ns.args["reg"])
            my = cf.CLb_1_reg(x, self.Q, None) / e_b**2
            assert_allclose(my, yad)

            fl_g = fl_nc.Gluon(esf, NF).NLO()
            # g reg
            yad = fl_g.reg(x, fl_g.args["reg"])
            my = NF * cf.CLg_1_reg(x, self.Q, None) / e_b**2
            assert_allclose(my, yad)

    def test_fl_nnlo(self):
        for x in self.xs:
            esf = MockESF(x, self.Q**2)
            fl_ns = fl_nc.NonSinglet(esf, NF).NNLO()
            # ns reg
            yad = fl_ns.reg(x, fl_ns.args["reg"])
            my = cf.CLb_2_reg(x, self.Q, None) / e_b**2
            # TODO: not passing?
            # assert_allclose(my, yad)

            # ns loc
            yad = fl_ns.loc(x, fl_ns.args["loc"])
            my = cf.CLb_2_loc(x, self.Q) / e_b**2
            assert_allclose(my, yad)

            fl_g = fl_nc.Gluon(esf, NF).NNLO()
            # g reg
            yad = fl_g.reg(x, fl_g.args["reg"])
            my = NF * cf.CLg_2_reg(x, self.Q, None) / e_b**2
            assert_allclose(my, yad)

            fl_s = fl_nc.Singlet(esf, NF).NNLO()
            # singlet reg
            yad = fl_s.reg(x, fl_s.args["reg"])
            my = NF * cf.CLq_2_reg(x, self.Q, None) / e_b**2
            assert_allclose(my, yad)
