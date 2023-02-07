import numpy as np
from numpy.testing import assert_allclose
from yadism.coefficient_functions.light import f2_nc, fl_nc

from dis_tp import MasslessCoeffFunc as cf
from dis_tp.parameters import charges

h_id = 4
e_h = np.array([charges(h_id)])


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
            f2_ns = f2_nc.NonSinglet(esf, h_id).NLO()
            # ns reg
            yad = f2_ns.reg(x, f2_ns.args["reg"])
            my = cf.Cb_1_reg(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)

            # ns loc
            yad = f2_ns.loc(x, f2_ns.args["loc"])
            my = cf.Cb_1_loc(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)
            # ns sing
            yad = f2_ns.sing(x, f2_ns.args["sing"])
            my = cf.Cb_1_sing(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)

            f2_g = f2_nc.Gluon(esf, h_id).NLO()
            # g reg
            yad = f2_g.reg(x, f2_g.args["reg"])
            my = h_id * cf.Cg_1_reg(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)

    def test_nnlo(self):
        for x in self.xs:
            esf = MockESF(x, self.Q**2)
            f2_ns = f2_nc.NonSinglet(esf, h_id).NNLO()
            # ns reg
            yad = f2_ns.reg(x, f2_ns.args["reg"])
            my = cf.Cb_2_reg(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)
            # ns loc, see comment NLO
            yad = f2_ns.loc(x, f2_ns.args["loc"])
            my = cf.Cb_2_loc(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)
            # ns sing
            yad = f2_ns.sing(x, f2_ns.args["sing"])
            my = cf.Cb_2_sing(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)

            f2_g = f2_nc.Gluon(esf, h_id).NNLO()
            # g reg
            yad = f2_g.reg(x, f2_g.args["reg"])
            my = h_id * cf.Cg_2_reg(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)

            f2_s = f2_nc.Singlet(esf, h_id).NNLO()
            # singlet reg
            yad = f2_s.reg(x, f2_s.args["reg"])
            my = h_id * cf.Cq_2_reg(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)

    def test_n3lo(self):
        for x in self.xs:
            esf = MockESF(x, self.Q**2)

            f2_ns = f2_nc.NonSinglet(esf, h_id).N3LO()
            # ns reg
            yad = f2_ns.reg(x, f2_ns.args["reg"])
            my = cf.Cb_3_reg(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)
            # ns loc, see comment NLO
            yad = f2_ns.loc(x, f2_ns.args["loc"])
            my = cf.Cb_3_loc(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)
            # ns sing
            yad = f2_ns.sing(x, f2_ns.args["sing"])
            my = cf.Cb_3_sing(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)

            f2_g = f2_nc.Gluon(esf, h_id).N3LO()
            # g reg
            yad = f2_g.reg(x, f2_g.args["reg"])
            my = h_id * cf.Cg_3_reg(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)
            # g loc
            yad = f2_g.loc(x, f2_g.args["loc"])
            my = h_id * cf.Cg_3_loc(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)

            f2_s = f2_nc.Singlet(esf, h_id).N3LO()
            # singlet reg
            yad = f2_s.reg(x, f2_s.args["reg"])
            my = h_id * cf.Cq_3_reg(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)
            # singlet loc
            yad = f2_s.loc(x, f2_s.args["loc"])
            my = h_id * cf.Cq_3_loc(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)


class Test_FL:
    xs = [0.0001, 0.0123, 0.456]
    Q = 10

    def test_nlo(self):
        for x in self.xs:
            esf = MockESF(x, self.Q**2)
            fl_ns = fl_nc.NonSinglet(esf, h_id).NLO()
            # ns reg
            yad = fl_ns.reg(x, fl_ns.args["reg"])
            my = cf.CLb_1_reg(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)

            fl_g = fl_nc.Gluon(esf, h_id).NLO()
            # g reg
            yad = fl_g.reg(x, fl_g.args["reg"])
            my = h_id * cf.CLg_1_reg(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)

    def test_nnlo(self):
        for x in self.xs:
            esf = MockESF(x, self.Q**2)
            fl_ns = fl_nc.NonSinglet(esf, h_id).NNLO()
            # ns reg
            yad = fl_ns.reg(x, fl_ns.args["reg"])
            my = cf.CLb_2_reg(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)

            # ns loc
            yad = fl_ns.loc(x, fl_ns.args["loc"])
            my = cf.CLb_2_loc(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)

            fl_g = fl_nc.Gluon(esf, h_id).NNLO()
            # g reg
            yad = fl_g.reg(x, fl_g.args["reg"])
            my = h_id * cf.CLg_2_reg(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)

            fl_s = fl_nc.Singlet(esf, h_id).NNLO()
            # singlet reg
            yad = fl_s.reg(x, fl_s.args["reg"])
            my = h_id * cf.CLq_2_reg(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)

    def test_n3lo(self):
        for x in self.xs:
            esf = MockESF(x, self.Q**2)
            fl_ns = fl_nc.NonSinglet(esf, h_id).N3LO()
            # ns reg
            yad = fl_ns.reg(x, fl_ns.args["reg"])
            my = cf.CLb_3_reg(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)

            # ns loc
            yad = fl_ns.loc(x, fl_ns.args["loc"])
            my = cf.CLb_3_loc(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)

            fl_g = fl_nc.Gluon(esf, h_id).N3LO()
            # g reg
            yad = fl_g.reg(x, fl_g.args["reg"])
            my = h_id * cf.CLg_3_reg(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)

            fl_s = fl_nc.Singlet(esf, h_id).N3LO()
            # singlet reg
            yad = fl_s.reg(x, fl_s.args["reg"])
            my = h_id * cf.CLq_3_reg(x, self.Q, e_h, h_id) / e_h**2
            assert_allclose(my, yad)
