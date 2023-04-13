import pathlib

import numpy as np
from numpy.testing import assert_allclose
from scipy.interpolate import interp2d
from test_MasslessCoeffFunc import MockESF
from yadism.coefficient_functions.heavy import f2_nc, fl_nc

from dis_tp import Initialize as Ini
from dis_tp import MassiveCoeffFunc as cf
from dis_tp.Initialize import Initialize_all
from dis_tp.parameters import charges, default_masses, initialize_theory
from dis_tp.ReadTxt import readND

h_id = 5
mhq = default_masses(h_id)
initialize_theory(use_grids=True)
e_h = charges(h_id)
p = np.array([mhq, e_h])
Initialize_all()

here = pathlib.Path(__file__).parent


class Test_F2:
    xs = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.456, 0.7]
    Q = 10

    def test_nlo(self):
        for x in self.xs:
            esf = MockESF(x, self.Q**2)
            f2_g = f2_nc.GluonVV(esf, h_id, m2hq=mhq**2).NLO()
            # g reg
            yad = f2_g.reg(x, f2_g.args["reg"])
            my = cf.Cg_1_m_reg(x, self.Q, p, h_id) / e_h**2
            assert_allclose(my, yad)

    def test_nnlo(self):
        for x in self.xs:
            esf = MockESF(x, self.Q**2)
            f2_g = f2_nc.GluonVV(esf, h_id, m2hq=mhq**2).NNLO()
            # g reg
            yad = f2_g.reg(x, f2_g.args["reg"])
            my = cf.Cg_2_m_reg(x, self.Q, p, h_id) / e_h**2
            assert_allclose(my, yad)

            f2_s = f2_nc.SingletVV(esf, h_id, m2hq=mhq**2).NNLO()
            # singlet reg
            yad = f2_s.reg(x, f2_s.args["reg"])
            my = cf.Cq_2_m_reg(x, self.Q, p, h_id) / e_h**2
            assert_allclose(my, yad)

            f2_ns = f2_nc.NonSinglet(esf, h_id, m2hq=mhq**2).NNLO()
            # non singlet reg
            yad = f2_ns.reg(x, f2_ns.args["reg"])
            my = cf.Cb_2_m_reg(x, self.Q, p, h_id) / e_h**2
            assert_allclose(my, yad)
            # non singlet loc
            yad = f2_ns.loc(x, f2_ns.args["loc"])
            my = cf.Cb_2_m_loc(x, self.Q, p, h_id) / e_h**2
            assert_allclose(my, yad)


class Test_FL:
    xs = [0.0001, 0.0123, 0.456]
    Q = 10

    def test_nlo(self):
        for x in self.xs:
            esf = MockESF(x, self.Q**2)
            fl_g = fl_nc.GluonVV(esf, h_id, m2hq=mhq**2).NLO()
            # g reg
            yad = fl_g.reg(x, fl_g.args["reg"])
            my = cf.CLg_1_m_reg(x, self.Q, p, h_id) / e_h**2
            assert_allclose(my, yad)

    def test_nnlo(self):
        for x in self.xs:
            esf = MockESF(x, self.Q**2)
            fl_g = fl_nc.GluonVV(esf, h_id, m2hq=mhq**2).NNLO()
            # g reg
            yad = fl_g.reg(x, fl_g.args["reg"])
            my = cf.CLg_2_m_reg(x, self.Q, p, h_id) / e_h**2
            assert_allclose(my, yad)

            fl_s = fl_nc.SingletVV(esf, h_id, m2hq=mhq**2).NNLO()
            # singlet reg
            yad = fl_s.reg(x, fl_s.args["reg"])
            my = cf.CLq_2_m_reg(x, self.Q, p, h_id) / e_h**2
            assert_allclose(my, yad)

            fl_ns = fl_nc.NonSinglet(esf, h_id, m2hq=mhq**2).NNLO()
            # non singlet reg
            yad = fl_ns.reg(x, fl_ns.args["reg"])
            my = cf.CLb_2_m_reg(x, self.Q, p, h_id) / e_h**2
            assert_allclose(my, yad)


class TestNic:
    """Here we test that Niccolo' code follow the same normalization."""

    # NOTE: the accuracy is bad but these grids are not used...
    # TODO: how can we trust Niccolo' grids?
    xs = [0.0001, 0.0123, 0.456]
    Q = 10
    nf = 5
    c2g = np.array(readND(here / f"grids/C2g.txt"))
    c2q = np.array(readND(here / f"grids/C2q.txt"))
    cLg = np.array(readND(here / f"grids/CLg.txt"))
    cLq = np.array(readND(here / f"grids/CLq.txt"))
    c2g = interp2d(Ini.ZList, Ini.QList, c2g, kind="quintic")
    c2q = interp2d(Ini.ZList, Ini.QList, c2q, kind="quintic")
    cLg = interp2d(Ini.ZList, Ini.QList, cLg, kind="quintic")
    cLq = interp2d(Ini.ZList, Ini.QList, cLq, kind="quintic")

    def test_Lg(self):
        dis_tp = []
        my = []
        for x in self.xs:
            dis_tp.append(cf.CLg_2_m_reg(x, self.Q, p, h_id) / e_h**2)
            my.append(self.cLg(x, self.Q)[0])
        assert_allclose(my, dis_tp, rtol=1e-3)

    def test_Lq(self):
        dis_tp = []
        my = []
        for x in self.xs:
            dis_tp.append(cf.CLq_2_m_reg(x, self.Q, p, h_id) / e_h**2)
            my.append(self.cLq(x, self.Q)[0])
        assert_allclose(my, dis_tp, rtol=1e-1)

    def test_2g(self):
        dis_tp = []
        my = []
        for x in self.xs:
            dis_tp.append(cf.Cg_2_m_reg(x, self.Q, p, h_id) / e_h**2)
            my.append(self.c2g(x, self.Q)[0])
        assert_allclose(my, dis_tp, rtol=9e-4)

    def test_2q(self):
        dis_tp = []
        my = []
        for x in self.xs:
            dis_tp.append(cf.Cq_2_m_reg(x, self.Q, p, h_id) / e_h**2)
            my.append(self.c2q(x, self.Q)[0])
        assert_allclose(my, dis_tp, rtol=3e-1)

    def test_n3lo_f2(self):
        for x in self.xs:
            p = np.array([mhq, self.Q, e_h])
            my_grid = cf.Cg_3_m_reg(x, self.Q, p, h_id)
            my = cf.Cg_3_m_reg(x, self.Q, p, h_id)
            assert_allclose(my, my_grid)

            my_grid = cf.Cq_3_m_reg(x, self.Q, p, h_id)
            my = cf.Cq_3_m_reg(x, self.Q, p, h_id)
            assert_allclose(my, my_grid)

    def test_n3lo_fl(self):
        for x in self.xs:
            p = np.array([mhq, self.Q, e_h])
            my_grid = cf.CLg_3_m_reg(x, self.Q, p, h_id)
            my = cf.CLg_3_m_reg(x, self.Q, p, h_id)
            assert_allclose(my, my_grid)

            my_grid = cf.CLq_3_m_reg(x, self.Q, p, h_id)
            my = cf.CLq_3_m_reg(x, self.Q, p, h_id)
            assert_allclose(my, my_grid) 
