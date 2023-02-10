import numpy as np
from numpy.testing import assert_allclose

from dis_tp import TildeCoeffFunc as tf
from dis_tp import MatchingFunc as mf
from dis_tp import MasslessCoeffFunc as cf
from dis_tp import MassiveCoeffFunc as mcf

from dis_tp import tools

from dis_tp.Initialize import Initialize_all
from dis_tp.parameters import charges, default_masses, initialize_theory

h_id = 4
mhq = default_masses(h_id)
initialize_theory(use_grids=True, h_id=h_id, mass=mhq)
e_h = charges(h_id)
Initialize_all(h_id)


def test_Cb1_Mbg1():
    xs = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.456, 0.7]
    Qs = [2,5, 10,25, 100]
    my = []
    ref = []
    for Q in Qs:
        for x in xs:
            p = [mhq, Q, e_h]
            my.append(np.log(Q**2/ mhq**2) *  tf.Cb1_Mbg1(x, p, h_id))
            ref.append(
                tools.Convolute(cf.Cb_1_reg, mf.Mbg_1, x, Q, p, h_id)
                + tools.Convolute_plus_coeff(cf.Cb_1_sing, mf.Mbg_1, x, Q, p, h_id)
                + cf.Cb_1_loc(x, Q, p, h_id) * mf.Mbg_1(x, p, h_id)
            )
    assert_allclose(my, ref)


def test_CLb1_Mbg1():
    xs = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.456, 0.7]
    Qs = [2,5, 10,25, 100]
    my = []
    ref = []
    for Q in Qs:
        for x in xs:
            p = [mhq, Q, e_h]
            my.append(np.log(Q**2/ mhq**2) *  tf.CLb1_Mbg1(x, p, h_id))
            ref.append(
                tools.Convolute(cf.CLb_1_reg, mf.Mbg_1, x, Q, p, h_id)
            )
    assert_allclose(my, ref)


class Test_F2:
    xs = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.456, 0.7]
    Q = 10

    def test_nnlo(self):
        for x in self.xs:
            p = np.array([mhq, self.Q, e_h])
            my_grid = tf.Cg_2_til_reg(x, self.Q, p, h_id)
            my = tf.Cg_2_til_reg(x, self.Q, p, h_id, use_analytic=True)
            assert_allclose(my, my_grid, rtol=4e-4)

            my_grid = tf.Cq_2_til_reg(x, self.Q, p, h_id)
            my = tf.Cq_2_til_reg(x, self.Q, p, h_id, use_analytic=True)
            assert_allclose(my, my_grid, rtol=8e-6)


    def test_n3lo(self):
        for x in self.xs:
            p = np.array([mhq, self.Q, e_h])
            my_grid = tf.Cg_3_til_reg(x, self.Q, p, h_id)
            my = tf.Cg_3_til_reg(x, self.Q, p, h_id, use_analytic=True)
            assert_allclose(my, my_grid, rtol=2e-5)

            my_grid = tf.Cq_3_til_reg(x, self.Q, p, h_id)
            my = tf.Cq_3_til_reg(x, self.Q, p, h_id, use_analytic=True)
            assert_allclose(my, my_grid, rtol=2e-6)


class Test_FL:
    xs = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.456, 0.7]
    Q = 10

    def test_nnlo(self):
        for x in self.xs:
            p = np.array([mhq, self.Q, e_h])
            my_grid = tf.CLg_2_til_reg(x, self.Q, p, h_id)
            my = tf.CLg_2_til_reg(x, self.Q, p, h_id, use_analytic=True)
            assert_allclose(my, my_grid, rtol=3e-4)

            my_grid = tf.CLq_2_til_reg(x, self.Q, p, h_id)
            my = tf.CLq_2_til_reg(x, self.Q, p, h_id, use_analytic=True)
            assert_allclose(my, my_grid, rtol=3e-5, atol=3e-14)

    def test_n3lo(self):
        for x in self.xs:
            p = np.array([mhq, self.Q, e_h])
            my_grid = tf.CLg_3_til_reg(x, self.Q, p, h_id)
            my = tf.CLg_3_til_reg(x, self.Q, p, h_id, use_analytic=True)
            assert_allclose(my, my_grid, rtol=2e-5)

            my_grid = tf.CLq_3_til_reg(x, self.Q, p, h_id)
            my = tf.CLq_3_til_reg(x, self.Q, p, h_id, use_analytic=True)
            assert_allclose(my, my_grid, rtol=3e-7)