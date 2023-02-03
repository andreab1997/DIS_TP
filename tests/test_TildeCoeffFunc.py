import numpy as np
from numpy.testing import assert_allclose

from dis_tp import TildeCoeffFunc as tf
from dis_tp.Integration import Initialize_all
from dis_tp.parameters import charges, default_masses, initialize_theory

h_id = 5
mhq = default_masses(h_id)
initialize_theory(use_grids=False, h_id=h_id, mass=mhq)
e_h = charges(h_id)


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
            assert_allclose(my, my_grid, rtol=7e-7)

    def test_n3lo(self):
        for x in self.xs:
            p = np.array([mhq, self.Q, e_h])
            my_grid = tf.Cg_3_til_reg(x, self.Q, p, h_id)
            my = tf.Cg_3_til_reg(x, self.Q, p, h_id, use_analytic=True)
            assert_allclose(my, my_grid, rtol=2e-5)

            my_grid = tf.Cq_3_til_reg(x, self.Q, p, h_id)
            my = tf.Cq_3_til_reg(x, self.Q, p, h_id, use_analytic=True)
            assert_allclose(my, my_grid, rtol=5e-7, atol=3e-5)


class Test_FL:
    xs = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.456, 0.7]
    Q = 10

    def test_nnlo(self):
        for x in self.xs:
            p = np.array([mhq, self.Q, e_h])
            my_grid = tf.CLg_2_til_reg(x, self.Q, p, h_id)
            my = tf.CLg_2_til_reg(x, self.Q, p, h_id, use_analytic=True)
            assert_allclose(my, my_grid, rtol=4e-4)

            my_grid = tf.CLq_2_til_reg(x, self.Q, p, h_id)
            my = tf.CLq_2_til_reg(x, self.Q, p, h_id, use_analytic=True)
            assert_allclose(my, my_grid, rtol=2e-6, atol=3e-14)
