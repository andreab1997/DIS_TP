from numpy.testing import assert_allclose

from dis_tp import Betas


def test_beta_0():
    assert_allclose(Betas.beta_0(), 11.0 - (8.0 / 3.0))


def test_beta_1():
    assert_allclose(Betas.beta_1(), 34.0 * 3.0 - 20.0 * 2 - 4.0 * 2.0 * (4.0 / 3.0))
