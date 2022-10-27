from dis_tp import Betas
from numpy.testing import assert_allclose

def test_beta_0():
    assert_allclose(Betas.beta_0(), 11. - (8./3.))

def test_beta_1():
    assert_allclose(Betas.beta_1(), 34.*3. - 20.*2 - 4.*2.*(4./3.))
