import numpy as np
from numpy.testing import assert_allclose

from dis_tp import zetas


def test_zeta_2():
    assert_allclose(zetas.zeta_2, (np.pi**2) / 6.0)
