import numpy as np
from numpy.testing import assert_allclose

from dis_tp.MatchingFunc import Mbg_3_reg_inv, Mbq_3_reg_inv

for x in np.geomspace(1e-4, 1, 10, endpoint=False):
    for Q in [5.1, 10, 50, 100]:
        tmp1 = Mbq_3_reg_inv(x, [Q, 5], r=2, s=0)
        tmp2 = Mbq_3_reg_inv(x, [Q, 5])
        assert_allclose(tmp1, tmp2, rtol=1e-6)

for x in np.geomspace(1e-4, 1, 10, endpoint=False):
    for Q in [5.1, 10, 50, 100]:
        tmp1 = Mbg_3_reg_inv(x, [Q, 5], r=2, s=0)
        tmp2 = Mbg_3_reg_inv(x, [Q, 5])
        assert_allclose(tmp1, tmp2, rtol=1e-6)
