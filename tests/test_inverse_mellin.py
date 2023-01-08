import numpy as np
from numpy.testing import assert_allclose

from dis_tp.MatchingFunc import Mbg_3_reg_inv, Mbq_3_reg_inv

Q = 10.0

# benchmark variation of the parameters of the Talbot path
for x in np.geomspace(1e-4, 1, 10, endpoint=False):
    tmp1 = Mbq_3_reg_inv(x, [Q, 5], r=2, s=0)
    tmp2 = Mbq_3_reg_inv(x, [Q, 5])
    assert_allclose(tmp1, tmp2, rtol=1e-6)

for x in np.geomspace(1e-4, 1, 10, endpoint=False):
    tmp1 = Mbg_3_reg_inv(x, [Q, 5], r=2, s=0)
    tmp2 = Mbg_3_reg_inv(x, [Q, 5])
    assert_allclose(tmp1, tmp2, rtol=1e-6)


for x in np.geomspace(1e-4, 1.0, 10, endpoint=False):
    tmp1 = Mbq_3_reg_inv(x, [Q, 5])
    tmp2 = Mbq_3_reg_inv(x, [Q, 5], r=1.0, s=1.5, path="linear")
    assert_allclose(tmp1, tmp2, rtol=1e-6)

# benchmark variation of the parameters of the Talbot path

for x in np.geomspace(1e-4, 1.0, 10, endpoint=False):
    tmp1 = Mbq_3_reg_inv(x, [Q, 5], path="linear")
    tmp2 = Mbq_3_reg_inv(x, [Q, 5], r=2.0, s=2.5, path="linear")
    assert_allclose(tmp1, tmp2, rtol=1e-6)
