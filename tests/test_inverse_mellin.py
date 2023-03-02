import numpy as np
import pytest
from numpy.testing import assert_allclose

from dis_tp.MatchingFunc import Mbg_3_reg, Mbq_3_reg
from dis_tp.parameters import initialize_theory

Q = 10.0
nf = 4
masses = [1.51, 4.92, 172.5]
mass = masses[nf - 4]
initialize_theory(use_grids=False, masses=masses)


def test_Talbot_parameters():
    """benchmark variation of the parameters of the Talbot path"""
    for x in np.geomspace(1e-4, 1, 10, endpoint=False):
        tmp1 = Mbq_3_reg(x, [Q, mass], nf, r=2, s=0)
        tmp2 = Mbq_3_reg(x, [Q, mass], nf)
        assert_allclose(tmp1, tmp2, rtol=1e-6)

    for x in np.geomspace(1e-4, 1, 10, endpoint=False):
        tmp1 = Mbg_3_reg(x, [Q, mass], nf, r=2, s=0)
        tmp2 = Mbg_3_reg(x, [Q, mass], nf)
        assert_allclose(tmp1, tmp2, rtol=1e-6)


def test_Talbot_linear_path():
    """benchmark Talbot path against linear path"""
    for x in np.geomspace(1e-4, 1.0, 10, endpoint=False):
        tmp1 = Mbq_3_reg(x, [Q, mass], nf)
        tmp2 = Mbq_3_reg(x, [Q, mass], nf, path="linear")
        assert_allclose(tmp1, tmp2, rtol=1e-6)

    for x in np.geomspace(1e-4, 1.0, 10, endpoint=False):
        tmp1 = Mbg_3_reg(x, [Q, mass], nf)
        tmp2 = Mbg_3_reg(x, [Q, mass], nf, path="linear")
        assert_allclose(tmp1, tmp2, rtol=1e-6)


def test_linear_parameters():
    """benchmark variation of the parameters of the linear path"""
    for x in np.geomspace(1e-4, 1.0, 10, endpoint=False):
        tmp1 = Mbq_3_reg(x, [Q, mass], nf, path="linear")
        tmp2 = Mbq_3_reg(x, [Q, mass], nf, r=2.0, s=2.5, path="linear")
        assert_allclose(tmp1, tmp2, rtol=1e-6)

    for x in np.geomspace(1e-4, 1.0, 10, endpoint=False):
        tmp1 = Mbg_3_reg(x, [Q, mass], nf, path="linear")
        tmp2 = Mbg_3_reg(x, [Q, mass], nf, r=2.0, s=2.5, path="linear")
        assert_allclose(tmp1, tmp2, rtol=1e-6)


def test_error():
    with pytest.raises(NotImplementedError):
        Mbg_3_reg(
            0.1, [Q, mass], nf, path="qesti_nso_du_dorsali_so_du_ali_de_pipistrello"
        )
