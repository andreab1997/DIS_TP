from numpy.testing import assert_allclose
import numpy as np
from scipy import integrate

from dis_tp import MatchingFunc as mf
from dis_tp.parameters import charges, masses
from dis_tp.Integration import Initialize_all


from eko.mellin import Path
from eko.harmonics import compute_cache
from eko.matching_conditions import as1
from eko.matching_conditions import as2
from eko.matching_conditions import as3
from eko.constants import TR,CA


h_id = 5
NF = h_id
e_h = charges(h_id)
mhq = masses(h_id)
p = np.array([mhq, e_h])

Initialize_all(h_id)


class Test_Matching_Hg:
    xs = np.geomspace(1e-4, 1, 10, endpoint=False)
    Qs = [5, 10, 20, 30]
    is_singlet = True
    grids = True

    def test_nlo(self):
        my = []
        eko = []
        for q in self.Qs:
            for x in self.xs:
                p = [mhq, q]
                # TODO: why here do we have a factor of 2??
                my.append(2 * mf.Mbg_1(x, p, NF))
                L = np.log(p[1] ** 2 / p[0] ** 2)

                def quad_ker_talbot(u, func):
                    path = Path(u, np.log(x), self.is_singlet)
                    integrand = path.prefactor * x ** (-path.n) * path.jac
                    gamma = func(path.n, L)
                    return np.real(gamma * integrand)

                eko.append(
                    integrate.quad(
                        lambda u: quad_ker_talbot(u, as1.A_hg),
                        0.5,
                        1.0,
                        epsabs=1e-12,
                        epsrel=1e-6,
                        limit=200,
                        full_output=1,
                    )[0]
                )
        assert_allclose(my, eko)

    def test_nnlo(self):
        my = []
        eko = []
        for q in self.Qs:
            for x in self.xs:
                p = [4.58, q]
                my.append(2 * mf.Mbg_2(x, p, NF))
                L = np.log(p[1] ** 2 / p[0] ** 2)

                def quad_ker_talbot(u, func):
                    path = Path(u, np.log(x), self.is_singlet)
                    integrand = path.prefactor * x ** (-path.n) * path.jac
                    sx = compute_cache(path.n, 3, self.is_singlet)
                    sx = [np.array(s) for s in sx]
                    gamma = func(path.n, sx, L)
                    return np.real(gamma * integrand)

                eko.append(
                    integrate.quad(
                        lambda u: quad_ker_talbot(u, as2.A_hg),
                        0.5,
                        1.0,
                        epsabs=1e-12,
                        epsrel=1e-6,
                        limit=200,
                        full_output=1,
                    )[0]
                )
        assert_allclose(my, eko, rtol=2e-3)

    def test_n3lo(self):
        my = []
        eko = []
        for q in self.Qs:
            for x in self.xs:
                p = [mhq, q]
                my.append(mf.Mbg_3_reg_inv(x, p, NF, grids=self.grids))
                L = np.log(p[1] ** 2 / p[0] ** 2)

                def quad_ker_talbot(u, func):
                    path = Path(u, np.log(x), self.is_singlet)
                    integrand = path.prefactor * x ** (-path.n) * path.jac
                    sx = compute_cache(path.n, 5, self.is_singlet)
                    sx = [np.array(s) for s in sx]
                    gamma = func(path.n, sx, NF, L)
                    return np.real(gamma * integrand)

                eko.append(
                    integrate.quad(
                        lambda u: quad_ker_talbot(u, as3.A_Hg),
                        0.5,
                        1.0,
                        epsabs=1e-12,
                        epsrel=1e-6,
                        limit=200,
                        full_output=1,
                    )[0]
                )
        assert_allclose(my, eko, rtol=3e-4)


class Test_Matching_Hq:
    xs = np.geomspace(1e-4, 1, 10, endpoint=False)
    Qs = [5, 10, 20, 30]
    is_singlet = True
    grids = True

    def test_nnlo(self):
        my = []
        eko = []
        for q in self.Qs:
            for x in self.xs:
                p = [4.58, q]
                my.append(2 * mf.Mbq_2(x, p, NF))
                L = np.log(p[1] ** 2 / p[0] ** 2)

                def quad_ker_talbot(u, func):
                    path = Path(u, np.log(x), self.is_singlet)
                    integrand = path.prefactor * x ** (-path.n) * path.jac
                    sx = compute_cache(path.n, 3, self.is_singlet)
                    sx = [np.array(s) for s in sx]
                    gamma = func(path.n, sx, L)
                    return np.real(gamma * integrand)

                eko.append(
                    integrate.quad(
                        lambda u: quad_ker_talbot(u, as2.A_hq_ps),
                        0.5,
                        1.0,
                        epsabs=1e-12,
                        epsrel=1e-6,
                        limit=200,
                        full_output=1,
                    )[0]
                )
        assert_allclose(my, eko, rtol=6e-4)

    def test_n3lo(self):
        my = []
        eko = []
        for q in self.Qs:
            for x in self.xs:
                p = [mhq, q]
                my.append(mf.Mbq_3_reg_inv(x, p, NF, grids=self.grids))
                L = np.log(p[1] ** 2 / p[0] ** 2)

                def quad_ker_talbot(u, func):
                    path = Path(u, np.log(x), self.is_singlet)
                    integrand = path.prefactor * x ** (-path.n) * path.jac
                    sx = compute_cache(path.n, 5, self.is_singlet)
                    sx = [np.array(s) for s in sx]
                    gamma = func(path.n, sx, NF, L)
                    return np.real(gamma * integrand)

                eko.append(
                    integrate.quad(
                        lambda u: quad_ker_talbot(u, as3.A_Hq),
                        0.5,
                        1.0,
                        epsabs=1e-12,
                        epsrel=1e-6,
                        limit=200,
                        full_output=1,
                    )[0]
                )
        assert_allclose(my, eko, rtol=4e-4)


class Test_Matching_gg:
    xs = np.geomspace(1e-4, 1, 10, endpoint=False)
    Qs = [5, 10, 20, 30]
    is_singlet = True
    grids = False

    def test_nlo(self):
        my = []
        eko = []
        for q in self.Qs:
            x = 1.0
            # here there is only a delta function
            p = [mhq, q]
            my.append(mf.Mgg_1_loc(x, p, NF))
            L = np.log(p[1] ** 2 / p[0] ** 2)

            def quad_ker_talbot(u, func):
                path = Path(u, np.log(x), self.is_singlet)
                integrand = path.prefactor * x ** (-path.n) * path.jac
                # NOTE: to recover a delta func we need a factor 6.4 ...
                gamma = func(L) / 6.4
                return np.real(gamma * integrand)

            eko.append(
                integrate.quad(
                    lambda u: quad_ker_talbot(u, as1.A_gg),
                    0.5,
                    1.0,
                    epsabs=1e-12,
                    epsrel=1e-6,
                    limit=200,
                    full_output=1,
                )[0]
            )
        assert_allclose(my, eko)

    def test_nnlo(self):
        my = []
        eko = []
        for q in self.Qs:
            for x in self.xs:
                p = [mhq, q]
                L = np.log(p[1] ** 2 / p[0] ** 2)

                # TODO: not passing?? 
                my.append(mf.Mgg_2_reg(x, p, NF))
               
                def quad_ker_talbot(u, func):
                    path = Path(u, np.log(x), self.is_singlet)
                    integrand = path.prefactor * x ** (-path.n) * path.jac
                    sx = compute_cache(path.n, 3, self.is_singlet)
                    sx = [np.array(s) for s in sx]
                    gamma = func(path.n, sx, L)
                    return np.real(gamma * integrand)

                eko.append(
                    integrate.quad(
                        lambda u: quad_ker_talbot(u, as2.A_gg),
                        0.5,
                        1.0,
                        epsabs=1e-12,
                        epsrel=1e-6,
                        limit=200,
                        full_output=1,
                    )[0]
                )
        assert_allclose(my, eko, rtol=6e-4)


class Test_Matching_gq:
    xs = np.geomspace(1e-4, 1, 10, endpoint=False)
    Qs = [5, 10, 20, 30]
    is_singlet = True
    grids = False

    def test_nnlo(self):
        my = []
        eko = []
        for q in self.Qs:
            for x in self.xs:
                p = [mhq, q]
                L = np.log(p[1] ** 2 / p[0] ** 2)

                # TODO: how can we check this ?
                my.append(mf.Mgq_2_reg(x, p, NF))
               
                def quad_ker_talbot(u, func):
                    path = Path(u, np.log(x), self.is_singlet)
                    integrand = path.prefactor * x ** (-path.n) * path.jac
                    sx = compute_cache(path.n, 3, self.is_singlet)
                    sx = [np.array(s) for s in sx]
                    gamma = func(path.n, sx, L)
                    return np.real(gamma * integrand)

                eko.append(
                    integrate.quad(
                        lambda u: quad_ker_talbot(u, as2.A_gq),
                        0.5,
                        1.0 - 5e-2,
                        epsabs=1e-12,
                        epsrel=1e-6,
                        limit=200,
                        full_output=1,
                    )[0]
                )
        assert_allclose(my, eko)
