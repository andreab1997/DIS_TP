import numpy as np
from eko.constants import CA, TR
from eko.mellin import Path
from ekore.harmonics import S1, cache
from ekore.operator_matrix_elements.unpolarized.space_like import as1, as2, as3
from numpy.testing import assert_allclose
from scipy import integrate

from dis_tp import MatchingFunc as mf
from dis_tp.Initialize import Initialize_all
from dis_tp.parameters import charges, default_masses, initialize_theory

h_id = 5
mhq = default_masses(h_id)
initialize_theory(use_grids=True)
e_h = charges(h_id)
p = np.array([mhq, e_h])
Initialize_all()


class Test_Matching_Hg:
    xs = [
        0.0001,
        0.001,
        0.01,
        0.1,
        0.2,
        0.456,
        0.7,
    ]  # np.geomspace(1e-4, 1, 10, endpoint=False)
    Qs = [5, 10, 20, 30]
    is_singlet = True

    def test_nlo(self):
        my = []
        eko = []
        for q in self.Qs:
            for x in self.xs:
                p = [mhq, q]
                my.append(2 * mf.Mbg_1(x, p, h_id))
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
                p = [mhq, q]
                my.append(2 * mf.Mbg_2(x, p, h_id))
                L = np.log(p[1] ** 2 / p[0] ** 2)

                def quad_ker_talbot(u, func):
                    path = Path(u, np.log(x), self.is_singlet)
                    integrand = path.prefactor * x ** (-path.n) * path.jac
                    sx_cache = cache.reset()
                    gamma = func(path.n, sx_cache, L)
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
        assert_allclose(my, eko, rtol=3e-6)

    def test_n3lo(self):
        my = []
        eko = []
        for q in self.Qs:
            for x in self.xs:
                p = [mhq, q]
                my.append(2 * mf.Mbg_3_reg(x, p, h_id))
                L = np.log(p[1] ** 2 / p[0] ** 2)

                def quad_ker_talbot(u, func):
                    path = Path(u, np.log(x), self.is_singlet)
                    integrand = path.prefactor * x ** (-path.n) * path.jac
                    sx_cache = cache.reset()
                    gamma = func(path.n, sx_cache, h_id - 1, L)
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
        assert_allclose(my, eko, rtol=2e-5)


class Test_Matching_Hq:
    xs = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.456, 0.7]
    Qs = [5, 10, 20, 30]
    is_singlet = True

    def test_nnlo(self):
        my = []
        eko = []
        for q in self.Qs:
            for x in self.xs:
                p = [mhq, q]
                my.append(2 * mf.Mbq_2(x, p, h_id))
                L = np.log(p[1] ** 2 / p[0] ** 2)

                def quad_ker_talbot(u, func):
                    path = Path(u, np.log(x), self.is_singlet)
                    integrand = path.prefactor * x ** (-path.n) * path.jac
                    sx_cache = cache.reset()
                    gamma = func(path.n, sx_cache, L)
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
        assert_allclose(my, eko)

    def test_n3lo(self):
        my = []
        eko = []
        for q in self.Qs:
            for x in self.xs:
                p = [mhq, q]
                my.append(2 * mf.Mbq_3_reg(x, p, h_id))
                L = np.log(p[1] ** 2 / p[0] ** 2)

                def quad_ker_talbot(u, func):
                    path = Path(u, np.log(x), self.is_singlet)
                    integrand = path.prefactor * x ** (-path.n) * path.jac
                    sx_cache = cache.reset()
                    gamma = func(path.n, sx_cache, h_id - 1, L)
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
        assert_allclose(my, eko)


class Test_Matching_gg:
    xs = np.geomspace(1e-4, 1, 10, endpoint=False)
    Ns = [2, 3, 4, 5, 6, 7]
    Qs = [5, 10, 20]
    is_singlet = True

    def test_nlo(self):
        my = []
        eko = []
        for q in self.Qs:
            x = 1.0
            # here there is only a delta function
            p = [mhq, q]
            my.append(mf.Mgg_1_loc(x, p, h_id))
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
            p = [mhq, q]
            L = np.log(p[1] ** 2 / p[0] ** 2)
            for n in self.Ns:

                def mellin_integrate(n):
                    return (
                        integrate.quad(
                            lambda x: mf.Mgg_2_reg(x, p, h_id) * x ** (n - 1),
                            0,
                            1,
                            epsabs=1e-12,
                            epsrel=1e-6,
                            limit=200,
                            full_output=1,
                        )[0]
                        - mf.Mgg_2_sing(0, p, h_id) * S1(n - 1)
                        + mf.Mgg_2_loc(0, p, h_id)
                    )

                my.append(mellin_integrate(n))
                sx_cache = cache.reset()
                eko.append(as2.A_gg(n, sx_cache, L))
        assert_allclose(my, eko)


class Test_Matching_qq:
    xs = np.geomspace(1e-4, 1, 10, endpoint=False)
    Ns = [2, 3, 4, 5, 6, 7]
    Qs = [5, 10, 20]
    is_singlet = True

    def test_nnlo(self):
        my = []
        eko = []
        for q in self.Qs:
            p = [mhq, q]
            L = np.log(p[1] ** 2 / p[0] ** 2)
            for n in self.Ns:

                def mellin_integrate(n):
                    return (
                        integrate.quad(
                            lambda x: mf.Mqq_2_reg(x, p, h_id) * x ** (n - 1),
                            0,
                            1,
                            epsabs=1e-12,
                            epsrel=1e-6,
                            limit=200,
                            full_output=1,
                        )[0]
                        - mf.Mqq_2_sing(0, p, h_id) * S1(n - 1)
                        + mf.Mqq_2_loc(0, p, h_id)
                    )

                my.append(mellin_integrate(n))
                sx_cache = cache.reset()
                eko.append(as2.A_qq_ns(n, sx_cache, L))
        assert_allclose(my, eko)


class Test_Matching_gq:
    xs = np.geomspace(1e-4, 1, 10, endpoint=False)
    Qs = [5, 10, 20, 30]
    is_singlet = True

    def test_nnlo(self):
        my = []
        eko = []
        for q in self.Qs:
            for x in self.xs:
                p = [mhq, q]
                L = np.log(p[1] ** 2 / p[0] ** 2)
                my.append(mf.Mgq_2_reg(x, p, h_id))

                def quad_ker_talbot(u, func):
                    path = Path(u, np.log(x), self.is_singlet)
                    integrand = path.prefactor * x ** (-path.n) * path.jac
                    sx_cache = cache.reset()
                    gamma = func(path.n, sx_cache, L)
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


class Test_Grids:
    xs = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.456, 0.7]
    Q = 10

    def test_n3lo(self):
        for x in self.xs:
            p = np.array([mhq, self.Q, e_h])
            my_grid = mf.Mbg_3_reg(x, p, h_id)
            my = mf.Mbg_3_reg(x, p, h_id, use_analytic=True)
            assert_allclose(my, my_grid)

            my_grid = mf.Mbq_3_reg(x, p, h_id)
            my = mf.Mbq_3_reg(x, p, h_id, use_analytic=True)
            assert_allclose(my, my_grid)
