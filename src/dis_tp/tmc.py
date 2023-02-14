import numpy as np
from scipy import integrate


class TMC_StructureFunction:
    """Compute TMC structure function."""

    def __init__(self, target_mass, f2, fl=None):
        self.f2 = f2
        self.fl = fl
        self.target_mass = target_mass
        self.kind = "F2" if self.fl is None else "FL"

    def __call__(self, x, Q):
        self.x = x
        self.Q = Q

        # compute variables
        self.mu = self.target_mass**2 / self.Q**2
        self.rho = np.sqrt(1 + 4 * self.x**2 * self.mu)  # = r = sqrt(tau)
        self.xi = 2 * self.x / (1 + self.rho)

        # run actual computaion
        if self.fl is None:
            return self.tmc_f2()
        return self.tmc_fl()

    def h2(self):
        # TODO: here you need to use interpolation this is too slow
        return integrate.quad(
            lambda u: self.f2(x=u, Q=self.Q) / u,
            self.xi,
            1,
        )

    # def tmc_f2_apfel(self):
    #     _factor_shifted = self.x**2 / (self.xi**2 * self.rho**3)
    #     _factor_h2 = 6.0 * self.mu * self.x**3 / (self.rho**4)
    #     F2out = self.f2(x=self.xi, Q=self.Q)
    #     h2out = self.h2()
    #     return _factor_shifted * F2out + _factor_h2 * h2out

    # def tmc_fl_apfel(self):
    #     _factor_shifted = self.x**2 / (self.xi**2 * self.rho)
    #     _factor_h2 = 4.0 * self.mu * self.x**3 / (self.rho**2)
    #     FLout = self.fl(x=self.xi, Q=self.Q)
    #     h2out = self.h2()
    #     return _factor_shifted * FLout + _factor_h2 * h2out

    def tmc_f2(self):
        _factor_shifted = self.x**2 / (self.xi**2 * self.rho**3)
        approx_prefactor = _factor_shifted * (
            1 + ((6 * self.mu * self.x * self.xi) / self.rho) * (1 - self.xi) ** 2
        )
        F2out = self.f2(x=self.xi, Q=self.Q)
        return approx_prefactor * F2out

    def tmc_fl(self):
        _factor_shifted = self.x**2 / (self.xi**2 * self.rho)
        approx_prefactor_F2 = _factor_shifted * (
            (4 * self.mu * self.x * self.xi) / self.rho * (1 - self.xi)
            + 8
            * (self.mu * self.x * self.xi / self.rho) ** 2
            * (-np.log(self.xi) - 1 + self.xi)
        )
        FLout = self.fl(x=self.xi, Q=self.Q)
        F2out = self.f2(x=self.xi, Q=self.Q)
        return _factor_shifted * FLout + approx_prefactor_F2 * F2out
