import yaml

from .configs import defaults, detect, load


class Observable:
    """Class describing observable settings"""

    def __init__(self, obs, pdf, restype, scalevar):
        self.obs = obs
        self.pdf = pdf
        self.restype = restype
        self.scalevar = scalevar


def load_theory_parameters(configs, name):
    """Return a TheoryParameters object."""
    with open(
        configs["paths"]["theory_cards"] / (name + ".yaml"), encoding="utf-8"
    ) as f:
        loaded = yaml.safe_load(f)
    return TheoryParameters(order=loaded["order"], hid=loaded["hid"])


def load_operator_parameters(configs, name):
    """Return a OperatorParameters object."""

    with open(
        configs["paths"]["operator_cards"] / (name + ".yaml"), encoding="utf-8"
    ) as f:
        loaded = yaml.safe_load(f)
    observables = []
    for ob in loaded["obs"]:
        observables.append(
            Observable(
                obs=ob,
                pdf=loaded["obs"][ob]["PDF"],
                restype=loaded["obs"][ob]["restype"],
                scalevar=loaded["obs"][ob]["scalevar"],
            )
        )
    return OperatorParameters(
        x_grid=loaded["x_grid"], q_grid=loaded["q_grid"], obs=observables
    )


class TheoryParameters:
    """Class containing all the theory parameters."""

    def __init__(self, order, hid):
        self.order = order
        self.hid = hid

    def order(self):
        return self.order

    def hid(self):
        return self.hid


class OperatorParameters:
    """Class containing all the operator parameters."""

    def __init__(self, x_grid, q_grid, obs):
        self.x_grid = x_grid
        self.q_grid = q_grid
        self.obs = obs

    def x_grid(self):
        return self.x_grid

    def q_grid(self):
        return self.q_grid

    def obs(self):
        return self.obs


class RunParameters:
    """Class to hold all the running parameters."""

    def __init__(self, theoryparam, operatorparam, resultpath):
        self.theoryparam = theoryparam
        self.operatorparam = operatorparam
        self.resultpath = resultpath

    def theory_parameters(self):
        return self.theory_parameters

    def operator_parameters(self):
        return self.operator_parameters

    def resultpath(self):
        return self.resultpath
