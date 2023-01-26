import yaml
import lhapdf
import pandas as pd

from .configs import defaults, detect, load
from .parameters import number_active_flavors


class Observable:
    """Class describing observable settings"""

    def __init__(self, obs, pdf, restype, scalevar, kinematics):
        self.obs = obs
        self.pdf = lhapdf.mkPDF(pdf, 0)
        self.restype = restype
        self.scalevar = scalevar
        self.kinematics = pd.DataFrame(kinematics)


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
                kinematics=loaded["obs"][ob]["kinematics"],
            )
        )
    return OperatorParameters(observables)


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

    def __init__(self, obs):

        self.x_grid = obs.kinematics.x
        self.q_grid = obs.kinematics.q
        self.y_grid = obs.kinematics.y
        self.obs = obs

    def x_grid(self):
        return self.x_grid

    def q_grid(self):
        return self.q_grid

    def y_grid(self):
        return self.y_grid

    def obs(self):
        return self.obs


class RunParameters:
    """Class to hold all the running parameters."""

    def __init__(self, theoryparam, operatorparam, resultpath):
        self.theoryparam = theoryparam
        self.operatorparam = operatorparam
        self.resultpath = resultpath
        self.results = {}

    def theory_parameters(self):
        return self.theoryparam

    def operator_parameters(self):
        return self.operatorparam

    def resultpath(self):
        return self.resultpath

    def dump_results(self):
        for ob, res in self.results.items():
            self.dump_result(ob, res)

    def dump_result(self, ob, ob_result):
        file_name = (
            ob.obs
            + "_"
            + ob.restype
            + "_"
            + str(self.theory_parameters().order)
            + "_"
            + str(self.theory_parameters().hid)
        )
        obs_path = self.resultpath / (file_name + ".yaml")
        # construct the object to dump
        to_dump = dict(
            x_grid=self.operator_parameters().x_grid,
            q_grid=self.operator_parameters().q_grid,
            obs=ob_result,
        )
        with open(obs_path, "w", encoding="UTF-8") as f:
            yaml.safe_dump(to_dump, f)


# TODO: rename External to be grids
# TODO: fix kinematics in runcards
