import pandas as pd
import yaml

from .configs import defaults, detect, load
from .parameters import number_active_flavors


class Observable:
    """Class describing observable settings"""

    def __init__(self, name, pdf, restype, kinematics):
        self.name = name
        self.pdf = pdf
        self.restype = restype
        self.kinematics = pd.DataFrame(kinematics)

    @property
    def x_grid(self):
        return self.kinematics.x.values

    @property
    def q_grid(self):
        return self.kinematics.q.values

    @property
    def y_grid(self):
        return self.kinematics.y.values


# TODO: make this NNPDF compatible!!!
def load_theory_parameters(configs, name):
    """Return a TheoryParameters object."""
    if isinstance(name, str):
        with open(
            configs["paths"]["theory_cards"] / (name + ".yaml"), encoding="utf-8"
        ) as f:
            loaded = yaml.safe_load(f)
    else:
        loaded = name
    return TheoryParameters(order=loaded["order"], hid=loaded["hid"], fns=loaded["fns"])


def load_operator_parameters(configs, name):
    """Return a OperatorParameters object."""
    if isinstance(name, str):
        with open(
            configs["paths"]["operator_cards"] / (name + ".yaml"), encoding="utf-8"
        ) as f:
            loaded = yaml.safe_load(f)
    else:
        loaded = name
    observables = []
    for ob in loaded["obs"]:
        observables.append(
            Observable(
                name=ob,
                pdf=loaded["obs"][ob]["PDF"],
                restype=loaded["obs"][ob]["restype"],
                kinematics=loaded["obs"][ob]["kinematics"],
            )
        )
    return OperatorParameters(observables)


class TheoryParameters:
    """Class containing all the theory parameters."""

    def __init__(self, order, hid, fns):
        self.order = order
        self.hid = hid
        self.fns = fns

    def order(self):
        return self.order

    def hid(self):
        return self.hid

    def fns(self):
        return self.fns


class OperatorParameters:
    """Class containing all the operator parameters."""

    def __init__(self, obs):
        self.obs = obs


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
            ob.name
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
            x_grid=ob.x_grid.tolist(),
            q_grid=ob.q_grid.tolist(),
            obs=ob_result.tolist(),
        )
        print(f"Saving results for {ob} in {obs_path}")
        with open(obs_path, "w", encoding="UTF-8") as f:
            yaml.safe_dump(to_dump, f)
