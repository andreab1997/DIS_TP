import yaml
import numpy as np

from dis_tp import Integration as Int

from .configs import defaults, detect, load
from .parameters import number_active_flavors

maporders = {"LO": 0, "NLO": 1, "NNLO": 2, "N3LO": 3}
mapfunc = {
    "F2": {"R": Int.F2_R, "M": Int.F2_M, "FO": Int.F2_FO},
    "FL": {"R": Int.FL_R, "M": Int.FL_M, "FO": Int.FL_FO},
}


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

        # TODO: fix kinematics in runcard
        self.x_grid = x_grid
        self.q_grid = q_grid
        self.y_grid = None
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

    def theory_parameters(self):
        return self.theoryparam

    def operator_parameters(self):
        return self.operatorparam

    def resultpath(self):
        return self.resultpath

    def dump_results(self, results):
        for ob in results:
            self.dump_result(ob, results[ob])

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


def compute(runparameters, n_cores):
    # Initializing
    hid = runparameters.theory_parameters().hid
    nf = number_active_flavors(hid)
    Int.Initialize_all(nf)

    o_par = runparameters.operator_parameters()
    t_par = runparameters.theory_parameters()

    order = maporders[t_par.order]
    results = {}
    
    # loop on observables
    for ob in o_par.obs:
        func_to_call = mapfunc[ob.obs][ob.restype]
        thisob_res = []
        sfs = []

        # loop o SF
        for func in func_to_call:
            xfix_res = []

            # TODO: parallelize here
            for x in o_par.x_grid:
                for q in o_par.q_grid:
                    if func in [Int.F2_M, Int.FL_M]:
                        xfix_res.append(
                            float(func(order, "our", ob.pdf, x, q, hid))
                        )
                    else:
                        xfix_res.append(float(func(order, ob.pdf, x, q, hid)))
            sfs.append(xfix_res)
        thisob_res.append(xfix_res)

        # Assembly the XS if needed
        if "XSHERANCAVG" in ob:
            yp = 1.0 + (1.0 - o_par.y_grid) ** 2
            # ym = 1.0 - (1.0 - y) ** 2
            yL = o_par.y_grid**2
            factors = np.array([1.0, -yL / yp])
            thisob_res = factors @ thisob_res
        results[ob] = thisob_res
    runparameters.dump_results(results)
