import pathlib
import numpy as np

from dis_tp import Integration as Int

from . import configs, io
from .parameters import number_active_flavors

maporders = {"LO": 0, "NLO": 1, "NNLO": 2, "N3LO": 3}
mapfunc = {
    "F2": {"R": Int.F2_R, "M": Int.F2_M, "FO": Int.F2_FO},
    "FL": {"R": Int.FL_R, "M": Int.FL_M, "FO": Int.FL_FO},
    "XSHERANCAVG": {
        "R": [Int.F2_R, Int.FL_R],
        "M": [Int.F2_M, Int.FL_M],
        "FO": [Int.F2_FO, Int.FL_FO],
    },
}


# TODO: rename External to be grids


class Runner:
    def __init__(self, o_card, t_card, dest_path: pathlib.Path) -> None:

        cfg = configs.load()
        if isinstance(o_card, str):
            obs_obj = io.load_operator_parameters(cfg, o_card)
        else:
            obs_obj = o_card
        if isinstance(t_card, str):
            th_obj = io.load_theory_parameters(cfg, t_card)
        else:
            th_obj = t_card
        self.runparameters = io.RunParameters(th_obj, obs_obj, dest_path)
        self.o_par = self.runparameters.operator_parameters()
        self.t_par = self.runparameters.theory_parameters()

        # Initializing
        hid = self.t_par.hid
        # TODO: this and the mass can be setted from runcard
        nf = number_active_flavors(hid)
        Int.Initialize_all(nf)

    def compute_xs(self, sfs):
        """Assembly the XS if needed according to 'XSHERANCAVG'"""
        yp = 1.0 + (1.0 - self.o_par.y_grid) ** 2
        yL = self.o_par.y_grid**2
        factors = np.array([1.0, -yL / yp])
        return factors @ sfs

    def compute(self, n_cores):
        order = maporders[self.t_par.order]
        # loop on observables
        for ob in self.o_par.obs:
            func_to_call = mapfunc[ob.obs][ob.restype]
            thisob_res = []

            # loop on SF
            for func in func_to_call:
                sf_res = []

                # TODO: parallelize here
                for x, q in zip(self.o_par.x_grid, self.o_par.q_grid):
                    if func in [Int.F2_M, Int.FL_M]:
                        sf_res.append(
                            float(func(order, "our", ob.pdf, x, q, self.t_par.hid))
                        )
                    else:
                        sf_res.append(float(func(order, ob.pdf, x, q, self.t_par.hid)))
                thisob_res.append(sf_res)
            thisob_res = np.array(thisob_res)
            if "XSHERANCAVG" in ob:
                thisob_res = self.compute_xs(thisob_res, self.o_par)
            self.runparameters.results[ob] = thisob_res

    def save_results(self):
        self.runparameters.dump_results()
