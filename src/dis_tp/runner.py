import functools
import pathlib
from multiprocessing import Pool

import numpy as np

from dis_tp import Integration as Int

from . import configs, io
from .parameters import number_active_flavors

maporders = {"LO": 0, "NLO": 1, "NNLO": 2, "N3LO": 3}
mapfunc = {
    "F2": {"R": [Int.F2_R], "M": [Int.F2_M], "FO": [Int.F2_FO]},
    "FL": {"R": [Int.FL_R], "M": [Int.FL_M], "FO": [Int.FL_FO]},
    "XSHERANCAVG": {
        "R": [Int.F2_R, Int.FL_R],
        "M": [Int.F2_M, Int.FL_M],
        "FO": [Int.F2_FO, Int.FL_FO],
    },
}

# TODO: rename External to be grids
class Runner:
    def __init__(self, o_card, t_card) -> None:

        cfg = configs.load()
        cfg = configs.defaults(cfg)
        dest_path = cfg["paths"]["results"]
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
        self.partial_sf = None

    @staticmethod
    def compute_xs(ob, sfs):
        """Assembly the XS if needed according to 'XSHERANCAVG'"""
        yp = 1.0 + (1.0 - ob.y_grid) ** 2
        yL = ob.y_grid**2
        factors = np.array([[1.0 for _ in ob.y_grid], (-yL / yp).tolist()])
        return np.sum(factors.T @ sfs, axis=0)

    def compute_sf(self, kins):
        x, q = kins
        print(f"x={x}, Q={q}")
        return float(self.partial_sf(x=x, Q=q))

    def compute(self, n_cores):
        order = maporders[self.t_par.order]
        # loop on observables
        for ob in self.o_par.obs:
            func_to_call = mapfunc[ob.name][ob.restype]
            thisob_res = []

            # loop on SF
            for func in func_to_call:
                # TODO: eliminate this if
                if func in [Int.F2_M, Int.FL_M]:
                    self.partial_sf = functools.partial(
                        func,
                        order=order,
                        meth=self.t_par.fns,
                        pdf=ob.pdf,
                        h_id=self.t_par.hid,
                    )
                else:
                    self.partial_sf = functools.partial(
                        func,
                        order=maporders[self.t_par.order],
                        pdf=ob.pdf,
                        h_id=self.t_par.hid,
                    )
                print(f"Start computation of {func.__name__} ...")
                args = (self.compute_sf, zip(ob.x_grid, ob.q_grid))
                if n_cores == 1:
                    sf_res = map(*args)
                else:
                    with Pool(n_cores) as pool:
                        sf_res = pool.map(*args)

                thisob_res.append(sf_res)
            thisob_res = np.array(thisob_res)
            if "XSHERANCAVG" in ob.name:
                thisob_res = self.compute_xs(ob, thisob_res)
            self.runparameters.results[ob] = thisob_res

    def save_results(self):
        self.runparameters.dump_results()
