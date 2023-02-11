import numpy as np
import pandas as pd
import yaml

from . import parameters


class TheoryParameters:
    """Class containing all the theory parameters."""

    def __init__(self, order, hid, fns, masses, grids, full_card=None):
        self.order = order
        self.hid = hid
        self.fns = fns
        self.masses = masses
        self.grids = grids
        self._t_card = full_card

    def yadism_like(self):
        return self._t_card

    @classmethod
    def load_card(cls, configs, name, hid):
        """Return a TheoryParameters object."""
        if isinstance(name, str):
            with open(
                configs["paths"]["theory_cards"] / (name + ".yaml"), encoding="utf-8"
            ) as f:
                th = yaml.safe_load(f)
        else:
            th = name

        # Disable some NNPDF features not included here
        if "TMC" in th and th["TMC"] == 1:
            print("Warning, disable Target Mass Corrections, TMC=0")
            th["TMC"] = 0
        if "IC" in th and th["IC"] == 1:
            print("Warning, disable Intrinsic Charm, IC=0")
            th["IC"] = 0

        # compatibility layer
        if "order" in th:
            order = th["order"]
        else:
            order = th["PTO"]

        if "hid" in th:
            hid = th["hid"]

        fns = th.get("fns", "fonll")
        grids = th.get("grids", True)
        mc = th.get("mc", parameters.default_masses(4))
        mb = th.get("mb", parameters.default_masses(5))
        mt = th.get("mt", parameters.default_masses(6))
        masses = [mc, mb, mt]
        # TODO: add here also some settings for alpha_s

        return cls(
            order=order, hid=hid, fns=fns, grids=grids, masses=masses, full_card=th
        )


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


class OperatorParameters:
    """Class containing all the operator parameters."""

    def __init__(self, obs, name, full_card=None):
        self.obs = obs
        self._o_card = full_card
        self.dataset_name = name

    def yadism_like(self):
        return self._o_card

    def dis_tp_like(self, pdf_name, restype):
        new_o_card = {}
        new_o_card["obs"] = {}
        for fx, kins in self.o_card["observables"].items():
            new_kins = [
                {"x": point["x"], "q": np.sqrt(point["Q2"]), "y": point["y"]}
                for point in kins
            ]
            new_o_card["obs"][fx.split("_")[0]] = {
                "PDF": pdf_name,
                "restype": restype,
                "scalevar": False,
                "kinematics": new_kins,
            }
        return new_o_card

    @classmethod
    def load_card(cls, configs, name, pdf_name=None):
        """Return a OperatorParameters object."""
        if isinstance(name, str):
            with open(
                configs["paths"]["operator_cards"] / (name + ".yaml"), encoding="utf-8"
            ) as f:
                obs = yaml.safe_load(f)
        else:
            obs = name

        # Disables some NNPDF settings
        if "prDIS" in obs and obs["prDIS"] != "EM":
            print("Warning, setting prDIS = EM")
        if "ProjectileDIS" in obs and obs["ProjectileDIS"] not in [
            "electron",
            "positron",
        ]:
            print("Warning, setting ProjectileDIS = electron")
        if "TargetDIS" in obs and obs["TargetDIS"] != "proton":
            print("Warning, setting TargetDIS = proton")
            obs["TargetDIS"] = "proton"

        # DIS_TP runcards
        observables = []
        if "obs" in obs:
            observables = []
            for ob in obs["obs"]:
                observables.append(
                    Observable(
                        name=ob,
                        pdf=obs["obs"][ob]["PDF"],
                        restype=obs["obs"][ob]["restype"],
                        kinematics=obs["obs"][ob]["kinematics"],
                    )
                )
        # Yadism runcard
        else:
            for fx, kins in obs["observables"].items():
                new_kins = [
                    {"x": point["x"], "q": np.sqrt(point["Q2"]), "y": point["y"]}
                    for point in kins
                ]
                # TODO: here you are introducing an inconsistency.
                # Heaviness and fns should not coincide ...
                restype = fx.split("_")[1]
                if restype in ["charm", "bottom"]:
                    restype = "FONLL"
                observables.append(
                    Observable(
                        name=fx.split("_")[0],
                        pdf=pdf_name,
                        restype=restype,
                        kinematics=new_kins,
                    )
                )
        return cls(observables, name, obs)


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
            + "_"
            + str(ob.pdf)
        )
        obs_path = self.resultpath / (file_name + ".yaml")
        # construct the object to dump
        to_dump = dict(
            x_grid=ob.x_grid.tolist(),
            q_grid=ob.q_grid.tolist(),
            obs=ob_result.tolist(),
        )
        print(f"Saving results for {ob.name} in {obs_path}")
        with open(obs_path, "w", encoding="UTF-8") as f:
            yaml.safe_dump(to_dump, f)
