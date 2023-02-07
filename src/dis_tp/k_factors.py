"""Script to produce the Yadism benchmark."""

import lhapdf
import numpy as np
import pandas as pd
import yadism
import yaml

from .runner import Runner
from . import configs, parameters


# TODO: build a proper Loader, NNPDF and DISTP compatible
class TheoryCard:
    def __init__(self, configs, name):
        if isinstance(name, str):
            with open(
                configs["paths"]["theory_cards"] / (name + ".yaml"), encoding="utf-8"
            ) as file:
                th = yaml.safe_load(file)
        else:
            th = name

        if th["TMC"] == 1:
            print("Warning, disable Target Mass Corrections, TMC=0")
            th["TMC"] = 0
        if th["IC"] == 1:
            print("Warning, disable Intrinsic Charm, IC=0")
            th["IC"] = 0
        self.t_card = th

    def yadism_like(self):
        return self.t_card

    def dis_tp_like(self, hid):
        new_t_card = {}
        new_t_card["grids"] = True
        new_t_card["hid"] = hid
        new_t_card["mass"] = parameters.default_masses(hid)
        new_t_card["fns"] = "fonll"
        new_t_card["order"] = "N" * self.t_card["PTO"] + "LO"
        return new_t_card


class Observable_card:
    def __init__(self, configs, name):
        if isinstance(name, str):
            with open(
                configs["paths"]["operator_cards"] / (name + ".yaml"), encoding="utf-8"
            ) as f:
                obs = yaml.safe_load(f)
        else:
            obs = name

        if obs["prDIS"] != "EM":
            print("Warning, setting prDIS = EM")
        if obs["ProjectileDIS"] != "electron":
            print("Warning, setting ProjectileDIS = electron")
        if obs["TargetDIS"] != "proton":
            print("Warning, setting TargetDIS = proton")
        obs["TargetDIS"] = "proton"
        self.o_card = obs

    def yadism_like(self):
        return self.o_card

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


class KfactorRunner:
    def __init__(self, t_card_name, o_card_name, pdf_name):
        cfg = configs.load()
        cfg = configs.defaults(cfg)
        self.theory = TheoryCard(cfg, t_card_name)
        self.observables = Observable_card(cfg, o_card_name)
        self.pdf_name = pdf_name
        self._results = None

    def run_yadism(self):
        output = yadism.run_yadism(
            self.theory.yadism_like(), self.observables.yadism_like()
        )
        yad_pred = output.apply_pdf(lhapdf.mkPDF(self.pdf_name))
        return yad_pred

    def run_dis_tp(self, hid, n_cores):
        # TODO: here we nned to run FO and M type
        restype = "FO"
        runner = Runner(
            self.observables.dis_tp_like(self.pdf_name, restype),
            self.theory.dis_tp_like(hid),
        )
        runner.compute(n_cores)
        return runner.results

    def compute(self, hid, n_cores):
        # TODO: cache results somewhere
        yad_log = self.run_yadism()
        dis_tp_log = self.run_dis_tp(hid, n_cores)
        self._results = self._log(dis_tp_log, yad_log)

    @staticmethod
    def _log(dis_tp_log, yad_log):
        for obs in yad_log:
            my_obs = obs.split("_")[0]
            yad_df = pd.DataFrame(yad_log[obs]).rename(columns={"result": "yadism"})
            dis_tp_df = dis_tp_log[my_obs].rename(columns={"result": "dis_tp"})
            log_df = pd.concat([yad_df, dis_tp_df], axis=1).T.drop_duplicates().T

            # construct some nice log table
            log_df.drop("q", axis=1, inplace=True)
            log_df.drop("y", axis=1, inplace=True)
            log_df.drop("error", axis=1, inplace=True)
            log_df["k-factor"] = log_df.dis_tp / log_df.yadism
        return log_df

    # TODO: add a save mathod
    def save_results(self):
        print(self._results)
