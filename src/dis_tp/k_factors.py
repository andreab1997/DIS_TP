"""Script to produce the Yadism benchmark."""

import lhapdf
import numpy as np
import pandas as pd
import yadism
import yaml
from datetime import date

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
        if self.t_card["PTO"] == 3:
            new_t_card["order"] = "N3LO"
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
        self.dataset_name = name

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
    def __init__(self, t_card_name, dataset_name, pdf_name, use_yadism):
        cfg = configs.load()
        cfg = configs.defaults(cfg)

        # Load the ymldb file
        with open(
            cfg["paths"]["ymldb"] / f"{dataset_name}.yaml", encoding="utf-8"
        ) as f:
            ymldb = yaml.safe_load(f)

        self.theory = TheoryCard(cfg, t_card_name)

        # TODO: do we need to support operations between FKtables?
        o_card_name = ymldb["operands"][0][0]
        self.observables = Observable_card(cfg, o_card_name)

        self.pdf_name = pdf_name
        self.use_yadism = use_yadism
        self.result_path = cfg["paths"]["results"]
        self.dataset_name = ymldb["target_dataset"]
        self._results = None

    def run_yadism(self):
        output = yadism.run_yadism(
            self.theory.yadism_like(), self.observables.yadism_like()
        )
        yad_pred = output.apply_pdf(lhapdf.mkPDF(self.pdf_name))
        return yad_pred

    def run_dis_tp(self, hid, n_cores):
        # TODO: how do we treat bottom mass effects in this code?
        # TODO: here we need to run FO and M type
        restype = "M"
        runner = Runner(
            self.observables.dis_tp_like(self.pdf_name, restype),
            self.theory.dis_tp_like(hid),
        )
        runner.compute(n_cores)
        return runner.results

    def compute(self, hid, n_cores):
        # TODO: cache results somewhere
        mumerator_log = self.run_dis_tp(hid, n_cores)
        if self.use_yadism:
            denominator_log = self.run_yadism()
        else:
            self.theory.t_card["PTO"] -= 1
            denominator_log = self.run_dis_tp(hid, n_cores)
        self._results = self._log(mumerator_log, denominator_log, self.use_yadism)
        print(self._results)

    @staticmethod
    def _log(mumerator_log, denominator_log, use_yadism):
        for obs in denominator_log:
            my_obs = obs.split("_")[0]
            if use_yadism:
                den_df = pd.DataFrame(denominator_log[obs]).rename(
                    columns={"result": "yadism"}
                )
                num_df = mumerator_log[my_obs].rename(columns={"result": "dis_tp"})
            else:
                den_df = denominator_log[my_obs].rename(columns={"result": "NNLO"})
                num_df = mumerator_log[my_obs].rename(columns={"result": "N3LO"})
            log_df = pd.concat([den_df, num_df], axis=1).T.drop_duplicates().T

            # construct some nice log table
            log_df.drop("y", axis=1, inplace=True)
            if use_yadism:
                log_df.drop("q", axis=1, inplace=True)
                log_df.drop("error", axis=1, inplace=True)
                log_df["k-factor"] = log_df.dis_tp / log_df.yadism
            else:
                log_df["Q2"] = log_df.q**2
                log_df.drop("q", axis=1, inplace=True)
                log_df["k-factor"] = log_df.N3LO / log_df.NNLO

        return log_df

    def save_results(self, author, th_input):

        if self.use_yadism:
            k_fatctor_type = "N3LO FONLL DIS_TP / N3LO ZM-VFNS Yadism"
        else:
            k_fatctor_type = "N3LO FONLL DIS_TP / NNLO FONLL DIS_TP"
        intro = [
            "********************************************************************************\n",
            f"SetName: {self.dataset_name}\n",
            f"Author: {author}\n",
            f"Date: {date.today()}\n",
            "CodesUsed: https://github.com/andreab1997/DIS_TP\n",
            f"TheoryInput: {th_input}\n",
            f"PDFset: {self.pdf_name}\n",
            f"Warnings: {k_fatctor_type}\n"
            "********************************************************************************\n",
        ]
        res_path = self.result_path / f"CF_QCD_{self.dataset_name}.dat"
        print(f"Saving the k-factors in: {res_path}")
        with open(res_path, "w", encoding="utf-8") as f:
            f.writelines(intro)
            f.writelines([f"{k:4f}   0.0000\n" for k in self._results["k-factor"]])
