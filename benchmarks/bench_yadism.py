"""Script to produce the Yadism benchmark."""
import pathlib

import lhapdf
import numpy as np
import pandas as pd
import yadism
import yaml
from df_to_table import df_to_table
from eko.interpolation import make_grid
from rich.console import Console
from yadmark.data import observables

from dis_tp import parameters
from dis_tp.runner import Runner

console = Console()

here = pathlib.Path(__file__).absolute().parent


class TheoryCard:
    def __init__(self, pto, hid):
        with open(
            here / "../project/theory_cards/400.yaml",
        ) as file:
            th = yaml.safe_load(file)

        th["TMC"] = 0
        th["IC"] = 0
        th["PTO"] = pto

        th["NfFF"] = hid
        self.t_card = th

    def yadism_like(self):
        return self.t_card

    def dis_tp_like(self):
        new_t_card = {}
        new_t_card["grids"] = True
        new_t_card["hid"] = self.t_card["NfFF"]
        new_t_card["mass"] = parameters.default_masses(new_t_card["hid"])
        new_t_card["fns"] = "fonll"
        new_t_card["order"] = "N" * self.t_card["PTO"] + "LO"
        if self.t_card["PTO"] == 3:
            new_t_card["order"] =  "N3LO"
        return new_t_card


class Observable_card:
    def __init__(self, obs_names, q_min, q_max, restype, x_fixed=0.01, q_fixed=30):

        x_grid = make_grid(30, 30, x_min=1e-6)
        q2_grid = np.geomspace(q_min**2, q_max**2, 30)
        q2_fixed = q_fixed**2

        obs = observables.default_card
        obs["interpolation_xgrid"] = x_grid.tolist()
        obs["prDIS"] = "EM"
        obs["ProjectileDIS"] = "electron"
        obs["TargetDIS"] = "proton"
        obs["observables"] = {}
        kinematics = [
            {"x": float(x_fixed), "Q2": float(q2), "y": 0.5} for q2 in q2_grid
        ]
        kinematics.extend(
            [{"x": float(x), "Q2": float(q2_fixed), "y": 0.5} for x in x_grid[15:-15]]
        )
        for fx in obs_names:
            obs["observables"][fx] = kinematics
        self.o_card = obs
        self.restype = restype

    def yadism_like(self):
        return self.o_card

    def dis_tp_like(self, pdf_name):
        new_o_card = {}
        new_o_card["obs"] = {}
        for fx, kins in self.o_card["observables"].items():
            new_kins = [
                {"x": point["x"], "q": np.sqrt(point["Q2"]), "y": point["y"]}
                for point in kins
            ]
            new_o_card["obs"][fx.split("_")[0]] = {
                "PDF": pdf_name,
                "restype": self.restype,
                "scalevar": False,
                "kinematics": new_kins,
            }
        return new_o_card


class BenchmarkRunner:
    def __init__(self, theory, observables, pdf_name):
        self.theory = theory
        self.observables = observables
        self.pdf_name = pdf_name

    def run_yadism(self):
        output = yadism.run_yadism(
            self.theory.yadism_like(), self.observables.yadism_like()
        )
        yad_pred = output.apply_pdf(lhapdf.mkPDF(self.pdf_name))
        return yad_pred

    def run_dis_tp(self):
        runner = Runner(
            self.observables.dis_tp_like(self.pdf_name), self.theory.dis_tp_like()
        )
        runner.compute(n_cores=4)
        return runner.results

    def run(self):
        yad_log = self.run_yadism()
        dis_tp_log = self.run_dis_tp()
        self.log(dis_tp_log, yad_log)

    @staticmethod
    def log(dis_tp_log, yad_log):
        for obs in yad_log:
            my_obs = obs.split("_")[0]
            yad_df = pd.DataFrame(yad_log[obs]).rename(columns={"result": "yadism"})
            dis_tp_df = dis_tp_log[my_obs].rename(columns={"result": "dis_tp"})
            benc_df = pd.concat([yad_df, dis_tp_df], axis=1).T.drop_duplicates().T

            # construct some nice log table
            benc_df.drop("q", axis=1, inplace=True)
            benc_df.drop("y", axis=1, inplace=True)
            benc_df.drop("error", axis=1, inplace=True)
            benc_df["absolute error"] = np.abs(benc_df.dis_tp - benc_df.yadism)
            benc_df["percent error"] = (
                (benc_df.dis_tp - benc_df.yadism) / benc_df.yadism * 100
            )
            console.log(df_to_table(benc_df, obs))


def benchmarkF_M_bottom(pto, pdf_name):
    obs_names = [f"F2_bottom", f"FL_bottom"]  # , f"XSHERANCAVG_{flavor}"]
    obs_obj = Observable_card(obs_names, q_min=5, q_max=100, restype="M")
    th_obj = TheoryCard(pto, hid=5)
    obj = BenchmarkRunner(th_obj, obs_obj, pdf_name)
    obj.run()


def benchmarkFO_bottom(pto, pdf_name):
    obs_names = [f"F2_bottom", f"FL_bottom"]  # , f"XSHERANCAVG_{flavor}"]
    obs_obj = Observable_card(obs_names, q_min=1.5, q_max=5, q_fixed=4.5, restype="FO")
    th_obj = TheoryCard(pto, hid=5)
    obj = BenchmarkRunner(th_obj, obs_obj, pdf_name)
    obj.run()

def benchmarkF_M_charm(pto, pdf_name):
    obs_names = ["XSHERANCAVG_charm"] #[f"F2_charm", f"FL_charm"]
    obs_obj = Observable_card(obs_names, q_min=1.5, q_max=5, q_fixed=3, restype="M")
    th_obj = TheoryCard(pto, hid=4)
    obj = BenchmarkRunner(th_obj, obs_obj, pdf_name)
    obj.run()


def benchmarkFO_charm(pto, pdf_name):
    obs_names = ["XSHERANCAVG_charm"] # [f"F2_charm", f"FL_charm"]
    obs_obj = Observable_card(
        obs_names, q_min=1.2, q_max=1.5, q_fixed=1.4, restype="FO"
    )
    th_obj = TheoryCard(pto, hid=4)
    obj = BenchmarkRunner(th_obj, obs_obj, pdf_name)
    obj.run()

def benchmarkF_light(pto, pdf_name):
    obs_names = ["F2_light", "FL_light"]
    obs_obj = Observable_card(
        obs_names, q_min=1.5, q_max=5, q_fixed=3, restype="light"
    )
    th_obj = TheoryCard(pto, hid=4)
    obj = BenchmarkRunner(th_obj, obs_obj, pdf_name)
    obj.run()


if __name__ == "__main__":

    pdf_name = "NNPDF40_nnlo_pch_as_01180"
    # obj = benchmarkF_M_bottom(pto=2, pdf_name=pdf_name)
    # obj = benchmarkFO_bottom(pto=1, pdf_name=pdf_name)

    # obj = benchmarkF_M_charm(pto=2, pdf_name=pdf_name)
    # obj = benchmarkFO_charm(pto=2, pdf_name=pdf_name)
    benchmarkF_light(pto=3, pdf_name=pdf_name)
