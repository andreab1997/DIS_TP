"""Script to produce the Yadism benchmark."""
import pathlib

import lhapdf
import numpy as np
import pandas as pd
import yadism
import yaml
from dis_tp.logging import df_to_table
from dis_tp.runner import Runner
from eko.interpolation import make_grid
from rich.console import Console
from yadmark.data import observables

console = Console()

here = pathlib.Path(__file__).absolute().parent


class TheoryCard:
    def __init__(self, pto, kcthr=1.0, kbthr=1.0):
        with open(
            here / "../project/theory_cards/400.yaml",
        ) as file:
            th = yaml.safe_load(file)

        th["TMC"] = 0
        th["IC"] = 0
        th["FactScaleVar"] = False
        th["RenScaleVar"] = False
        th["PTO"] = pto
        th["kcThr"] = kcthr
        th["kbThr"] = kbthr
        self.t_card = th

    def yadism_like(self):
        return self.t_card

    def dis_tp_like(self):
        new_t_card = self.t_card
        new_t_card["grids"] = True
        new_t_card["fns"] = "fonll"
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
            # {"x": float(x_fixed), "Q2": float(q2), "y": 0.0018429} for q2 in q2_grid
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
        new_o_card["TargetDIS"] = self.o_card["TargetDIS"]
        for fx, kins in self.o_card["observables"].items():
            new_kins = [
                {"x": point["x"], "q": np.sqrt(point["Q2"]), "y": point["y"]}
                for point in kins
            ]
            new_o_card["obs"][fx] = {
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
        dis_tp_log = self.run_dis_tp()
        yad_log = self.run_yadism()
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
    th_obj = TheoryCard(pto)
    obj = BenchmarkRunner(th_obj, obs_obj, pdf_name)
    obj.run()


def benchmarkFO_bottom(pto, pdf_name):
    obs_names = [f"F2_bottom", f"FL_bottom"]  # , f"XSHERANCAVG_{flavor}"]
    obs_obj = Observable_card(obs_names, q_min=1.5, q_max=6, q_fixed=4.5, restype="FO")
    th_obj = TheoryCard(pto)
    obj = BenchmarkRunner(th_obj, obs_obj, pdf_name)
    obj.run()

def benchmarkF_M_charm(pto, pdf_name):
    obs_names = ["XSHERANCAVG_charm"] #[f"F2_charm", f"FL_charm"]
    obs_obj = Observable_card(obs_names, q_min=1.5, q_max=5, q_fixed=3, restype="M")
    th_obj = TheoryCard(pto)
    obj = BenchmarkRunner(th_obj, obs_obj, pdf_name)
    obj.run()


def benchmarkFO_charm(pto, pdf_name):
    obs_names = [f"F2_charm", f"FL_charm"] # ["XSHERANCAVG_charm"]
    obs_obj = Observable_card(
        obs_names, q_min=1, q_max=1.6, q_fixed=1.4, restype="FO"
    )
    th_obj = TheoryCard(pto)
    obj = BenchmarkRunner(th_obj, obs_obj, pdf_name)
    obj.run()


def benchmarkFONLL(pto, pdf_name, heavyness):
    obs_names = [f"F2_{heavyness}", f"FL_{heavyness}"] # [f"XSHERANCAVG_{heavyness}"]
    q_fixed= 10 if heavyness=="charm" else 30
    x_fixed= 0.01 if heavyness=="charm" else 0.001
    obs_obj = Observable_card(
        obs_names, q_min=1, q_max=100, q_fixed=q_fixed, x_fixed=x_fixed, restype="FONLL"
    )
    th_obj = TheoryCard(pto)
    obj = BenchmarkRunner(th_obj, obs_obj, pdf_name)
    obj.run()

def benchmarkFONLL_kth(pto, pdf_name, heavyness):
    obs_names = [f"F2_{heavyness}", f"FL_{heavyness}"]
    if heavyness == "charm":
        x_fixed= 0.01
        q_fixed= 10
        kcThr = 2.0
        kbThr = 4.0 
    else:
        q_fixed= 30
        x_fixed= 0.001
        kbThr = 2.0
        kcThr = 1.0
    if heavyness in ["light", "total"]:    
        kcThr = 2.0 
    obs_obj = Observable_card(
        obs_names, q_min=1, q_max=100, q_fixed=q_fixed, x_fixed=x_fixed, restype="FONLL"
    )
    th_obj = TheoryCard(pto,kcThr,kbThr)
    obj = BenchmarkRunner(th_obj, obs_obj, pdf_name)
    obj.run()

def benchmarkFONLL_incomplete(pto, pdf_name, heavyness):
    obs_names = [f"FL_{heavyness}", f"F2_{heavyness}"] # [f"XSHERANCAVG_{heavyness}"]
    q_fixed= 10 if heavyness=="charm" else 30
    x_fixed= 0.01 if heavyness=="charm" else 0.001
    obs_obj = Observable_card(
        obs_names, q_min=1, q_max=100, q_fixed=q_fixed, x_fixed=x_fixed, restype="FONLL_incomplete"
    )
    th_obj = TheoryCard(pto,kbthr=4.0)
    obj = BenchmarkRunner(th_obj, obs_obj, pdf_name)
    obj.run()

if __name__ == "__main__":

    pdf_name = "NNPDF40_nnlo_pch_as_01180"
    # obj = benchmarkF_M_bottom(pto=2, pdf_name=pdf_name)
    # obj = benchmarkFO_bottom(pto=1, pdf_name=pdf_name)

    # obj = benchmarkF_M_charm(pto=1, pdf_name=pdf_name)
    # obj = benchmarkFO_charm(pto=1, pdf_name=pdf_name)

    # benchmarkFONLL(pto=2, pdf_name=pdf_name, heavyness="charm")
    # benchmarkFONLL(pto=2, pdf_name=pdf_name, heavyness="bottom")
    # benchmarkFONLL(pto=2, pdf_name=pdf_name, heavyness="light")
    # benchmarkFONLL(pto=2, pdf_name=pdf_name, heavyness="total")

    # benchmarkFONLL_kth(pto=2, pdf_name=pdf_name, heavyness="charm")
    # benchmarkFONLL_kth(pto=2, pdf_name=pdf_name, heavyness="bottom")
    # benchmarkFONLL_kth(pto=2, pdf_name=pdf_name, heavyness="light")
    # benchmarkFONLL_kth(pto=2, pdf_name=pdf_name, heavyness="total")

    benchmarkFONLL_incomplete(pto=3, pdf_name=pdf_name, heavyness="charm")
    benchmarkFONLL_incomplete(pto=3, pdf_name=pdf_name, heavyness="bottom")
    benchmarkFONLL_incomplete(pto=3, pdf_name=pdf_name, heavyness="total")
