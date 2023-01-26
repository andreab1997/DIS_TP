"""Script to produce the Yadism benchmark."""
import pathlib

import lhapdf
import numpy as np
import pandas as pd
import yadism
from eko.interpolation import make_grid
from yadmark.data import observables
from dis_tp.runner import Runner

from df_to_table import df_to_table
from rich.console import Console

from banana.data.theories import default_card as theory_default

console = Console()

here = pathlib.Path(__file__).absolute().parent
pdf_name = "NNPDF40_nnlo_as_01180"
obs_names = ["F2_bottom", "FL_bottom"]  # "XSHERANCAVG_bottom"]
# obs_names = ["F2_charm", "XSHERANCAVG_charm"]


def load_theory():
    th = theory_default
    # make dis_tp comatible
    th["order"] = "NLO"
    th["fns"] = "fonll"
    th["hid"] = 5

    th["PTO"] = 1
    th["TMC"] = 0
    th["NfFF"] = 4
    th["FNS"] = "FONLL-A"
    return th


class Observable_card:
    def __init__(self, obs_names) -> None:

        x_grid = make_grid(30, 30, x_min=1e-6)
        q2_grid = np.geomspace(1**2, 30**2, 60)
        x_fixed = 0.01
        q2_fixed = 30**2

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
            [{"x": float(x), "Q2": float(q2_fixed), "y": 0.5} for x in x_grid[4:-3]]
        )
        for fx in obs_names:
            obs["observables"][fx] = kinematics
        self.o_card = obs

    def yadism_like(self):
        return self.o_card

    def dis_tp_like(self, pdf_name, restype="FO"):
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


def run_yadism(theory, observables, pdf):
    output = yadism.run_yadism(theory, observables)
    yad_pred = output.apply_pdf(pdf)
    return yad_pred


def run_dis_tp(theory, observables):
    runner = Runner(observables, theory)
    runner.compute(n_cores=4)
    return runner.results


def run_bench(obs_names):
    theory = load_theory()
    obs_obj = Observable_card(obs_names)

    yad_log = run_yadism(theory, obs_obj.yadism_like(), lhapdf.mkPDF(pdf_name))
    dis_tp_log = run_dis_tp(theory, obs_obj.dis_tp_like(pdf_name))

    for obs in obs_names:

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


if __name__ == "__main__":

    run_bench(obs_names)
