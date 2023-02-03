import pathlib

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import parameters


class Plot:
    """Class for handling plots"""

    def __init__(self, configs: dict, plot_dir: pathlib.Path):
        self.result_path = configs["paths"]["results"]
        self.plot_dir = plot_dir

    def plot_single_obs(self, obs, order, h_id):
        parameters.initialize_theory(True, int(h_id), None)
        mass = parameters.masses(int(h_id))
        orderstrings = {"NLO": "nlo", "NNLO": "nnlo", "N3LO": "nnlo"}
        restypes = {
            "FO": {
                "color": "violet",
                "pdf": "NNPDF40_"
                + orderstrings[order]
                + "_as_01180_nf_"
                + str(int(h_id) - 1),
            },
            "M": {
                "color": "blue",
                "pdf": [
                    "NNPDF_" + h_id + "F_" + orderstrings[order] + "_mub=05mb",
                    "NNPDF_" + h_id + "F_" + orderstrings[order],
                    "NNPDF_" + h_id + "F_" + orderstrings[order] + "_mub=2mb",
                ],
            },
            "R": {
                "color": "green",
                "pdf": [
                    "NNPDF_" + h_id + "F_" + orderstrings[order] + "_mub=05mb",
                    "NNPDF_" + h_id + "F_" + orderstrings[order],
                    "NNPDF_" + h_id + "F_" + orderstrings[order] + "_mub=2mb",
                ],
            },
        }
        filename_FO = (
            obs + "_" + "FO" + "_" + order + "_" + h_id + "_" + restypes["FO"]["pdf"]
        )
        filenames_R = [
            obs + "_" + "R" + "_" + order + "_" + h_id + "_" + pdf
            for pdf in restypes["R"]["pdf"]
        ]
        filenames_M = [
            obs + "_" + "M" + "_" + order + "_" + h_id + "_" + pdf
            for pdf in restypes["M"]["pdf"]
        ]
        with open(self.result_path / (filename_FO + ".yaml"), encoding="utf-8") as f:
            result_FO = yaml.safe_load(f)
        results_R = {}
        results_M = {}
        for filepath in filenames_M:
            with open(self.result_path / (filepath + ".yaml"), encoding="utf-8") as f:
                string = filepath.split(orderstrings[order])[-1]
                results_M[string] = yaml.safe_load(f)
        for filepath in filenames_R:
            with open(self.result_path / (filepath + ".yaml"), encoding="utf-8") as f:
                string = filepath.split(orderstrings[order])[-1]
                results_R[string] = yaml.safe_load(f)
        # The x and qgrid should be the same across different restypes
        x_grid = result_FO["x_grid"]
        q_grid = result_FO["q_grid"]
        ordered_result_FO = []
        for x, q, res in zip(x_grid, q_grid, result_FO["obs"][0]):
            ordered_result_FO.append(dict(x=x, q=q, res=res))
        ordered_result_R = {}
        ordered_result_M = {}
        for result in results_R:
            ordered_result = []
            for x, q, res in zip(x_grid, q_grid, results_R[result]["obs"][0]):
                ordered_result.append(dict(x=x, q=q, res=res))
            ordered_result_R[result] = ordered_result
        for result in results_M:
            ordered_result = []
            for x, q, res in zip(x_grid, q_grid, results_M[result]["obs"][0]):
                ordered_result.append(dict(x=x, q=q, res=res))
            ordered_result_M[result] = ordered_result
        diff_x_points = list(set(x_grid))
        for x in diff_x_points:
            plot_name = obs + "_" + order + "_" + h_id
            plot_path = self.plot_dir / (plot_name + "_" + str(x) + ".pdf")
            q_plot = [res["q"] for res in ordered_result_FO if res["x"] == x]
            res_plot_FO = [res["res"] for res in ordered_result_FO if res["x"] == x]
            res_plot_R = {}
            res_plot_M = {}
            shifts = {
                "_mub=05mb": 0.5,
                "": 1.0,
                "_mub=2mb": 2.0,
            }
            for sv in ordered_result_R:
                res_plot_tmp = [res for res in ordered_result_R[sv] if res["x"] == x]
                res_plot = [
                    res["res"] if res["q"] > shifts[sv] * mass else np.nan
                    for res in res_plot_tmp
                ]
                res_plot_R[sv] = res_plot
            for sv in ordered_result_M:
                res_plot_tmp = [res for res in ordered_result_M[sv] if res["x"] == x]
                res_plot = [
                    res["res"] if res["q"] > shifts[sv] * mass else np.nan
                    for res in res_plot_tmp
                ]
                res_plot_M[sv] = res_plot
            plt.xscale("log")
            plt.plot(
                q_plot,
                res_plot_FO,
                label="FO",
                color="violet",
                linestyle="--",
                linewidth=3.5,
            )
            plt.plot(
                q_plot,
                res_plot_R[list(shifts.keys())[1]],
                label="R",
                color="green",
                linewidth=0.8,
            )
            plt.plot(
                q_plot,
                res_plot_M[list(shifts.keys())[1]],
                label="M",
                color="blue",
                linewidth=2.2,
            )
            to_fill_R = [
                (
                    lambda q: abs(
                        res_plot_R[list(shifts.keys())[0]][q_plot.index(q)]
                        - res_plot_R[list(shifts.keys())[1]][q_plot.index(q)]
                    )
                    if abs(
                        res_plot_R[list(shifts.keys())[0]][q_plot.index(q)]
                        - res_plot_R[list(shifts.keys())[1]][q_plot.index(q)]
                    )
                    > abs(
                        res_plot_R[list(shifts.keys())[2]][q_plot.index(q)]
                        - res_plot_R[list(shifts.keys())[1]][q_plot.index(q)]
                    )
                    else abs(
                        res_plot_R[list(shifts.keys())[2]][q_plot.index(q)]
                        - res_plot_R[list(shifts.keys())[1]][q_plot.index(q)]
                    )
                )(q)
                for q in q_plot
            ]
            to_fill_M = [
                (
                    lambda q: abs(
                        res_plot_M[list(shifts.keys())[0]][q_plot.index(q)]
                        - res_plot_M[list(shifts.keys())[1]][q_plot.index(q)]
                    )
                    if abs(
                        res_plot_M[list(shifts.keys())[0]][q_plot.index(q)]
                        - res_plot_M[list(shifts.keys())[1]][q_plot.index(q)]
                    )
                    > abs(
                        res_plot_M[list(shifts.keys())[2]][q_plot.index(q)]
                        - res_plot_M[list(shifts.keys())[1]][q_plot.index(q)]
                    )
                    else abs(
                        res_plot_M[list(shifts.keys())[2]][q_plot.index(q)]
                        - res_plot_M[list(shifts.keys())[1]][q_plot.index(q)]
                    )
                )(q)
                for q in q_plot
            ]
            plt.fill_between(
                q_plot,
                np.array(res_plot_R[list(shifts.keys())[1]]) + np.array(to_fill_R),
                np.array(res_plot_R[list(shifts.keys())[1]]) - np.array(to_fill_R),
                color="green",
                alpha=0.25,
            )
            plt.fill_between(
                q_plot,
                np.array(res_plot_M[list(shifts.keys())[1]]) + np.array(to_fill_M),
                np.array(res_plot_M[list(shifts.keys())[1]]) - np.array(to_fill_M),
                color="blue",
                alpha=0.25,
            )
            plt.xlabel("Q[GeV]")
            plt.ylabel("x" + obs)
            plt.legend()
            plt.grid(alpha=0.75)
            plt.savefig(plot_path)
            plt.close()
