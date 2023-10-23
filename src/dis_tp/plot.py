import itertools
import pathlib

import cycler
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import io, parameters
from .Initialize import PATH_TO_GLOBAL

orderstrings = {"1": "NLO", "2": "NNLO", "3": "N3LO"}
heavy_dict = {"4": "charm", "5": "bottom"}
mu_list = ["0.5", "1.0", "2.0"]
n3lo_cf_variations = ["-1", "1"]
shifts = {
    "05kb": 0.5,
    "0": 1.0,
    "20kb": 2.0,
}


class Plot:
    """Class for handling plots"""

    def __init__(self, configs: dict, plot_dir: pathlib.Path):
        plt.style.use(self.load_style(PATH_TO_GLOBAL + "/External/style.yaml"))
        self.result_path = configs["paths"]["results"]
        self.plot_dir = plot_dir

    def flatten(self, d):
        newd = {}
        for k, v in d.items():
            if isinstance(v, dict):
                for ik, iv in self.flatten(v).items():
                    newd[f"{k}.{ik}"] = iv
            else:
                newd[k] = v
        return newd

    def load_style(self, path):
        style = self.flatten(yaml.safe_load(pathlib.Path(path).read_text()))

        capstyle = "lines.solid_capstyle"
        prop_cycle = "axes.prop_cycle"

        if capstyle in style:
            style[capstyle] = mpl._enums.CapStyle(style[capstyle])
        pcd = {k: v for k, v in style.items() if prop_cycle in k}
        if len(pcd) > 0:
            length = max(len(l) for l in pcd.values())
            for k, v in pcd.items():
                del style[k]
                cyc = cycler.cycler(
                    k.split(".")[-1], itertools.islice(itertools.cycle(v), length)
                )
                if prop_cycle not in style:
                    style[prop_cycle] = cyc
                else:
                    style[prop_cycle] += cyc

        return style

    def get_restypes(self, order, h_id, restype):
        prefix = "NNPDF40ev"
        restypes = {
            "FO": {
                "color": "violet",
                "pdf": prefix
                + "_"
                + orderstrings[order]
                + "_"
                + str(int(h_id) - 1)
                + "F",
            },
            "M": {
                "color": "blue",
                "pdf": [
                    prefix + "_" + orderstrings[order] + "_" + h_id + "F" + "_05kb",
                    prefix + "_" + orderstrings[order] + "_" + h_id + "F",
                    prefix + "_" + orderstrings[order] + "_" + h_id + "F" + "_20kb",
                ],
            },
            "R": {
                "color": "green",
                "pdf": [
                    prefix + "_" + orderstrings[order] + "_" + h_id + "F" + "_05kb",
                    prefix + "_" + orderstrings[order] + "_" + h_id + "F",
                    prefix + "_" + orderstrings[order] + "_" + h_id + "F" + "_20kb",
                ],
            },
        }
        return restypes[restype]

    def x_q_grids(self, obs, order, h_id):
        _, x_grid, q_grid = self.get_FO_result(obs, order, h_id)
        return (x_grid, q_grid)

    def get_FO_result(self, obs, order, h_id):
        mu_list = ["0.5", "1.0", "2.0"]
        filenames_FO = [
            obs
            + "_"
            + "FO"
            + "_"
            + order
            + "_"
            + h_id
            + "_"
            + heavy_dict[h_id]
            + "_thr="
            + thr
            + "_"
            + self.get_restypes(order, h_id, "FO")["pdf"]
            + "_0"
            for thr in mu_list
        ]
        results_FO = {}
        for filepath, mu in zip(filenames_FO, mu_list):
            with open(self.result_path / (filepath + ".yaml"), encoding="utf-8") as f:
                results_FO[mu] = yaml.safe_load(f)
        x_grid = results_FO[mu_list[1]]["x_grid"]
        q_grid = results_FO[mu_list[1]]["q_grid"]
        ordered_results_FO = {}
        for mu in mu_list:
            tmp_list = []
            for x, q, res in zip(x_grid, q_grid, results_FO[mu]["obs"][0]):
                tmp_list.append(dict(x=x, q=q, res=res))
            ordered_results_FO[mu] = tmp_list
        return (ordered_results_FO, x_grid, q_grid)

    def get_FO_n3lo_var_results(self, obs, h_id):
        n3lo_cf_variations = ["-1", "1"]
        filenames_FO = [
            obs
            + "_"
            + "FO"
            + "_"
            + "3"
            + "_"
            + h_id
            + "_"
            + heavy_dict[h_id]
            + "_thr="
            + "1.0"
            + "_"
            + self.get_restypes("3", h_id, "FO")["pdf"]
            + "_"
            + var
            for var in n3lo_cf_variations
        ]
        results_FO = {}
        for filepath, var in zip(filenames_FO, n3lo_cf_variations):
            with open(self.result_path / (filepath + ".yaml"), encoding="utf-8") as f:
                results_FO[var] = yaml.safe_load(f)
        x_grid = results_FO[n3lo_cf_variations[0]]["x_grid"]
        q_grid = results_FO[n3lo_cf_variations[0]]["q_grid"]
        ordered_results_FO = {}
        for var in n3lo_cf_variations:
            tmp_list = []
            for x, q, res in zip(x_grid, q_grid, results_FO[var]["obs"][0]):
                tmp_list.append(dict(x=x, q=q, res=res))
            ordered_results_FO[var] = tmp_list
        return ordered_results_FO

    def get_R_results(self, obs, order, h_id):
        mu_list = ["0.5", "1.0", "2.0"]
        filenames_R = [
            obs
            + "_"
            + "R"
            + "_"
            + order
            + "_"
            + h_id
            + "_"
            + heavy_dict[h_id]
            + "_thr="
            + mu
            + "_"
            + pdf
            + "_0"
            for pdf, mu in zip(self.get_restypes(order, h_id, "R")["pdf"], mu_list)
        ]
        results_R = {}
        for filepath in filenames_R:
            with open(self.result_path / (filepath + ".yaml"), encoding="utf-8") as f:
                string = filepath.split("_5F_")[-1].split("_0")[0]
                results_R[string] = yaml.safe_load(f)
        x_grid, q_grid = self.x_q_grids(obs, order, h_id)
        ordered_result_R = {}
        for result in results_R:
            ordered_result = []
            for x, q, res in zip(x_grid, q_grid, results_R[result]["obs"][0]):
                ordered_result.append(dict(x=x, q=q, res=res))
            ordered_result_R[result] = ordered_result
        return ordered_result_R

    def get_M_results(self, obs, order, h_id):
        mu_list = ["0.5", "1.0", "2.0"]
        filenames_M = [
            obs
            + "_"
            + "M"
            + "_"
            + order
            + "_"
            + h_id
            + "_"
            + heavy_dict[h_id]
            + "_thr="
            + mu
            + "_"
            + pdf
            + "_0"
            for pdf, mu in zip(self.get_restypes(order, h_id, "M")["pdf"], mu_list)
        ]
        results_M = {}
        for filepath in filenames_M:
            with open(self.result_path / (filepath + ".yaml"), encoding="utf-8") as f:
                string = filepath.split("_5F_")[-1].split("_0")[0]
                results_M[string] = yaml.safe_load(f)
        x_grid, q_grid = self.x_q_grids(obs, order, h_id)
        ordered_result_M = {}
        for result in results_M:
            ordered_result = []
            for x, q, res in zip(x_grid, q_grid, results_M[result]["obs"][0]):
                ordered_result.append(dict(x=x, q=q, res=res))
            ordered_result_M[result] = ordered_result
        return ordered_result_M

    def get_M_n3lo_var_results(self, obs, h_id):
        n3lo_cf_variations = ["-1", "1"]
        filenames_M = [
            obs
            + "_"
            + "M"
            + "_"
            + "3"
            + "_"
            + h_id
            + "_"
            + heavy_dict[h_id]
            + "_thr="
            + "1.0"
            + "_"
            + pdf
            + "_"
            + var
            for pdf, var in zip(
                [
                    self.get_restypes("3", h_id, "M")["pdf"][1],
                    self.get_restypes("3", h_id, "M")["pdf"][1],
                ],
                n3lo_cf_variations,
            )
        ]
        results_M = {}
        for filepath, var in zip(filenames_M, n3lo_cf_variations):
            with open(self.result_path / (filepath + ".yaml"), encoding="utf-8") as f:
                results_M[var] = yaml.safe_load(f)
        x_grid, q_grid = self.x_q_grids(obs, "3", h_id)
        ordered_result_M = {}
        for var in results_M:
            ordered_result = []
            for x, q, res in zip(x_grid, q_grid, results_M[var]["obs"][0]):
                ordered_result.append(dict(x=x, q=q, res=res))
            ordered_result_M[var] = ordered_result
        return ordered_result_M

    def q_plot_x(self, ordered_results_FO, x):
        return [res["q"] for res in ordered_results_FO[mu_list[1]] if res["x"] == x]

    def get_FO_res_plot_x_and_sv(self, ordered_results_FO, x):
        res_plot_FO = [
            res["res"] for res in ordered_results_FO[mu_list[1]] if res["x"] == x
        ]
        res_plot_FO_2mu = [
            res["res"] for res in ordered_results_FO[mu_list[2]] if res["x"] == x
        ]
        res_plot_FO_05mu = [
            res["res"] for res in ordered_results_FO[mu_list[0]] if res["x"] == x
        ]
        return [res_plot_FO_05mu, res_plot_FO, res_plot_FO_2mu]

    def get_FO_res_plot_x_n3lo_var(self, n3lo_var_FO, x):
        return [
            [res["res"] for res in n3lo_var_FO[n3lo_cf_variations[0]] if res["x"] == x],
            [res["res"] for res in n3lo_var_FO[n3lo_cf_variations[1]] if res["x"] == x],
        ]

    def get_R_res_plot_x_sv(self, ordered_result, x, mass):
        res_plot_fin = {}
        for sv in ordered_result:
            res_plot_tmp = [res for res in ordered_result[sv] if res["x"] == x]
            res_plot = [
                res["res"] if res["q"] >= (shifts[sv] * mass) else np.nan
                for res in res_plot_tmp
            ]
            res_plot_fin[sv] = res_plot
        return res_plot_fin

    def get_M_res_plot_x_sv(self, ordered_result, x, mass, sv_FO_coll):
        res_plot_fin = {}
        for sv, fo_sv in zip(ordered_result, sv_FO_coll):
            res_plot_tmp = [res for res in ordered_result[sv] if res["x"] == x]
            res_plot = [
                res["res"] if res["q"] >= (shifts[sv] * mass) else res_FO
                for res, res_FO in zip(res_plot_tmp, fo_sv)
            ]
            res_plot_fin[sv] = res_plot
        return res_plot_fin

    def get_M_res_plot_x_var(self, n3lo_var_M, x, mass, res_plot_FO_n3lo_variations):
        res_plot_fin = {}
        for var, fo_var in zip(n3lo_var_M, res_plot_FO_n3lo_variations):
            res_plot_tmp = [res for res in n3lo_var_M[var] if res["x"] == x]
            res_plot = [
                res["res"] if res["q"] >= mass else res_FO
                for res, res_FO in zip(res_plot_tmp, fo_var)
            ]
            res_plot_fin[var] = res_plot
        return res_plot_fin

    def construct_sv_band(self, res_plot, q_plot):
        return [
            (
                lambda q: 0.5
                * np.sqrt(
                    (
                        (
                            res_plot[list(shifts.keys())[0]][q_plot.index(q)]
                            - res_plot[list(shifts.keys())[1]][q_plot.index(q)]
                        )
                        ** 2
                    )
                    + (
                        (
                            res_plot[list(shifts.keys())[2]][q_plot.index(q)]
                            - res_plot[list(shifts.keys())[1]][q_plot.index(q)]
                        )
                        ** 2
                    )
                )
            )(q)
            for q in q_plot
        ]

    def construct_M_var_band(self, res_plot_var, res_plot, q_plot):
        return [
            (
                lambda q: 0.5
                * np.sqrt(
                    (
                        (
                            res_plot_var["-1"][q_plot.index(q)]
                            - res_plot[list(shifts.keys())[1]][q_plot.index(q)]
                        )
                        ** 2
                    )
                    + (
                        (
                            res_plot_var["1"][q_plot.index(q)]
                            - res_plot[list(shifts.keys())[1]][q_plot.index(q)]
                        )
                        ** 2
                    )
                )
            )(q)
            for q in q_plot
        ]

    def construct_FO_var_band(self, res_plot_var, res_plot, q_plot):
        return [
            (
                lambda q: 0.5
                * np.sqrt(
                    (
                        (res_plot_var[0][q_plot.index(q)] - res_plot[q_plot.index(q)])
                        ** 2
                    )
                    + (
                        (res_plot_var[1][q_plot.index(q)] - res_plot[q_plot.index(q)])
                        ** 2
                    )
                )
            )(q)
            for q in q_plot
        ]

    def plot_single_obs(self, obs, order, h_id):
        parameters.initialize_theory(True)
        mass = parameters.masses(int(h_id))
        ordered_results_FO, x_grid, _q_grid = self.get_FO_result(obs, order, h_id)
        ordered_result_M = self.get_M_results(obs, order, h_id)
        ordered_result_R = self.get_R_results(obs, order, h_id)
        n3lo_var_FO = {}
        n3lo_var_M = {}
        if order == "3":
            n3lo_var_FO = self.get_FO_n3lo_var_results(obs, h_id)
            n3lo_var_M = self.get_M_n3lo_var_results(obs, h_id)
        diff_x_points = list(set(x_grid))
        for x in diff_x_points:
            plot_name = obs + "_" + order + "_" + h_id
            plot_path = self.plot_dir / (plot_name + "_" + str(x) + ".pdf")
            q_plot = self.q_plot_x(ordered_results_FO, x)
            res_plot_FO_n3lo_variations = []
            if order == "3":
                res_plot_FO_n3lo_variations = self.get_FO_res_plot_x_n3lo_var(
                    n3lo_var_FO, x
                )
            sv_FO_coll = self.get_FO_res_plot_x_and_sv(ordered_results_FO, x)
            res_plot_R = self.get_R_res_plot_x_sv(ordered_result_R, x, mass)
            res_plot_M = self.get_M_res_plot_x_sv(ordered_result_M, x, mass, sv_FO_coll)
            res_plot_M_var = self.get_M_res_plot_x_var(
                n3lo_var_M, x, mass, res_plot_FO_n3lo_variations
            )
            plt.xscale("log")
            plt.plot(
                q_plot,
                sv_FO_coll[1],
                label="FO",
                color="violet",
                linestyle="--",
                linewidth=2.0,
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
                linewidth=1.2,
            )
            to_fill_R = self.construct_sv_band(res_plot_R, q_plot)
            to_fill_M = self.construct_sv_band(res_plot_M, q_plot)
            to_fill_M_n3lo_var = []
            to_fill_FO_n3lo_var = []
            if order == "3":
                to_fill_M_n3lo_var = self.construct_M_var_band(
                    res_plot_M_var, res_plot_M, q_plot
                )
                to_fill_FO_n3lo_var = self.construct_FO_var_band(
                    res_plot_FO_n3lo_variations, sv_FO_coll[1], q_plot
                )
            plt.fill_between(
                q_plot,
                np.array(res_plot_R[list(shifts.keys())[1]]) + np.array(to_fill_R),
                np.array(res_plot_R[list(shifts.keys())[1]]) - np.array(to_fill_R),
                color="green",
                label="scale_unc",
                alpha=0.25,
            )
            plt.fill_between(
                q_plot,
                np.array(res_plot_M[list(shifts.keys())[1]]) + np.array(to_fill_M),
                np.array(res_plot_M[list(shifts.keys())[1]]) - np.array(to_fill_M),
                color="blue",
                label="scale_unc",
                alpha=0.25,
            )
            if order == "3":
                plt.fill_between(
                    q_plot,
                    np.array(res_plot_M[list(shifts.keys())[1]])
                    + np.array(to_fill_M_n3lo_var),
                    np.array(res_plot_M[list(shifts.keys())[1]])
                    - np.array(to_fill_M_n3lo_var),
                    color="lightseagreen",
                    label="cf_unc",
                    alpha=0.25,
                )
                plt.fill_between(
                    q_plot,
                    np.array(sv_FO_coll[1]) + np.array(to_fill_FO_n3lo_var),
                    np.array(sv_FO_coll[1]) - np.array(to_fill_FO_n3lo_var),
                    color="darkorchid",
                    label="cf_unc",
                    alpha=0.25,
                    linestyle="--",
                )
            plt.xlabel("Q[GeV]")
            plt.ylabel("x" + obs)
            plt.legend()
            plt.grid(alpha=0.75)
            plt.savefig(plot_path)
            plt.close()

    def plot_fonll_order_comparison(self, obs, _order, h_id):
        parameters.initialize_theory(True)
        mass = parameters.masses(int(h_id))
        ordered_result_FO_NLO, x_grid, _q_grid = self.get_FO_result(obs, "1", h_id)
        ordered_result_FO_NNLO, x_grid, _q_grid = self.get_FO_result(obs, "2", h_id)
        ordered_result_FO_N3LO, x_grid, _q_grid = self.get_FO_result(obs, "3", h_id)
        n3lo_var_FO = self.get_FO_n3lo_var_results(obs, h_id)
        ordered_result_M_NLO = self.get_M_results(obs, "1", h_id)
        ordered_result_M_NNLO = self.get_M_results(obs, "2", h_id)
        ordered_result_M_N3LO = self.get_M_results(obs, "3", h_id)
        n3lo_var_M = self.get_M_n3lo_var_results(obs, h_id)
        diff_x_points = list(set(x_grid))
        for x in diff_x_points:
            plot_name = obs + "_comporders_" + h_id
            plot_path = self.plot_dir / (plot_name + "_" + str(x) + ".pdf")
            q_plot = self.q_plot_x(ordered_result_FO_NNLO, x)
            res_plot_FO_n3lo_variations = self.get_FO_res_plot_x_n3lo_var(
                n3lo_var_FO, x
            )
            sv_FO_coll_NLO = self.get_FO_res_plot_x_and_sv(ordered_result_FO_NLO, x)
            sv_FO_coll_NNLO = self.get_FO_res_plot_x_and_sv(ordered_result_FO_NNLO, x)
            sv_FO_coll_N3LO = self.get_FO_res_plot_x_and_sv(ordered_result_FO_N3LO, x)
            res_plot_M_NLO = self.get_M_res_plot_x_sv(
                ordered_result_M_NLO, x, mass, sv_FO_coll_NLO
            )
            res_plot_M_NNLO = self.get_M_res_plot_x_sv(
                ordered_result_M_NNLO, x, mass, sv_FO_coll_NNLO
            )
            res_plot_M_N3LO = self.get_M_res_plot_x_sv(
                ordered_result_M_N3LO, x, mass, sv_FO_coll_N3LO
            )
            to_fill_M_NLO = self.construct_sv_band(res_plot_M_NLO, q_plot)
            to_fill_M_NNLO = self.construct_sv_band(res_plot_M_NNLO, q_plot)
            to_fill_M_N3LO = self.construct_sv_band(res_plot_M_N3LO, q_plot)
            res_plot_M_var = self.get_M_res_plot_x_var(
                n3lo_var_M, x, mass, res_plot_FO_n3lo_variations
            )
            to_fill_M_n3lo_var = self.construct_M_var_band(
                res_plot_M_var, res_plot_M_N3LO, q_plot
            )
            plt.plot(
                q_plot,
                res_plot_M_NLO[list(shifts.keys())[1]],
                label="NLO",
                color="green",
                linewidth=2.0,
            )
            plt.plot(
                q_plot,
                res_plot_M_NNLO[list(shifts.keys())[1]],
                label="NNLO",
                color="violet",
                linewidth=2.0,
            )
            plt.plot(
                q_plot,
                res_plot_M_N3LO[list(shifts.keys())[1]],
                label="N3LO",
                color="blue",
                linewidth=2.5,
            )
            plt.fill_between(
                q_plot,
                np.array(res_plot_M_NLO[list(shifts.keys())[1]])
                + np.array(to_fill_M_NLO),
                np.array(res_plot_M_NLO[list(shifts.keys())[1]])
                - np.array(to_fill_M_NLO),
                color="green",
                label="scale_unc",
                alpha=0.25,
            )
            plt.fill_between(
                q_plot,
                np.array(res_plot_M_NNLO[list(shifts.keys())[1]])
                + np.array(to_fill_M_NNLO),
                np.array(res_plot_M_NNLO[list(shifts.keys())[1]])
                - np.array(to_fill_M_NNLO),
                color="violet",
                label="scale_unc",
                alpha=0.25,
            )
            plt.fill_between(
                q_plot,
                np.array(res_plot_M_N3LO[list(shifts.keys())[1]])
                + np.array(to_fill_M_N3LO),
                np.array(res_plot_M_N3LO[list(shifts.keys())[1]])
                - np.array(to_fill_M_N3LO),
                color="blue",
                label="scale_unc",
                alpha=0.25,
            )
            plt.fill_between(
                q_plot,
                np.array(res_plot_M_N3LO[list(shifts.keys())[1]])
                + np.array(to_fill_M_n3lo_var),
                np.array(res_plot_M_N3LO[list(shifts.keys())[1]])
                - np.array(to_fill_M_n3lo_var),
                color="lightseagreen",
                label="cf_unc",
                alpha=0.25,
            )
            plt.xscale("log")
            plt.xlabel("Q[GeV]")
            plt.ylabel("x" + obs)
            plt.legend()
            plt.grid(alpha=0.75)
            plt.savefig(plot_path)
            plt.close()
