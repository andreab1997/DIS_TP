import pathlib

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import yaml
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class Plot:
    """Class for handling plots"""

    def __init__(self, configs: dict, plot_dir: pathlib.Path):
        self.result_path = configs["paths"]["results"]
        self.plot_dir = plot_dir

    def plot_single_obs(self, obs, order, h_id):
        restypes = {"FO": "violet", "M": "blue", "R": "green"}
        filenames = [
            obs + "_" + restype + "_" + order + "_" + h_id for restype in restypes
        ]
        filepaths = [self.result_path / (filename + ".yaml") for filename in filenames]
        results = []
        for filepath in filepaths:
            with open(filepath, encoding="utf-8") as f:
                results.append(yaml.safe_load(f))
        # The x and qgrid should be the same across different restypes
        x_grid = results[0]["x_grid"]
        q_grid = results[0]["q_grid"]
        obs_ress = [result["obs"][0] for result in results]
        ordered_results = []
        diff_x_points = list(set(x_grid))
        for result in obs_ress:
            ordered_result = []
            for x, q, res in zip(x_grid, q_grid, result):
                ordered_result.append(dict(x=x, q=q, res=res))
            ordered_results.append(ordered_result)
        for x in diff_x_points:
            plot_name = obs + "_" + order + "_" + h_id
            plot_path = self.plot_dir / (plot_name + "_" + str(x) + ".pdf")
            q_plot = [res["q"] for res in ordered_results[0] if res["x"] == x]
            res_plots = [
                [res["res"] for res in ordered_result if res["x"] == x]
                for ordered_result in ordered_results
            ]
            for restype in restypes:
                plt.plot(
                    q_plot,
                    res_plots[list(restypes.keys()).index(restype)],
                    label=restype,
                    color=restypes[restype],
                )
                plt.xlabel("Q[GeV]")
                plt.ylabel("x" + obs)
                plt.legend()
            plt.savefig(plot_path)
            plt.close()
