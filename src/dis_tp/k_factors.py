"""Script to produce the k-factors."""

from datetime import date
import lhapdf
import pandas as pd
import yadism
import yaml

from . import configs
from .io import OperatorParameters, TheoryParameters
from .runner import Runner


class KfactorRunner:
    def __init__(self, t_card_name, dataset_name, pdf_name, h_id, use_yadism):
        cfg = configs.load()
        cfg = configs.defaults(cfg)

        # Load the ymldb file
        with open(
            cfg["paths"]["ymldb"] / f"{dataset_name}.yaml", encoding="utf-8"
        ) as f:
            ymldb = yaml.safe_load(f)

        self.theory = TheoryParameters.load_card(cfg, t_card_name, h_id)

        # TODO: do we need to support operations between FKtables?
        o_card_name = ymldb["operands"][0][0]
        self.observables = OperatorParameters.load_card(cfg, o_card_name, pdf_name)

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

    def run_dis_tp(self, n_cores):
        # TODO: how do we treat bottom mass effects in this code?
        runner = Runner(
            self.observables,
            self.theory,
        )
        runner.compute(n_cores)
        return runner.results

    def compute(self, n_cores):
        # TODO: cache results somewhere
        mumerator_log = self.run_dis_tp(n_cores)
        if self.use_yadism:
            denominator_log = self.run_yadism()
        else:
            self.theory.order -= 1
            denominator_log = self.run_dis_tp(n_cores)
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
