"""Script to produce the k-factors."""

from datetime import date
import lhapdf
import pandas as pd
import yadism
import yaml
import copy

from . import configs
from .io import OperatorParameters, TheoryParameters
from .runner import Runner
from .logging import df_to_table, console



class KfactorRunner:
    def __init__(self, t_card_name, dataset_name, pdf_name, use_yadism):
        cfg = configs.load()
        cfg = configs.defaults(cfg)

        # Load the ymldb file
        console.log(f"Computing dataset: {dataset_name}")
        with open(
            cfg["paths"]["ymldb"] / f"{dataset_name}.yaml", encoding="utf-8"
        ) as f:
            ymldb = yaml.safe_load(f)

        self.theory = TheoryParameters.load_card(cfg, t_card_name)

        o_card_names = ymldb["operands"]
        self.observables = []
        for ocard_list in o_card_names:
            for o_card_name in ocard_list:
                self.observables.append(
                    OperatorParameters.load_card(cfg, o_card_name, pdf_name)
                )

        self.pdf_name = pdf_name
        self.use_yadism = use_yadism
        self.result_path = cfg["paths"]["results"]
        self.dataset_name = ymldb["target_dataset"]
        self.operation = ymldb["operation"]
        # This is not needed for the time being
        # self.conversion_factor = ymldb["conversion_factor"]
        self._results = None

    def run_yadism(self):
        yad_pred = {}
        for observable in self.observables:
            output = yadism.run_yadism(
                self.theory.yadism_like(), observable.yadism_like()
            )
            yad_pred[observable.dataset_name] = output.apply_pdf(
                lhapdf.mkPDF(self.pdf_name)
            )
        return yad_pred

    def run_dis_tp(self, n_cores):
        distp_pred = {}
        for observable in self.observables:
            runner = Runner(
                observable,
                self.theory,
            )
            runner.compute(n_cores)
            distp_pred[observable.dataset_name] = copy.deepcopy(runner.results)
        return distp_pred

    def compute(self, n_cores):
        # TODO: cache results somewhere
        mumerator_log = self.run_dis_tp(n_cores)
        if self.use_yadism:
            denominator_log = self.run_yadism()
        else:
            self.theory.order -= 1
            denominator_log = self.run_dis_tp(n_cores)
        logs_df = self._log(mumerator_log, denominator_log, self.use_yadism)
        self._results = self.build_kfactor(logs_df)
        console.log(df_to_table(self._results, self.dataset_name))

    @staticmethod
    def _log(mumerator_log, denominator_log, use_yadism):
        logs_df = {}
        # loop on operands
        for (num_name, num), (den_name, den) in zip(
            mumerator_log.items(), denominator_log.items()
        ):

            if num_name != den_name:
                raise ValueError(
                    "Numerator dataset name do not coincide with denominator name."
                )

            # loop on SF
            for obs in den:
                my_obs = obs.split("_")[0]
                if use_yadism:
                    den_df = pd.DataFrame(den[obs]).rename(columns={"result": "yadism"})
                    num_df = num[my_obs].rename(columns={"result": "dis_tp"})
                else:
                    den_df = den[my_obs].rename(columns={"result": "NNLO"})
                    num_df = num[my_obs].rename(columns={"result": "N3LO"})
                log_df = pd.concat([den_df, num_df], axis=1).T.drop_duplicates().T

                # construct some nice log table
                log_df.drop("y", axis=1, inplace=True)
                if use_yadism:
                    log_df.drop("q", axis=1, inplace=True)
                    log_df.drop("error", axis=1, inplace=True)
                else:
                    log_df["Q2"] = log_df.q**2
                    log_df.drop("q", axis=1, inplace=True)
            logs_df[num_name] = log_df

        return logs_df

    def build_kfactor(self, logs_df):

        if self.operation == "null" or self.operation is None:
            for log_df in logs_df.values():
                if self.use_yadism:
                    log_df["k-factor"] = log_df.dis_tp / log_df.yadism
                else:
                    log_df["k-factor"] = log_df.N3LO / log_df.NNLO
            return log_df

        elif self.operation == "RATIO":
            data1, data2 = (*logs_df,)
            k_fact_log = logs_df[data1]
            if self.use_yadism:
                k_fact_log["dis_tp"] = logs_df[data1].dis_tp / logs_df[data2].dis_tp
                k_fact_log["yadism"] = logs_df[data1].yadism / logs_df[data2].yadism
                k_fact_log["k-factor"] = k_fact_log.dis_tp / k_fact_log.yadism
            else:
                k_fact_log["N3LO"] = logs_df[data1].N3LO / logs_df[data2].N3LO
                k_fact_log["NNLO"] = logs_df[data1].NNLO / logs_df[data2].NNLO
                k_fact_log["k-factor"] = k_fact_log.N3LO / k_fact_log.NNLO
            return k_fact_log

        else:
            raise ValueError(f"Operation {self.operation} no implemented")

    def save_results(self, author, th_input):

        if self.use_yadism:
            k_fatctor_type = "N3LO FONLL DIS_TP / N3LO ZM-VFNS Yadism"
        else:
            k_fatctor_type = "N3LO FONLL DIS_TP / NNLO FONLL DIS_TP"
        intro = [
            "********************************************************************************\n",
            f"SetName: {self.dataset_name}\n",
            f'Author: {author.replace("_", " ")}\n',
            f"Date: {date.today()}\n",
            "CodesUsed: https://github.com/andreab1997/DIS_TP\n",
            f"TheoryInput: {th_input}\n",
            f"PDFset: {self.pdf_name}\n",
            f"Warnings: {k_fatctor_type}\n"
            "********************************************************************************\n",
        ]
        res_path = self.result_path / f"CF_QCD_{self.dataset_name}.dat"
        console.log(f"[green]Saving the k-factors in: {res_path}")
        with open(res_path, "w", encoding="utf-8") as f:
            f.writelines(intro)
            f.writelines([f"{k:4f}   0.0000\n" for k in self._results["k-factor"]])
