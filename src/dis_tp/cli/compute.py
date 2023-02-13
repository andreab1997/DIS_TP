import pathlib

import click
import numpy as np
import yaml

from .. import configs, io, parameters, plot, runner, k_factors
from .base import command
from .grids import n_cores


def provide_default_kinematics(obs):
    """Provide default kinematics for an observable."""
    h_id = 4 if "charm" in obs else 5
    mass = parameters.masses(h_id)
    Qlogmin = np.log10(1.0)
    Qlogmax = np.log10(150.0)
    Qlog = np.linspace(Qlogmin, Qlogmax, 200)
    Qcommon = pow(10, Qlog)
    eps = 0.5
    thre = [mass * ratio for ratio in [0.5, 1.0, 2.0]]
    Qsing = [np.linspace(thr - eps, thr + eps, 5) for thr in thre]
    Q = np.sort(np.concatenate((Qcommon, Qsing[0], Qsing[1], Qsing[2]))).tolist()
    X = [0.1, 0.01, 0.001, 0.0001]
    kinematics = []
    for x in X:
        for q in Q:
            dictx = {}
            dictx["x"] = x
            dictx["q"] = q
            kinematics.append(dictx)
    return kinematics


t_card = click.argument("t_card", type=str)
o_card = click.argument("o_card", type=str)
obs = click.argument("obs", type=str)
restype = click.argument("restype", type=str)
pdf = click.argument("pdf", type=str)
scalevar = click.argument("scalevar", type=bool)
plot_dir = click.argument("plot_dir", type=str)
order = click.argument("order", type=str)
h_id = click.argument("h_id", type=str)


@command.command("compute")
@o_card
@t_card
@n_cores
def generate_matching_grids(o_card: str, t_card: str, n_cores: int):
    """
    Run a computation.

    USAGE dis_tp compute <o_card> <t_card>
    """

    obj = runner.Runner(o_card, t_card)
    obj.compute(n_cores)
    obj.save_results()


@command.command("add_observable")
@o_card
@t_card
@obs
@restype
@pdf
@scalevar
def add_obs_opcard(
    o_card: str, t_card: str, obs: str, restype: str, pdf: str, scalevar: bool
):
    """
    Add an observable to the operator card with default kinematics.

    USAGE dis_tp add_observable <o_card> <obs> <restype> <pdf> <scalevar>
    """

    cfg = configs.load()
    cfg = configs.defaults(cfg)
    th_obj = io.TheoryParameters.load_card(cfg, t_card)
    ocard_path = cfg["paths"]["operator_cards"] / (o_card + ".yaml")
    old_ocard = {}
    if ocard_path.is_file():
        with open(
            cfg["paths"]["operator_cards"] / (o_card + ".yaml"), encoding="utf-8"
        ) as f:
            old_ocard = yaml.safe_load(f)
    kinematics = provide_default_kinematics(obs)
    to_update = {
        "obs": {
            obs: dict(
                PDF=pdf, restype=restype, scalevar=scalevar, kinematics=kinematics
            )
        }
    }
    old_ocard["obs"].update(to_update["obs"])
    with open(
        cfg["paths"]["operator_cards"] / (o_card + ".yaml"), "w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(old_ocard, f)


@command.command("plot")
@plot_dir
@obs
@order
@h_id
def plot_observable(plot_dir: str, obs: str, order: str, h_id: str):
    """
    Plot (and save the results in plot_dir) an observable with all the method overimposed for a certain order
    and heavy quark id.

    USAGE dis_tp plot_observable <plot_dir> <obs> <order> <h_id>
    """
    cfg = configs.load()
    cfg = configs.defaults(cfg)
    plot_dir_path = cfg["paths"]["root"] / plot_dir
    plotclass = plot.Plot(cfg, plot_dir_path)
    plotclass.plot_single_obs(obs, order, h_id)


@command.command("k-factors")
@o_card
@t_card
@pdf
@n_cores
@click.argument(
    "author",
    type=str,
)
@click.option(
    "-yad",
    "--use_yadism",
    is_flag=True,
    type=bool,
    default=False,
    required=False,
    help="If True compute the k-factor w.r.t. Yadism",
)
@click.option(
    "-th",
    "--th_description",
    type=str,
    default="NNPDF4.0 pch with alphas(MZ)=0.118",
    required=False,
    help="TheoryInput to be stored in the CF file",
)
def generate_kfactors(
    o_card: str,
    t_card: str,
    pdf: str,
    author: str,
    n_cores: int,
    use_yadism: bool,
    th_description: str,
):
    """Generate k-factors.

    USAGE: dis_tp k-factors HERACOMB_SIGMARED_C 400 NNPDF40_nnlo_pch_as_01180 "Your Name" [-n 4 -yad -th "Theory Input"]
    """

    obj = k_factors.KfactorRunner(t_card, o_card, pdf, use_yadism)
    obj.compute(n_cores)
    obj.save_results(author, th_input=th_description)
