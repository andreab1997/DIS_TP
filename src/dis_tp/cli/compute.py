import pathlib

import click
import numpy as np
import yaml

from .. import configs, io, parameters, runner
from .base import command, root_path
from .grids import n_cores

t_card = click.argument("t_card", type=str)
o_card = click.argument("o_card", type=str)
obs = click.argument("obs", type=str)
restype = click.argument("restype", type=str)
pdf = click.argument("pdf", type=str)
scalevar = click.argument("scalevar", type=bool)


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
    th_obj = io.load_theory_parameters(cfg, t_card)
    ocard_path = cfg["paths"]["operator_cards"] / (o_card + ".yaml")
    old_ocard = {}
    if ocard_path.is_file():
        with open(
            cfg["paths"]["operator_cards"] / (o_card + ".yaml"), encoding="utf-8"
        ) as f:
            old_ocard = yaml.safe_load(f)
    kinematics = provide_default_kinematics(th_obj.hid)
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


def provide_default_kinematics(h_id):
    """Provide default kinematics for an observable."""
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
