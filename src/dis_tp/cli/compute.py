import click
import pathlib

from .. import io, configs
from .base import command, root_path
from .grids import n_cores

t_card = click.argument("t_card", type=str)
o_card = click.argument("o_card", type=str)

dest_path = click.option(
    "-d",
    "--dest_path",
    type=pathlib.Path,
    default=root_path / "results",
    required=False,
    help="result path",
)

@command.command("compute")
@t_card
@o_card
@n_cores
@dest_path
def generate_matching_grids(
    t_card: str, o_card: str, n_cores: int, dest_path: pathlib.Path
):
    """Run a computation."""

    cfg = configs.load()
    obs_obj = io.load_operator_parameters(cfg, o_card)
    th_obj = io.load_theory_parameters(cfg, t_card)
    par_obj = io.RunParameters(th_obj, obs_obj, dest_path)
    io.compute(par_obj, n_cores)