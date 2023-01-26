import pathlib

import click

from .. import configs, runner
from .base import command, root_path
from .grids import n_cores

t_card = click.argument("t_card", type=str)
o_card = click.argument("o_card", type=str)


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
