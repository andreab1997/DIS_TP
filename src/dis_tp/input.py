from .configs import defaults, detect, load


class TheoryParameters:
    """Class containing all the theory parameters."""

    def __init__(self, order, hid):
        self.order = order
        self.hid = hid


class OperatorParameters:
    """Class containing all the operator parameters."""

    def __init__(self, x_grid, q_grid):
        self.x_grid = x_grid
        self.q_grid = q_grid


class InputParameters:
    """Class to hold all the theory and operator parameters."""

    def __init__(self, theoryparam, operatorparam, path):
        self.theoryparam = theoryparam
        self.operatorparam = operatorparam
        configs = defaults(load(detect(path)))
        self.configs = configs
