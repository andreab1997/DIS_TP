import numpy as np

from . import parameters

pids = {"g": 21, "c": 4, "b": 5, "t": 6}


def number_active_flavors(h_id):
    return np.abs(h_id)


def number_light_flavors(h_id):
    return np.abs(h_id) - 1


def charges(h_id):
    ch = {
        4: 2.0 / 3.0,
        5: -1.0 / 3.0,
        6: 2.0 / 3.0,
    }
    return np.sign(h_id) * ch[h_id]


def default_masses(h_id):
    m = {4: 1.51, 5: 4.92, 6: 172.5}
    return m[h_id]


def initialize_theory(use_grids, h_id=None, mass=None):
    if not use_grids and mass is None:
        raise ValueError(
            f"Need to specify heavy particle mass when grids are not used."
        )
    if use_grids and h_id is None:
        raise ValueError(f"Need to specify heavy particle id in order to use grids.")
    if use_grids and mass is not None:
        if not np.isclose(mass, default_masses(h_id)):
            raise ValueError(
                f"Grids are only available for the default mass {default_masses(h_id)}."
            )
    global grids
    global _mass
    grids = use_grids
    if mass is None:
        _mass = default_masses(h_id)
    else:
        _mass = mass


def masses(h_id):
    return _mass
