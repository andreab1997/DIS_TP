import numpy as np

from . import parameters

pids = {"g": 21, "c": 4, "b": 5, "t": 6}


def number_active_flavors(h_id):
    return np.abs(h_id)


def number_light_flavors(h_id):
    return np.abs(h_id) - 1


def charges(h_id):
    ch = {
        1: -1.0 / 3.0,
        2: 2.0 / 3.0,
        3: -1.0 / 3.0,
        4: 2.0 / 3.0,
        5: -1.0 / 3.0,
        6: 2.0 / 3.0,
    }
    return np.sign(h_id) * ch[h_id]


def default_masses(h_id):
    m = {4: 1.51, 5: 4.92, 6: 172.5}
    return m[h_id]


def initialize_theory(use_grids, masses=None):
    if not use_grids and masses is None:
        raise ValueError(
            f"Need to specify heavy particle masses when grids are not used."
        )
    if use_grids and masses is not None:
        for i, mass in enumerate(masses):
            if not np.isclose(mass, default_masses(i+4)):
                raise ValueError(
                    f"Grids are only available for the default mass {default_masses(i+4)}."
                )
    global grids
    global _masses
    grids = use_grids
    if masses is None:
        _masses = [
            default_masses(4), default_masses(5), default_masses(6)
        ]
    else:
        _masses = masses


def masses(h_id):

    return _masses[h_id-4]
