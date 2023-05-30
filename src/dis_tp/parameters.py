import numpy as np

from eko.matchings import Atlas, nf_default
from eko.quantities import heavy_quarks
import yadism.coefficient_functions.coupling_constants as coupl
from yadism.coefficient_functions.light import n3lo

pids = {"g": 21, "c": 4, "b": 5, "t": 6}


def number_active_flavors(Q):
    return nf_default(Q**2, _thr_atlas)


def number_light_flavors(Q):
    """This should match the FONLL prescription."""
    nf = nf_default(Q**2, _thr_atlas)
    if nf > 3:
        return nf - 1
    return nf


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


def initialize_theory(
    use_grids, masses=None, strong_coupling=None, thr_atlas=None, thr_atlas_as=None
):
    if not use_grids and masses is None:
        raise ValueError(
            f"Need to specify heavy particle masses when grids are not used."
        )
    if use_grids and masses is not None:
        for i, mass in enumerate(masses):
            if not np.isclose(mass, default_masses(i + 4)):
                raise ValueError(
                    f"Grids are only available for the default mass {default_masses(i+4)}."
                )
    global grids
    global _masses
    grids = use_grids
    if masses is None:
        _masses = [default_masses(4), default_masses(5), default_masses(6)]
    else:
        _masses = masses

    if strong_coupling is not None:
        global _alpha_s
        _alpha_s = strong_coupling.a_s

    global _thr_atlas
    global _thr_atlas_as

    # enforce some defaults: Thr Atlas for Q scale
    if thr_atlas is None:
        thresholds_ratios = np.array([1, 1, 1])
        _thr_atlas = Atlas(
            matching_scales=heavy_quarks.MatchingScales(_masses * thresholds_ratios),
            origin=(1.65**2, 4),
        )
    else:
        _thr_atlas = thr_atlas

    # enforce some defaults: Thr Atlas for alphas
    if thr_atlas_as is None:
        thresholds_ratios = np.array([1, 1, 1])
        _thr_atlas_as = Atlas(
            matching_scales=heavy_quarks.MatchingScales(_masses * thresholds_ratios),
            origin=(91.2**2, 5),
        )
    else:
        _thr_atlas_as = thr_atlas_as


def masses(h_id):
    return _masses[h_id - 4]


def alpha_s(mur2):
    return _alpha_s(mur2, nf_default(mur2, _thr_atlas_as))


# some default values for EM (therory is ignored)
_th_d = dict(
    SIN2TW=0.23126,
    MZ=91.1876,
    CKM="0.97428 0.22530 0.003470 0.22520 0.97345 0.041000 0.00862 0.04030 0.999152",
)
obs_d = dict(
    projectilePID=11,
    PolarizationDIS=0.0,
    prDIS="EM",
    PropagatorCorrection=0,
    NCPositivityCharge=None,
)
coupl_const = coupl.CouplingConstants.from_dict(_th_d, obs_d)


def n3lo_color_factors(partonic_channel, nf, skip_heavylight):
    """Compute N3LO color facotrs. nf is the number of total active flavors"""
    return n3lo.common.nc_color_factor(
        coupl_const, nf, partonic_channel, skip_heavylight
    )
