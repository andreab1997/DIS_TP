"""Tilde coefficents functions for the matched scheme for the light components and asymptotics."""
import numpy as np

from .MasslessCoeffFunc import (
    Cb_1_loc,
    Cb_1_reg,
    Cb_1_sing,
    Cb_2_loc,
    Cb_2_reg,
    Cb_2_sing,
    
    Cg_2_reg,
    Cq_2_reg,

    CLb_1_reg,
    CLb_2_loc,
    CLb_2_reg,
    CLq_2_reg,
    CLg_2_reg,
)
from .MatchingFunc import (
    P1,
    Mqq_2_reg,
    Mqq_2_loc,
    Mqq_2_sing,
)


from yadism.coefficient_functions.fonll import raw_nc


############ NNLO Massive Non Singlet Asymptotics ########

# F2
def Cb_2_asy_reg(z, Q, p, _nf):
    e_q_light = p[-1] ** 2
    L = np.log((p[1] ** 2) / (p[0] ** 2))
    return e_q_light * (
        raw_nc.c2ns2am0_aq2(z) * L**2
        + raw_nc.c2ns2am0_aq(z) * L
        + raw_nc.c2ns2am0_a0(z)
    )

def Cb_2_asy_loc(z, Q, p, _nf):
    e_q_light = p[-1] ** 2
    L = np.log((p[1] ** 2) / (p[0] ** 2))
    return e_q_light * (
        raw_nc.c2ns2cm0_aq2(z) * L**2
        + raw_nc.c2ns2cm0_aq(z) * L
        + raw_nc.c2ns2cm0_a0(z)
    )

def Cb_2_asy_sing(z, Q, p, _nf):
    e_q_light = p[-1] ** 2
    L = np.log((p[1] ** 2) / (p[0] ** 2))
    return e_q_light * (
        raw_nc.c2ns2bm0_aq2(z) * L**2
        + raw_nc.c2ns2bm0_aq(z) * L
        + raw_nc.c2ns2bm0_a0(z)
    )

# FL
def CLb_2_asy_reg(z, Q, p, _nf):
    e_q_light = p[-1] ** 2
    L = np.log((p[1] ** 2) / (p[0] ** 2))
    return e_q_light * (
        + raw_nc.clns2am0_aq(z) * L
        + raw_nc.clns2am0_a0(z)
    )

############ NNLO and N3LO Tilde coefficients ########
# NOTE: at LO, NLO all these functions 
# should trivially reduce to the massless ones.

# TODO: construct N3LO tilde once massive 
#  coefficient functions to light will be available

# F2
def Cb_2_til_reg(z, Q, p, nl):
    e_q_light = p[-1] ** 2
    return Cb_2_reg(z, Q, p, nl) -(e_q_light * Mqq_2_reg(z, p, nl) + P1(p, nl) * Cb_1_reg(z, Q, p, nl))


def Cb_2_til_loc(z, Q, p, nl):
    e_q_light = p[-1] ** 2
    return Cb_2_loc(z, Q, p, nl) -(e_q_light * Mqq_2_loc(z, p, nl) + P1(p, nl) * Cb_1_loc(z, Q, p, nl))


def Cb_2_til_sing(z, Q, p, nl):
    e_q_light = p[-1] ** 2
    return Cb_2_sing(z, Q, p, nl) -(e_q_light * Mqq_2_sing(z, p, nl) + P1(p, nl) * Cb_1_sing(z, Q, p, nl))

def Cg_2_til_reg(z, Q, p, _nl):
    return Cg_2_reg(z, Q, p, _nl)


def Cq_2_til_reg(z, Q, p, _nl):
    return Cq_2_reg(z, Q, p, _nl)


# FL
def CLb_2_til_reg(z, Q, p, nl):
    return CLb_2_reg(z, Q, p, nl) - P1(p, nl) * CLb_1_reg(z, Q, p, nl)

def CLb_2_til_loc(z, Q, p, _nl):
    return CLb_2_loc(z, Q, p, _nl)


def CLg_2_til_reg(z, Q, p, _nl):
    return CLg_2_reg(z, Q, p, _nl)


def CLq_2_til_reg(z, Q, p, _nl):
    return CLq_2_reg(z, Q, p, _nl)
