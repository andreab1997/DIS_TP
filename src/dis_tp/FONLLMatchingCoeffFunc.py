""""""
import numpy as np
from eko.constants import TR

from .MasslessCoeffFunc import Cb_1_reg, Cb_1_loc, Cb_1_sing, CLb_1_reg
from .MatchingFunc import (
    Mqq_2_reg,
    Mqq_2_loc,
    Mqq_2_sing,
    P1
)

# F2
def Mb_2_reg(z, Q, p, _nl):
    e_q_light = p[-1] ** 2
    return - (
        e_q_light**2 * Mqq_2_reg(z, p, _nl)
        + P1(p, _nl) * Cb_1_reg(z, Q, p, _nl)
    )


def Mb_2_loc(z, Q, p, _nl):
    e_q_light = p[-1] ** 2
    return - (
        e_q_light * Mqq_2_loc(z, p, _nl) 
        + P1(p, _nl) * Cb_1_loc(z, Q, p, _nl)
    )


def Mb_2_sing(z, Q, p, _nl):
    e_q_light = p[-1] ** 2
    return -(
        e_q_light * Mqq_2_sing(z, p, _nl)
        + P1(p, _nl) * Cb_1_sing(z, Q, p, _nl)
    )

# FL
def MLb_2_reg(z, Q, p, _nl):
    return - P1(p, _nl) * CLb_1_reg(z, Q, p, _nl)
