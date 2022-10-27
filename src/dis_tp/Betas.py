# This contains the beta functions to be used in future analytic expression. Not used at the moment.


import numpy as np


def beta_0():
    CA = 3.0
    TF = 1.0 / 2.0
    NF = 4.0
    return (11.0 / 3.0) * CA - (4.0 / 3.0) * TF * NF


def beta_1():
    CA = 3.0
    CF = 4.0 / 3.0
    TF = 1.0 / 2.0
    NF = 4.0
    return (34.0 / 3.0) * CA * CA - (20.0 / 3.0) * CA * TF * NF - 4 * CF * TF * NF
