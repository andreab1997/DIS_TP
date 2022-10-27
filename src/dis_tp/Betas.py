#This contains the beta functions to be used in future analytic expression. Not used at the moment.


import numpy as np

def beta_0():
    CA = 3.
    TF = 1./2.
    NF = 4.
    return (11./3.)*CA - (4./3.)*TF*NF

def beta_1():
    CA = 3.
    CF = 4./3.
    TF = 1./2.
    NF = 4.
    return (34./3.)*CA*CA - (20./3.)*CA*TF*NF - 4*CF*TF*NF