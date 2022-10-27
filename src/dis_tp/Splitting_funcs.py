#This contains the splitting functions needed to compute some of the scale dependent term. Since at the moment 
#the code is using grids, these are not needed. 
from . import zetas as zet
from . import Harmonics as harm

import numpy as np
#splitting funcs expanded in alpha_s/4pi

def Pqq_0_plus(z):
    CF = 4./3.
    return 2*CF*((1+z**2)/(1-z))

def Pqg_0_reg(z):
    NF = 4.
    return 2*NF*((z**2)+((1-z)**2))

def Pgq_0_reg(z):
    CF = 4./3.
    return 2*CF*((1+((1-z)**2))/z)

def Pgg_0_reg(z):
    CA = 3.
    return 4*CA*(((1-z)/z)+z*(1-z)) 

def Pgg_0_local(z):
    CA = 3.
    NF = 4.
    return 4*CA*((11./12.)-(NF/(3*2*CA))) 

def Pgg_0_plus(z):
    CA = 3.
    return 4*CA*(z*(1./(1-z)))

def Pgg_1_reg(z):
    CF = 4./3.
    CA = 3.
    NF = 4.
    pgg_reg = (1./z) - 2 + z - z**2
    pgg_reg_mz = (-1./z) - 2 - z - z**2 + (1./(1+z))
    return 4*CA*NF*(1-z - (10./9.)*pgg_reg - (13./9.)*((1./z)-z**2)-(2./3.)*(1+z)*harm.H_0(z)) + 4*CA*CA*(27 + 
    (1+z)*((11./3.)*harm.H_0(z) + 8*harm.H_00(z)-(27./2.)) + 2*pgg_reg_mz*(harm.H_00(z)-2*harm.H_m10(z) - zet.zeta_2) - 
    (67./9.)*((1./z)-z**2) -12*harm.H_0(z)-(44./3.)*z*z*harm.H_0(z) + 2*pgg_reg*((67./18.)-zet.zeta_2+harm.H_00(z)+2*harm.H_10(z)+2*harm.H_01(z))
    ) +4*CF*NF*(2*harm.H_0(z)+(2./(3*z)) + (10./3.)*z*z -12 + (1+z)*(4-5*harm.H_0(z)-2*harm.H_00(z)))

def Pgg_1_local(z):
    CF = 4./3.
    CA = 3.
    NF = 4.
    return -4*CA*NF*(2./3.) + 4*CA*CA*((8./3.)+3*zet.zeta_3) - 4*CF*NF*(1./2.)

def Pgg_1_plus(z):
    CF = 4./3.
    CA = 3.
    NF = 4.
    pgg_plus = (1./(1-z)) 
    return 4*CA*NF*(-(10./9.)*pgg_plus) +4*CA*CA*(2*pgg_plus*((67./18.)-zet.zeta_2+harm.H_00(z)+2*harm.H_10(z)+2*harm.H_01(z)))
