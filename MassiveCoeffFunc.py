import numpy as np
import Initialize
import parameters as para
from tools import Convolute, Convolute_matching, Convolute_plus_coeff, Convolute_plus_matching
import Splitting_funcs as Sp
import Betas as bet
#F2
def Cg_1_m_reg(z,Q,p):
    Q2 = Q*Q
    m_b = p[0]
    eps = m_b*m_b/Q2
    thre = 4.*eps*z/(1-z)
    v = np.sqrt(1-thre)
    TR = 1./2.
    e_b = para.parameters["e_b"]
    if thre > 1.:
        return 0
    return  4 * TR * e_b*e_b * (v * ( 8 * z * (1-z) - 1 - 4 * z * (1-z) * eps) + np.log( (1+v) / (1-v) ) * ( z*z + (1-z)**2 + 4 * z * eps * (1-3*z) - 8 * z*z * eps*eps)) 
        
def Cg_2_m_reg(z,Q,p):
    Q2 = Q*Q
    m_b = p[0]
    eps = m_b*m_b/Q2
    thre = 4.*eps*z/(1-z)
    if thre > 1.:
        return 0
    return Initialize.Cg2m(z,Q)[0]

def Cg_3_m_reg(z,Q,p):
    Q2 = Q*Q
    m_b = p[0]
    eps = m_b*m_b/Q2
    thre = 4.*eps*z/(1-z)
    if thre > 1.:
        return 0
    #return Initialize.Cg3m(z,Q)[0]
    return 0.

def Cq_2_m_reg(z,Q,p):
    Q2 = Q*Q
    m_b = p[0]
    eps = m_b*m_b/Q2
    thre = 4.*eps*z/(1-z)
    if thre > 1.:
        return 0
    return Initialize.Cq2m(z,Q)[0]

def Cq_3_m_reg(z,Q,p):
    Q2 = Q*Q
    m_b = p[0]
    eps = m_b*m_b/Q2
    thre = 4.*eps*z/(1-z)
    if thre > 1.:
        return 0
    #return Initialize.Cq3m(z,Q)[0]
    return 0
#FL
def CLg_1_m_reg(z,Q,p):
    Q2 = Q*Q
    m_b = p[0]
    eps = m_b*m_b/Q2
    z2 = z*z
    thre = 4.*eps*z/(1-z)
    v = np.sqrt(1-thre)
    TR = 1./2.
    e_b = para.parameters["e_b"]
    if thre > 1.:
        return 0
    return  4 * TR * e_b*e_b *( - 8 * eps * z2 * np.log( ( 1 + v ) / ( 1 - v ) ) + 4 * v * z * ( 1 - z ) )
def CLg_2_m_reg(z,Q,p):
    Q2 = Q*Q
    m_b = p[0]
    eps = m_b*m_b/Q2
    thre = 4.*eps*z/(1-z)
    if thre > 1.:
        return 0
    return Initialize.CLg2m(z,Q)[0]
def CLg_3_m_reg(z,Q,p):
    Q2 = Q*Q
    m_b = p[0]
    eps = m_b*m_b/Q2
    thre = 4.*eps*z/(1-z)
    if thre > 1.:
        return 0
    #return Initialize.CLg3m(z,Q)[0]
    return 0

def CLq_2_m_reg(z,Q,p):
    Q2 = Q*Q
    m_b = p[0]
    eps = m_b*m_b/Q2
    thre = 4.*eps*z/(1-z)
    if thre > 1.:
        return 0
    return Initialize.CLq2m(z,Q)[0]
def CLq_3_m_reg(z,Q,p):
    Q2 = Q*Q
    m_b = p[0]
    eps = m_b*m_b/Q2
    thre = 4.*eps*z/(1-z)
    if thre > 1.:
        return 0
    #return Initialize.CLq3m(z,Q)[0]
    return 0

