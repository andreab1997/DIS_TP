import numpy as np

from MassiveCoeffFunc import CLg_3_m_reg, CLq_3_m_reg, Cg_1_m_reg, CLg_1_m_reg, Cg_2_m_reg, Cg_3_m_reg, Cq_2_m_reg, CLg_2_m_reg, CLq_2_m_reg, Cq_3_m_reg
from MasslessCoeffFunc import CLb_2_loc, CLb_2_reg, Cb_0_loc, Cb_1_loc,Cb_1_reg,Cb_1_sing, CLb_1_reg, Cb_2_loc, Cb_2_reg, Cb_2_sing
from MatchingFunc import Mbg_1, Mbg_2, Mbg_3_reg, Mbq_2, Mbq_3_reg, Mgg_1_loc, Mgg_2_loc, Mgg_2_reg, Mgg_2_sing, Mgq_2_reg, P1, P2
import scipy.special as special
import parameters as para
from tools import Convolute, Convolute_matching, Convolute_plus_coeff, Convolute_plus_matching, Convolute_plus_matching_per_matching

def Cb1_Mbg1(z):
    TR = 1./2.
    CF = 4./3.
    e_b = para.parameters["e_b"]
    return  4 * CF * TR * pow(e_b,2) * (-(5./2.) + 2 * z * (3 - 4 * z) + (pow(np.pi,2)/6.) * (-1 + 2 * z - 4 * pow(z,2)) + pow(np.log(1-z),2) * (1 - 2 * z * (1-z)) - (1./2.) * np.log(1-z) * (7 + 4 * z * (3 * z -4) + (4 - 8 * z * (1-z)) * np.log(z)) + (1./2.) * np.log(z) * (-1 + 4 * z * (3 * z -2) + (1 - 2 * z + 4 * pow(z,2)) * np.log(z)) + (2 * z - 1) * special.spence(z) )
def CLb1_Mbg1(z):
    TR = 1./2.
    CF = 4./3.
    e_b = para.parameters["e_b"]
    return 8 * CF * TR * pow(e_b,2) * (1 + z - 2 * pow(z,2) + 2 * z * np.log(z)) 
#F2
def Cg_1_til_reg(z,Q,p):
    q = [p[0],Q] #in the place of Q then there will be the factorization scale
    return Cg_1_m_reg(z,Q,p) - 2 * Cb_0_loc(z,Q) * Mbg_1(z,q)
def Cg_2_til_reg(z,Q,p):
    q = [p[0],Q] #in the place of Q then there will be the factorization scale
    return Cg_2_m_reg(z,Q,p) - 2 * Cb_0_loc(z,Q) * (Mbg_2(z,q) - Mbg_1(z,q)*Mgg_1_loc(z,q))  -2 * np.log((Q**2)/(p[0]**2))*Cb1_Mbg1(z)
def Cg_3_til_reg(z,Q,p):
    q = [p[0],Q] #in the place of Q then there will be the factorization scale
    return Cg_3_m_reg(z,Q,p) +Cg_2_m_reg(z,Q,p)*Mgg_1_loc(z,q) + P2(q)*Cg_1_m_reg(z,Q,p) - (Cg_1_m_reg(z,Q,p)*Mgg_2_loc(z,q) + Convolute(Cg_1_m_reg,Mgg_2_reg,z,Q,p) + Convolute_plus_matching(Cg_1_m_reg,Mgg_2_sing,z,Q,p)) - 2*Cb_0_loc(z,Q)*(Mbg_3_reg(z,q) - Mgg_1_loc(z,q)*Mbg_2(z,q) + Mbg_1(z,q)*Mgg_1_loc(z,q)*Mgg_1_loc(z,q) - (Mbg_1(z,q)*Mgg_2_loc(z,q) + Convolute_matching(Mbg_1,Mgg_2_reg,z,Q,p) + Convolute_plus_matching_per_matching(Mgg_2_sing,Mbg_1,z,Q,p) )) -2*(Cb_1_loc(z,Q)*Mbg_2(z,q) + Convolute(Cb_1_reg,Mbg_2,z,Q,p) + Convolute_plus_coeff(Cb_1_sing,Mbg_2,z,Q,p) - Cb1_Mbg1(z)*Mgg_1_loc(z,q)) -2*(Mbg_1(z,q)*Cb_2_loc(z,Q) + Convolute(Cb_2_reg,Mbg_1,z,Q,p) + Convolute_plus_coeff(Cb_2_sing,Mbg_1,z,Q,p)) 

def Cq_2_til_reg(z,Q,p):
    q = [p[0],Q] #in the place of Q then there will be the factorization scale
    return Cq_2_m_reg(z,Q,p) - 2 * Cb_0_loc(z,Q) * Mbq_2(z,q)
def Cq_3_til_reg(z,Q,p):
    q = [p[0],Q] #in the place of Q then there will be the factorization scale
    return Cq_3_m_reg(z,Q,p) + 2*Cq_2_m_reg(z,Q,p)*Mgg_1_loc(z,q) - Convolute(Cg_1_m_reg,Mgq_2_reg,z,Q,p) - 2*(Cb_1_loc(z,Q)*Mbq_2(z,q) + Convolute(Cb_1_reg,Mbq_2,z,Q,p) + Convolute_plus_coeff(Cb_1_sing,Mbq_2,z,Q,p)) -2*(Cb_0_loc(z,Q)*Mbq_3_reg(z,q))
#FL
def CLg_1_til_reg(z,Q,p):
    q = [p[0],Q] #in the place of Q then there will be the factorization scale
    return CLg_1_m_reg(z,Q,p) 
def CLg_2_til_reg(z,Q,p):
    q = [p[0],Q] #in the place of Q then there will be the factorization scale
    return CLg_2_m_reg(z,Q,p) - 2 * np.log((Q**2)/(p[0]**2)) * CLb1_Mbg1(z)
def CLg_3_til_reg(z,Q,p):
    q = [p[0],Q] #in the place of Q then there will be the factorization scale
    return CLg_3_m_reg(z,Q,p) +CLg_2_m_reg(z,Q,p)*Mgg_1_loc(z,q)+ P2(q)*CLg_1_m_reg(z,Q,p) - (CLg_1_m_reg(z,Q,p)*Mgg_2_loc(z,q) + Convolute(CLg_1_m_reg,Mgg_2_reg,z,Q,p) + Convolute_plus_matching(CLg_1_m_reg,Mgg_2_sing,z,Q,p)) -2*(Convolute(CLb_1_reg,Mbg_2,z,Q,p) - CLb1_Mbg1(z)*Mgg_1_loc(z,q)) -2*(CLb_2_loc(z,Q)*Mbg_1(z,q) + Convolute(CLb_2_reg,Mbg_1,z,Q,p))

def CLq_2_til_reg(z,Q,p):
    q = [p[0],Q] #in the place of Q then there will be the factorization scale
    return CLq_2_m_reg(z,Q,p) 
def CLq_3_til_reg(z,Q,p):
    q = [p[0],Q] #in the place of Q then there will be the factorization scale
    return CLq_3_m_reg(z,Q,p) + 2*CLq_2_m_reg(z,Q,p)*Mgg_1_loc(z,q) - Convolute(CLg_1_m_reg,Mgq_2_reg,z,Q,p) - 2*Convolute(CLb_1_reg,Mbq_2,z,Q,p) 
