import numpy as np
import ReadTxt as readt
import Initialize
def Mbg_1(z,p):
    TR = 1./2.
    return 2 * TR * np.log((p[1]**2)/(p[0]**2)) * (z*z + (1-z)*(1-z))

def Mbg_2(z,p):
    return Initialize.Mbg2(z,p[1])

def Mbg_3_reg(z,p):
    return 0

def Mgg_1_loc(z,p):
    TR = 1./2.
    return -(4./3.) * TR * np.log((p[1]**2)/(p[0]**2)) 

def Mgg_2_reg(z,p):
    CF = 4./3.
    TR = 1./2.
    L = np.log((p[1]**2)/(p[0]**2))
    z1 = 1-z 
    L1 = np.log(z1)
    return 0

def Mgg_2_loc(z,p):
    CF = 4./3.
    TR = 1./2.
    CA = 1. #change here
    L = np.log((p[1]**2)/(p[0]**2))
    #return (L**2)*((TR**2)*(16./9.))+L*(CF*TR*4 + CA*TR*(16./3.)) - CF*TR*15 + CA*TR*(10./9.)
    return 0
    
def Mgg_2_sing(z,p):
    CF = 4./3.
    TR = 1./2.
    L = np.log((p[1]**2)/(p[0]**2))
    z1 = 1-z 
    L1 = np.log(z1)
    return 0

def Mbq_2(z,p):
    return Initialize.Mbq2(z,p[1])

def Mbq_3_reg(z,p):
    return 0
#In the one above maybe there is a minus sign missing in front 
def Mgq_2_reg(z,p):
    CF = 4./3.
    TR = 1./2.
    L = np.log((p[1]**2)/(p[0]**2))
    z1 = 1-z 
    L1 = np.log(z1)
    return CF*TR*(((16./(3*z))- (16./3.) +z*(8./3.))*(L**2) + ((160./(9*z)) - (160./9.) +z*(128./9.) + L1*((32./(3*z)) - (32./3.) + z*(16./3.)))*L + (4./3.)*((2./z)-2+z)*(L1**2) + (8./9.)*((10./z)-10+8*z)*L1 + (1./27.)*((448./z)-448+344*z))



#alphas[4] to alphas[5] pieces---> alphas[5] = alphas[4](1+alphas[4]P(1)+(alphas[4]**2)P(2)+...)

def P1(p):
    return Mgg_1_loc(0,p)
#TODO: implement P2
def P2(p):
    return 0
