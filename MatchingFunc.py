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
    CA = 1
    L = np.log((p[1]**2)/(p[0]**2))
    z1 = 1-z 
    LO = np.log(z)
    L1 = np.log(z1)
    return (L**2)*(CF*TR*(8*(1+z)*LO + (16./(3*z)) + 4 - 4*z - (z**2)*(16./3.))+CA*TR*((8./(3*z))-(16./3.)+(8./3.)*z-(z**2)*(8./3.)))+L*(CF*TR*(8*(1+z)*(LO**2) + LO*(24+40*z) - (16./(3*z)) + 64 - 32*z - (80./3.)*(z**2))+CA*TR*((16./3.)*(1+z)*LO + (184./(9*z)) - (232./9.) + z*(152./9.) - (z**2)*(184./9.)))+CF*TR*((4./3.)*(1+z)*(LO**3) + (LO**2)*(6+10*z) + LO*(32+48*z)-(8./z) + 80 - 48*z - 24*(z**2))+CA*TR*((4./3.)*(1+z)*(LO**2) + (1./9.)*(52+88*z)*LO - (4./3.)*LO*z + (1./27.)*((556./z) - 628 + 548*z - 700*(z**2)))

def Mgg_2_loc(z,p):
    CF = 4./3.
    TR = 1./2.
    CA = 1. #change here
    L = np.log((p[1]**2)/(p[0]**2))
    return (L**2)*((TR**2)*(16./9.))+L*(CF*TR*4 + CA*TR*(16./3.)) - CF*TR*15 + CA*TR*(10./9.)
    
def Mgg_2_sing(z,p):
    CF = 4./3.
    TR = 1./2.
    CA = 1.
    L = np.log((p[1]**2)/(p[0]**2))
    z1 = 1-z 
    return (L**2)*(CA*TR*(8./3.)*(1./z1))+L*(CA*TR*(80./9.)*(1./z1))+CA*TR*((224./27.)*(1./z1))

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
    fact = np.log((p[1]**2)/(p[0]**2))
    return (2./9.)*(2*(fact**2)+ 33*fact - 7)
