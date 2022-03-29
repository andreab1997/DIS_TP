import numpy as np
from mpmath import *
import zetas as zet
mp.dps = 15 
mp.pretty = True


def H_0(x):
    return np.log(x)
def H_1(x):
    return -np.log(1-x)
def H_01(x):
    return polylog(2,x)
def H_001(x):
    return polylog(3,x)
def H_011(x):
    return (1./2.)*(np.log(x))*(np.log(1-x)**2) + (np.log(1-x))*polylog(2,1-x)-polylog(3,1-x)+zet.zeta_3
def H_0001(x):
    return polylog(4,x)
def H_0011(x):
    return 0. #polylog(2,2,x) 
    #Nielsen Polylog?
def H_0111(x):
    return (np.pi**4)/(90.) - (1./6.)*(np.log(1 - x)**3)*np.log(x) - (1./2.)*(np.log(1 - x)**2)*polylog[2, 1 - x] + np.log(1 - x)*polylog[3, 1 - x] - polylog[4, 1 - x]
def H_00001(x):
    return polylog(5,x)
def H_00011(x):
    return 0. #polylog(3,2,x) 
    #Nielsen Polylog?
def H_00101(x):
    return 0.  
    #HPL does not know 
def H_00111(x):
    return 0. #polylog(2,3,x) 
    #Nielsen Polylog?
def H_01011(x):
    return 0. 
    #HPL does not know 
def H_01111(x):
    return (1./24.)*(np.log(1 - x)**4)*np.log(x) + (1./6.)*(np.log(1 - x)**3)*polylog(2, 1 - x) - (1./2.)*(np.log(1 - x)**2)*polylog(3, 1 - x) + np.log(1 - x)*polylog(4, 1 - x) - polylog(5, 1 - x) + zet.zeta_5
def H_0111(x):
    return (np.pi**4)/(90.) - (1./6.)*(np.log(1 - x)**3)*np.log(x) - (1./2.)*(np.log(1 - x)**2)*polylog(2, 1 - x) + np.log(1 - x)*polylog(3, 1 - x) - polylog(4, 1 - x)
def H_0m1(x):
    return -polylog(2,-x)
def H_m1(x):
    return np.log(1+x)
def H_00m1(x):
    return -polylog(3,-x)
def H_0m1m1(x):
    return S12(-x)
def H_0m11(x):
    return ((np.pi**2)/(12.) - (np.log(2)**2)/(2.))*np.log(1 - x) + (1./2.)*np.log(2)*(np.log(1 - x)**2) - (1./6.)*(np.log(1 - x)**3) - (1./2.)*(np.log(1 - x)**2)*np.log(x) - np.log(1 - x)*polylog(2, 1 - x) + np.log(1 - x)*polylog(2, -x) - np.log(1 - x)*polylog(2, x) - polylog(3, (1 - x)/2.) + polylog(3, 1 - x) - polylog(3, -x) + polylog(3, x) + polylog(3, -((2*x)/(1 - x))) - zet.zeta_3 + (1./24.)*(-2*(np.pi**2)*np.log(2) + 4*(np.log(2)**3) + 21*zet.zeta_3)
def H_01m1(x):
    return  ((np.pi**2)/(12.) - (np.log(2)**2)/(2.))*np.log(1 + x) + (1./2.)*np.log(2)*(np.log(1 + x)**2) + np.log(1 + x)*polylog(2, x) - polylog(3, x) - polylog(3, (x)/(1 + x)) + polylog(3, (2*x)/(1 + x)) - polylog(3, (1 + x)/2) + (1./24.)*(-2*(np.pi**2)*np.log(2) + 4*(np.log(2)**3) + 21*zet.zeta_3)
def H_0m1m1m1(x):
    return 0.
    #return -((np.pi**4)/90.) + (1./6.)*np.log(-x)*(np.log(1 + x)**3) + (1./2.)*(np.log(1 + x)**2)*polylog(2, 1 + x) - np.log(1 + x)*polylog(3, 1 + x) + polylog(4, 1 + x)
    #chiedere a Marco
def H_0m101(x):
    return 0. 
    #HPL does not know
def H_00m1m1(x):
    return 0. #polylog(2,2,-x) 
    #Nielsen Polylog?
def H_00m11(x):
    return 0. 
    #HPL does not know
def H_000m1(x):
    return -polylog(4,-x)
def H_001m1(x):
    return 0. 
    #HPL does not know
def H_0m10m1m1(x):
    return 0. 
    #HPL does not know
def H_00m1m1m1(x):
    return 0. 
    #HPL does not know
def H_00m10m1(x):
    return 0. 
    #HPL does not know
def H_00m101(x):
    return 0. 
    #HPL does not know
def H_000m1m1(x):
    return 0. 
    #HPL does not know
def H_000m11(x):
    return 0. 
    #HPL does not know
def H_0000m1(x):
    return -polylog(5,-x)
def H_0001m1(x):
    return 0. 
    #HPL does not know
def H_0010m1(x):
    return 0. 
    #HPL does not know

#Generalized Nielsen 
def S12(x):
  ZETA3=zet.zeta_3
  if x > 1: 
    return - polylog(3,(1.-x)) + ZETA3 + np.log(x-1.) * polylog(2,(1.-x)) + 0.5 * np.log(x) * ( np.log(x-1.)*np.log(x-1.) - np.pi*np.pi ) 
           
  elif x==1: 
    return ZETA3 
  
  elif (0. < x) and (x < 1): 
    return - polylog(3,(1.-x)) + ZETA3 + np.log(1.-x) * polylog(2,(1.-x)) + 0.5 * np.log(x) * np.log(1.-x)*np.log(1.-x)
           
  elif x==0:
    return 0. 
  
  elif x<0.:
    c = 1./(1.-x)
    return - polylog(3,c) + ZETA3 + np.log(c) * polylog(2,c) + 0.5 * np.log(c)*np.log(c) * np.log(1.-c) - (1./6.) * np.log(c)*np.log(c)*np.log(c) 
           
  else: 
    return 0. 
  

