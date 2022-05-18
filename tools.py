#tool functions to compute convolutions between PDFs and coefficients functions but also between matching functions and 
# coefficient functions and between matching functions alone

import scipy.integrate as integrate
import numpy as np

def PDFConvolute(func1, pdf, x, Q, pid, p1=None):
    np.seterr(invalid='ignore')
    lower = x 
    upper = 1.
    if pid == 21:
        result, error = integrate.quad(lambda z: func1(z,Q,p1)*pdf.xfxQ2(pid, x*(1./z),Q*Q),lower,upper, epsrel = 1.e-02,points=(x,1.))
    if pid == 5:
        result, error = integrate.quad(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(pid, x*(1./z),Q*Q) + pdf.xfxQ2(-pid, x*(1./z),Q*Q)),lower,upper, epsrel = 1.e-02,points=(x,1.)) 
    if pid == 1:
        result, error = integrate.quad(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(1, x*(1./z),Q*Q) + pdf.xfxQ2(2, x*(1./z),Q*Q) + pdf.xfxQ2(3, x*(1./z),Q*Q) + pdf.xfxQ2(4, x*(1./z),Q*Q) + pdf.xfxQ2(-1, x*(1./z),Q*Q) + pdf.xfxQ2(-2, x*(1./z),Q*Q) + pdf.xfxQ2(-3, x*(1./z),Q*Q) + pdf.xfxQ2(-4, x*(1./z),Q*Q)),lower,upper, epsrel = 1.e-02,points=(x,1.)) 
    return result

def Convolute(func1,matching, x,Q,p1=None):
    np.seterr(invalid='ignore')
    lower = x 
    upper = 1.
    q = [p1[0],Q]
    result, error = integrate.quad(lambda z: (1./z)*func1(z,Q,p1)*matching(x*(1./z),q),lower,upper, epsrel = 1.e-02,points=(x,1.))
    return result

def Convolute_matching(matching1,matching2, x,Q,p1=None):
    np.seterr(invalid='ignore')
    lower = x 
    upper = 1.
    q = [p1[0],Q]
    result, error = integrate.quad(lambda z: (1./z)*matching1(z,q)*matching2(x*(1./z),q),lower,upper, epsrel = 1.e-02,points=(x,1.))
    return result

def PDFConvolute_plus(func1,pdf,x,Q,pid,p1=None):
    if pid == 21:
        plus1, error1 = integrate.quad(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(pid, x*(1./z),Q*Q) - pdf.xfxQ2(pid, x,Q*Q)),x,1., epsrel = 1.e-02, points=(x,1.))
        plus2, error2 = integrate.quad(lambda z: func1(z,Q,p1)*pdf.xfxQ2(pid,x,Q*Q),0.,x, epsrel = 1.e-02, points=(0.,x))
    if pid == 5:
        plus1, error1 = integrate.quad(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(pid, x*(1./z),Q*Q) + pdf.xfxQ2(-pid, x*(1./z),Q*Q) - pdf.xfxQ2(pid,x,Q*Q) - pdf.xfxQ2(-pid,x,Q*Q) ),x,1., epsrel = 1.e-02, points=(x,1.))
        plus2, error2 = integrate.quad(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(pid,x,Q*Q) + pdf.xfxQ2(-pid,x,Q*Q)),0.,x, epsrel = 1.e-02, points=(0.,x))
    if pid == 1:
        plus1, error1 = integrate.quad(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(1, x*(1./z),Q*Q) + pdf.xfxQ2(2, x*(1./z),Q*Q) + pdf.xfxQ2(3, x*(1./z),Q*Q) + pdf.xfxQ2(4, x*(1./z),Q*Q) + pdf.xfxQ2(-1, x*(1./z),Q*Q) + pdf.xfxQ2(-2, x*(1./z),Q*Q) + pdf.xfxQ2(-3, x*(1./z),Q*Q) + pdf.xfxQ2(-4, x*(1./z),Q*Q)-(pdf.xfxQ2(1, x,Q*Q) + pdf.xfxQ2(2, x,Q*Q) + pdf.xfxQ2(3, x,Q*Q) + pdf.xfxQ2(4, x,Q*Q) + pdf.xfxQ2(-1, x,Q*Q) + pdf.xfxQ2(-2, x,Q*Q) + pdf.xfxQ2(-3, x,Q*Q) + pdf.xfxQ2(-4, x,Q*Q))),x,1., epsrel = 1.e-02, points=(x,1.))
        plus2, error2 = integrate.quad(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(1, x,Q*Q) + pdf.xfxQ2(2, x,Q*Q) + pdf.xfxQ2(3, x,Q*Q) + pdf.xfxQ2(4, x,Q*Q) + pdf.xfxQ2(-1, x,Q*Q) + pdf.xfxQ2(-2, x,Q*Q) + pdf.xfxQ2(-3, x,Q*Q) + pdf.xfxQ2(-4, x,Q*Q)),0.,x, epsrel = 1.e-02, points=(0.,x))
    return plus1 - plus2

def Convolute_plus_coeff(func1,matching, x,Q,p1=None):
    np.seterr(invalid='ignore')
    q = [p1[0],Q]
    plus1, error1 = integrate.quad(lambda z: func1(z,Q,p1)*((1./z)*matching(x*(1./z),q) - matching(x,q)),x,1., epsrel = 1.e-02, points=(x,1.))
    plus2, error2 = integrate.quad(lambda z: func1(z,Q,p1)*matching(x,q),0.,x, epsrel = 1.e-02, points=(0.,x))
    return plus1-plus2

def Convolute_plus_matching(func1,matching, x,Q,p1=None):
    np.seterr(invalid='ignore')
    q = [p1[0],Q]
    plus1, error1 = integrate.quad(lambda z: matching(z,q)*((1./z)*func1(x*(1./z),Q,p1) - func1(x,Q,p1)),x,1., epsrel = 1.e-02, points=(x,1.))
    plus2, error2 = integrate.quad(lambda z: matching(z,q)*func1(x,Q,p1),0.,x, epsrel = 1.e-02, points=(0.,x))
    return plus1-plus2

def Convolute_plus_matching_per_matching(matchingplus,matching2, x,Q,p1=None):
    np.seterr(invalid='ignore')
    q = [p1[0],Q]
    plus1, error1 = integrate.quad(lambda z: matchingplus(z,q)*((1./z)*matching2(x*(1./z),q) - matching2(x,q)),x,1., epsrel = 1.e-02, points=(x,1.))
    plus2, error2 = integrate.quad(lambda z: matchingplus(z,q)*matching2(x,q),0.,x, epsrel = 1.e-02, points=(0.,x))
    return plus1-plus2