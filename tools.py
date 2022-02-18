import scipy.integrate as integrate
import numpy as np

def PDFConvolute(func1, pdf, x, Q, pid, p1=None):
    np.seterr(invalid='ignore')
    result = 0.
    error = 0.
    lower = x 
    upper = 1.
    if pid == 21:
        resultmid, errormid = integrate.quad(lambda z: func1(z,Q,p1)*pdf.xfxQ2(pid, x*(1./z),Q*Q),lower,upper, limit=500, epsrel = 1.e-02,points=(x,1.))
        #resultmid = integrate.romberg(lambda z: func1(z,Q,p1)*pdf.xfxQ2(pid, x*(1./z),Q*Q),lower,upper,rtol = 1e-3)
        result = resultmid
        errormid = 0 
        error = errormid
    if pid == 5:
        resultmid, errormid = integrate.quad(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(pid, x*(1./z),Q*Q) + pdf.xfxQ2(-pid, x*(1./z),Q*Q)),lower,upper, limit=500, epsrel = 1.e-02,points=(x,1.))
        #resultmid = integrate.romberg(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(pid, x*(1./z),Q*Q) + pdf.xfxQ2(-pid, x*(1./z),Q*Q)),lower,upper,rtol = 1e-3)
        result = resultmid 
        errormid = 0 
        error = errormid
    if pid == 1:
        resultmid, errormid = integrate.quad(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(1, x*(1./z),Q*Q) + pdf.xfxQ2(2, x*(1./z),Q*Q) + pdf.xfxQ2(3, x*(1./z),Q*Q) + pdf.xfxQ2(4, x*(1./z),Q*Q) + pdf.xfxQ2(-1, x*(1./z),Q*Q) + pdf.xfxQ2(-2, x*(1./z),Q*Q) + pdf.xfxQ2(-3, x*(1./z),Q*Q) + pdf.xfxQ2(-4, x*(1./z),Q*Q)),lower,upper, limit=500, epsrel = 1.e-02,points=(x,1.))
        #resultmid= integrate.romberg(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(1, x*(1./z),Q*Q) + pdf.xfxQ2(2, x*(1./z),Q*Q) + pdf.xfxQ2(3, x*(1./z),Q*Q) + pdf.xfxQ2(4, x*(1./z),Q*Q) + pdf.xfxQ2(-1, x*(1./z),Q*Q) + pdf.xfxQ2(-2, x*(1./z),Q*Q) + pdf.xfxQ2(-3, x*(1./z),Q*Q) + pdf.xfxQ2(-4, x*(1./z),Q*Q)),lower,upper,rtol = 1e-3)
        result = resultmid 
        errormid = 0 
        error = errormid
    return result

def Convolute(func1,matching, x,Q,p1=None):
    np.seterr(invalid='ignore')
    lower = x 
    upper = 1.
    q = [p1[0],Q]
    resultmid, errormid = integrate.quad(lambda z: (1./z)*func1(z,Q,p1)*matching(x*(1./z),q),lower,upper, limit=500, epsrel = 1.e-02,points=(x,1.))
    return resultmid

def Convolute_matching(matching1,matching2, x,Q,p1=None):
    np.seterr(invalid='ignore')
    lower = x 
    upper = 1.
    q = [p1[0],Q]
    resultmid, errormid = integrate.quad(lambda z: (1./z)*matching1(z,q)*matching2(x*(1./z),q),lower,upper, limit=500, epsrel = 1.e-02,points=(x,1.))
    return resultmid

def PDFConvolute_plus(func1,pdf,x,Q,pid,p1=None):
    plus1 = 0.
    plus2 = 0.
    error1 = 0.
    error2 = 0.
    if pid == 21:
        plus1, error1 = integrate.quad(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(pid, x*(1./z),Q*Q) - pdf.xfxQ2(pid, x,Q*Q)),x,1., limit=200, epsrel = 1.e-02, points=(x,1.))
        plus2, error2 = integrate.quad(lambda z: func1(z,Q,p1)*pdf.xfxQ2(pid,x,Q*Q),0.,x, limit=200, epsrel = 1.e-02, points=(0.,x))
        #plus1 = integrate.romberg(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(pid, x*(1./z),Q*Q) - pdf.xfxQ2(pid, x,Q*Q)),x,1., rtol = 1e-3)
        #plus2 = integrate.romberg(lambda z: func1(z,Q,p1)*pdf.xfxQ2(pid,x,Q*Q),0.,x, rtol = 1e-3)
    if pid == 5:
        plus1, error1 = integrate.quad(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(pid, x*(1./z),Q*Q) + pdf.xfxQ2(-pid, x*(1./z),Q*Q) - pdf.xfxQ2(pid,x,Q*Q) - pdf.xfxQ2(-pid,x,Q*Q) ),x,1., limit=200, epsrel = 1.e-02, points=(x,1.))
        plus2, error2 = integrate.quad(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(pid,x,Q*Q) + pdf.xfxQ2(-pid,x,Q*Q)),0.,x, limit=200, epsrel = 1.e-02, points=(0.,x))
        #plus1 = integrate.romberg(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(pid, x*(1./z),Q*Q) + pdf.xfxQ2(-pid, x*(1./z),Q*Q) - pdf.xfxQ2(pid,x,Q*Q) - pdf.xfxQ2(-pid,x,Q*Q) ),x,1., rtol = 1e-3)
        #plus2 = integrate.romberg(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(pid,x,Q*Q) + pdf.xfxQ2(-pid,x,Q*Q)),0.,x, rtol = 1e-3)
    if pid == 1:
        plus1, error1 = integrate.quad(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(1, x*(1./z),Q*Q) + pdf.xfxQ2(2, x*(1./z),Q*Q) + pdf.xfxQ2(3, x*(1./z),Q*Q) + pdf.xfxQ2(4, x*(1./z),Q*Q) + pdf.xfxQ2(-1, x*(1./z),Q*Q) + pdf.xfxQ2(-2, x*(1./z),Q*Q) + pdf.xfxQ2(-3, x*(1./z),Q*Q) + pdf.xfxQ2(-4, x*(1./z),Q*Q)-(pdf.xfxQ2(1, x,Q*Q) + pdf.xfxQ2(2, x,Q*Q) + pdf.xfxQ2(3, x,Q*Q) + pdf.xfxQ2(4, x,Q*Q) + pdf.xfxQ2(-1, x,Q*Q) + pdf.xfxQ2(-2, x,Q*Q) + pdf.xfxQ2(-3, x,Q*Q) + pdf.xfxQ2(-4, x,Q*Q))),x,1., limit=200, epsrel = 1.e-02, points=(x,1.))
        plus2, error2 = integrate.quad(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(1, x,Q*Q) + pdf.xfxQ2(2, x,Q*Q) + pdf.xfxQ2(3, x,Q*Q) + pdf.xfxQ2(4, x,Q*Q) + pdf.xfxQ2(-1, x,Q*Q) + pdf.xfxQ2(-2, x,Q*Q) + pdf.xfxQ2(-3, x,Q*Q) + pdf.xfxQ2(-4, x,Q*Q)),0.,x, limit=200, epsrel = 1.e-02, points=(0.,x))
        #plus1 = integrate.romberg(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(1, x*(1./z),Q*Q) + pdf.xfxQ2(2, x*(1./z),Q*Q) + pdf.xfxQ2(3, x*(1./z),Q*Q) + pdf.xfxQ2(4, x*(1./z),Q*Q) + pdf.xfxQ2(-1, x*(1./z),Q*Q) + pdf.xfxQ2(-2, x*(1./z),Q*Q) + pdf.xfxQ2(-3, x*(1./z),Q*Q) + pdf.xfxQ2(-4, x*(1./z),Q*Q)-(pdf.xfxQ2(1, x,Q*Q) + pdf.xfxQ2(2, x,Q*Q) + pdf.xfxQ2(3, x,Q*Q) + pdf.xfxQ2(4, x,Q*Q) + pdf.xfxQ2(-1, x,Q*Q) + pdf.xfxQ2(-2, x,Q*Q) + pdf.xfxQ2(-3, x,Q*Q) + pdf.xfxQ2(-4, x,Q*Q))),x,1., rtol = 1e-3)
        #plus2 = integrate.romberg(lambda z: func1(z,Q,p1)*(pdf.xfxQ2(1, x,Q*Q) + pdf.xfxQ2(2, x,Q*Q) + pdf.xfxQ2(3, x,Q*Q) + pdf.xfxQ2(4, x,Q*Q) + pdf.xfxQ2(-1, x,Q*Q) + pdf.xfxQ2(-2, x,Q*Q) + pdf.xfxQ2(-3, x,Q*Q) + pdf.xfxQ2(-4, x,Q*Q)),0.,x, rtol = 1e-3)
    return plus1 - plus2

def Convolute_plus_coeff(func1,matching, x,Q,p1=None):
    np.seterr(invalid='ignore')
    q = [p1[0],Q]
    plus1, error1 = integrate.quad(lambda z: func1(z,Q,p1)*((1./z)*matching(x*(1./z),q) - matching(x,q)),x,1., limit=200, epsrel = 1.e-02, points=(x,1.))
    plus2, error2 = integrate.quad(lambda z: func1(z,Q,p1)*matching(x,q),0.,x, limit=200, epsrel = 1.e-02, points=(0.,x))
    return plus1-plus2

def Convolute_plus_matching(func1,matching, x,Q,p1=None):
    np.seterr(invalid='ignore')
    q = [p1[0],Q]
    plus1, error1 = integrate.quad(lambda z: matching(z,q)*((1./z)*func1(x*(1./z),Q,p1) - func1(x,Q,p1)),x,1., limit=200, epsrel = 1.e-02, points=(x,1.))
    plus2, error2 = integrate.quad(lambda z: matching(z,q)*func1(x,Q,p1),0.,x, limit=200, epsrel = 1.e-02, points=(0.,x))
    return plus1-plus2

def Convolute_plus_matching_per_matching(matchingplus,matching2, x,Q,p1=None):
    np.seterr(invalid='ignore')
    q = [p1[0],Q]
    plus1, error1 = integrate.quad(lambda z: matchingplus(z,q)*((1./z)*matching2(x*(1./z),q) - matching2(x,q)),x,1., limit=200, epsrel = 1.e-02, points=(x,1.))
    plus2, error2 = integrate.quad(lambda z: matchingplus(z,q)*matching2(x,q),0.,x, limit=200, epsrel = 1.e-02, points=(0.,x))
    return plus1-plus2