#This is the actual code for computing structure functions at fixed order, in zero-mass scheme and in a matched massive scheme
from tools import PDFConvolute_plus,PDFConvolute
import MassiveCoeffFunc
import TildeCoeffFunc
import MasslessCoeffFunc
import lhapdf
import numpy as np
import Initialize as Ini

def Initialize_all():
    """
    Initialize all the needed global lists
    """
    Ini.InitializeQX()
    Ini.InitializeCg2_m()
    Ini.InitializeCq2_m()
    Ini.InitializeCLg2_m()
    Ini.InitializeCLq2_m()
    Ini.InitializeMbg2()
    Ini.InitializeMbq2()
    Ini.InitializeHPL()
    #TODO:add the 4 new funcs

def F2_FO(order,pdf,x,Q,muF_ratio=1,muR_ratio=1):
    """
    Compute the FO result for the structure function F2 

    Parameters:
        order : int
            requested perturbative order (0 == LO, 1 == NLO,...)
        pdf : str or list(str)
            pdf(s) to be used 
        x : float
            x-value 
        Q : float
            Q-value
        muF_ratio : float
            ratio to Q of the factorization scale
        muR_ratio : float
            ratio to Q of the renormalization scale
    Returns:
            : float
            result
    """
    lhapdf.setVerbosity(0)
    Mypdf = None
    if isinstance(pdf,list):
        Mypdf = lhapdf.mkPDF(pdf[order-1],0)
    elif isinstance(pdf,str):
        Mypdf = lhapdf.mkPDF(pdf,0)
    muF = muF_ratio*Q
    muR = muR_ratio*Q
    res = 0.
    if order >=  0:
        res += 0.
    if order >=  1:
        res += (1/(4*np.pi))*Mypdf.alphasQ(muR)*PDFConvolute(MassiveCoeffFunc.Cg_1_m_reg,Mypdf,x,Q,21,p1=[Mypdf.quarkMass(5)])
    if order >=  2:
        res += pow((1/(4*np.pi))*Mypdf.alphasQ(muR),2)*(PDFConvolute(MassiveCoeffFunc.Cg_2_m_reg,Mypdf,x,Q,21,p1=[Mypdf.quarkMass(5)]) + PDFConvolute(MassiveCoeffFunc.Cq_2_m_reg,Mypdf,x,Q,1,p1=[Mypdf.quarkMass(5)]))
    if order >=  3:
        res += pow((1/(4*np.pi))*Mypdf.alphasQ(muR),3)*(PDFConvolute(MassiveCoeffFunc.Cg_3_m_reg,Mypdf,x,Q,21,p1=[Mypdf.quarkMass(5)]) + PDFConvolute(MassiveCoeffFunc.Cq_3_m_reg,Mypdf,x,Q,1,p1=[Mypdf.quarkMass(5)]))
    return res 

def F2_R(order,pdf,x,Q,muF_ratio=1,muR_ratio=1):
    """
    Compute the R result for the structure function F2 

    Parameters:
        order : int
            requested perturbative order (0 == LO, 1 == NLO,...)
        pdf : str or list(str)
            pdf(s) to be used 
        x : float
            x-value 
        Q : float
            Q-value
        muF_ratio : float
            ratio to Q of the factorization scale
        muR_ratio : float
            ratio to Q of the renormalization scale
    Returns:
            : float
            result
    """
    lhapdf.setVerbosity(0)
    Mypdf = None
    if isinstance(pdf,list):
        Mypdf = lhapdf.mkPDF(pdf[order-1],0)
    elif isinstance(pdf,str):
        Mypdf = lhapdf.mkPDF(pdf,0)
    muF = muF_ratio*Q
    muR = muR_ratio*Q
    res = 0.
    if order >=  0:
        res += 0.
    if order >=  1:
        nll_reg = (1/(4*np.pi))*Mypdf.alphasQ(muR)*PDFConvolute(MasslessCoeffFunc.Cg_1_reg,Mypdf,x,Q,21)
        nll_local = MasslessCoeffFunc.Cb_0_loc(x,Q)*(Mypdf.xfxQ2(5,x,Q*Q) + Mypdf.xfxQ2(-5,x,Q*Q))
        res += nll_reg + nll_local
    if order >=  2:
        nnll_reg = (1/(4*np.pi))*Mypdf.alphasQ(muR)*((1/(4*np.pi))*Mypdf.alphasQ(muR)*(PDFConvolute(MasslessCoeffFunc.Cg_2_reg,Mypdf,x,Q,21) + PDFConvolute(MasslessCoeffFunc.Cq_2_reg,Mypdf,x,Q,1) ) + PDFConvolute(MasslessCoeffFunc.Cb_1_reg,Mypdf,x,Q,5) )
        nnll_local = (1/(4*np.pi))*Mypdf.alphasQ(muR)*MasslessCoeffFunc.Cb_1_loc(x,Q)*(Mypdf.xfxQ2(5,x,Q*Q) + Mypdf.xfxQ2(-5,x,Q*Q))
        nnll_sing = (1/(4*np.pi))*Mypdf.alphasQ(muR)*PDFConvolute_plus(MasslessCoeffFunc.Cb_1_sing,Mypdf,x,Q,5)
        res += nnll_reg + nnll_local + nnll_sing
    if order >= 3: 
        n3ll_reg = (((1/(4*np.pi))*Mypdf.alphasQ(muR))**2)*((1/(4*np.pi))*Mypdf.alphasQ(muR)*(PDFConvolute(MasslessCoeffFunc.Cg_3_reg,Mypdf,x,Q,21) + PDFConvolute(MasslessCoeffFunc.Cq_3_reg,Mypdf,x,Q,1) ) + PDFConvolute(MasslessCoeffFunc.Cb_2_reg,Mypdf,x,Q,5) )
        n3ll_local = (((1/(4*np.pi))*Mypdf.alphasQ(muR))**2)*(MasslessCoeffFunc.Cb_2_loc(x,Q)*(Mypdf.xfxQ2(5,x,Q*Q) + Mypdf.xfxQ2(-5,x,Q*Q))+ (1/(4*np.pi))*Mypdf.alphasQ(muR) * (MasslessCoeffFunc.Cg_3_loc(x,Q)*Mypdf.xfxQ2(5,x,Q*Q) + MasslessCoeffFunc.Cq_3_loc(x,Q)*(Mypdf.xfxQ2(1,x,Q*Q)+Mypdf.xfxQ2(-1,x,Q*Q)+Mypdf.xfxQ2(2,x,Q*Q)+Mypdf.xfxQ2(-2,x,Q*Q)+ Mypdf.xfxQ2(3,x,Q*Q)+Mypdf.xfxQ2(-3,x,Q*Q)+Mypdf.xfxQ2(4,x,Q*Q)+Mypdf.xfxQ2(-4,x,Q*Q))))
        n3ll_sing = (1/(4*np.pi))*Mypdf.alphasQ(muR)*(1/(4*np.pi))*Mypdf.alphasQ(muR)*PDFConvolute_plus(MasslessCoeffFunc.Cb_2_sing,Mypdf,x,Q,5)
        res += n3ll_reg + n3ll_local + n3ll_sing
    return res
    
def F2_M(order,meth,pdf,x,Q,muF_ratio=1,muR_ratio=1):
    """
    Compute the M result for the structure function F2 

    Parameters:
        order : int
            requested perturbative order (0 == LO, 1 == NLO,...)
        meth : str
            method to be used (our, fonll)
        pdf : str or list(str)
            pdf(s) to be used 
        x : float
            x-value 
        Q : float
            Q-value
        muF_ratio : float
            ratio to Q of the factorization scale
        muR_ratio : float
            ratio to Q of the renormalization scale
    Returns:
            : float
            result
    """
    lhapdf.setVerbosity(0)
    Mypdf = None
    if isinstance(pdf,list):
        Mypdf = lhapdf.mkPDF(pdf[order-1],0)
    elif isinstance(pdf,str):
        Mypdf = lhapdf.mkPDF(pdf,0)
    muF = muF_ratio*Q
    muR = muR_ratio*Q
    res = 0.
    if meth == 'our':
        if order >= 0:
            res += 0.
        if order >= 1:
            nlo_nll_reg = (1/(4*np.pi))*Mypdf.alphasQ(muR)*PDFConvolute(TildeCoeffFunc.Cg_1_til_reg,Mypdf,x,Q,21,p1=[Mypdf.quarkMass(5)])
            nlo_nll_local = MasslessCoeffFunc.Cb_0_loc(x,Q)*(Mypdf.xfxQ2(5,x,Q*Q) + Mypdf.xfxQ2(-5,x,Q*Q))
            res += nlo_nll_reg + nlo_nll_local
        if order >= 2:
            nnlo_nnll_reg = (1/(4*np.pi))*Mypdf.alphasQ(muR)*((1/(4*np.pi))*Mypdf.alphasQ(muR)*(PDFConvolute(TildeCoeffFunc.Cg_2_til_reg,Mypdf,x,Q,21,p1=[Mypdf.quarkMass(5)]) + PDFConvolute(TildeCoeffFunc.Cq_2_til_reg,Mypdf,x,Q,1,p1=[Mypdf.quarkMass(5)]) ) + PDFConvolute(MasslessCoeffFunc.Cb_1_reg,Mypdf,x,Q,5) )
            nnlo_nnll_local = (1/(4*np.pi))*Mypdf.alphasQ(muR)*MasslessCoeffFunc.Cb_1_loc(x,Q)*(Mypdf.xfxQ2(5,x,Q*Q) + Mypdf.xfxQ2(-5,x,Q*Q))
            nnlo_nnll_sing = (1/(4*np.pi))*Mypdf.alphasQ(muR)*PDFConvolute_plus(MasslessCoeffFunc.Cb_1_sing,Mypdf,x,Q,5)
            res += nnlo_nnll_reg + nnlo_nnll_local + nnlo_nnll_sing
        if order >= 3:
            n3lo_n3ll_reg = (((1/(4*np.pi))*Mypdf.alphasQ(muR))**2)*((1/(4*np.pi))*Mypdf.alphasQ(muR)*(PDFConvolute(TildeCoeffFunc.Cg_3_til_reg,Mypdf,x,Q,21,p1=[Mypdf.quarkMass(5)]) + PDFConvolute(TildeCoeffFunc.Cq_3_til_reg,Mypdf,x,Q,1,p1=[Mypdf.quarkMass(5)]) ) + PDFConvolute(MasslessCoeffFunc.Cb_2_reg,Mypdf,x,Q,5) )
            n3lo_n3ll_local = (((1/(4*np.pi))*Mypdf.alphasQ(muR))**2)*MasslessCoeffFunc.Cb_2_loc(x,Q)*(Mypdf.xfxQ2(5,x,Q*Q) + Mypdf.xfxQ2(-5,x,Q*Q))
            n3lo_n3ll_sing = (1/(4*np.pi))*Mypdf.alphasQ(muR)*(1/(4*np.pi))*Mypdf.alphasQ(muR)*PDFConvolute_plus(MasslessCoeffFunc.Cb_2_sing,Mypdf,x,Q,5)
            res += n3lo_n3ll_reg + n3lo_n3ll_local + n3lo_n3ll_sing
    if meth == 'fonll':
        if order >= 0:
            res += MasslessCoeffFunc.Cb_0_loc(x,Q)*(Mypdf.xfxQ2(5,x,Q*Q) + Mypdf.xfxQ2(-5,x,Q*Q)) 
        if order >= 1:
            nlo_nll_reg = (1/(4*np.pi))*Mypdf.alphasQ(muR)*(PDFConvolute(TildeCoeffFunc.Cg_1_til_reg,Mypdf,x,Q,21,p1=[Mypdf.quarkMass(5)]) + PDFConvolute(MasslessCoeffFunc.Cb_1_reg,Mypdf,x,Q,5,p1=[Mypdf.quarkMass(5)])) 
            nlo_nll_local = ((1/(4*np.pi))*Mypdf.alphasQ(muR)*MasslessCoeffFunc.Cb_1_loc(x,Q)*(Mypdf.xfxQ2(5,x,Q*Q) + Mypdf.xfxQ2(-5,x,Q*Q))) 
            nlo_nll_singular = (1/(4*np.pi))*Mypdf.alphasQ(muR)*PDFConvolute_plus(MasslessCoeffFunc.Cb_1_sing,Mypdf,x,Q,5,p1=[Mypdf.quarkMass(5)])
            res += nlo_nll_reg + nlo_nll_local + nlo_nll_singular
        if order >= 2:
            nnlo_nnll_reg = (1/(4*np.pi))*Mypdf.alphasQ(muR)*(1/(4*np.pi))*Mypdf.alphasQ(muR)*(PDFConvolute(TildeCoeffFunc.Cg_2_til_reg,Mypdf,x,Q,21,p1=[Mypdf.quarkMass(5)]) + PDFConvolute(TildeCoeffFunc.Cq_2_til_reg,Mypdf,x,Q,1,p1=[Mypdf.quarkMass(5)])  + PDFConvolute(MasslessCoeffFunc.Cb_2_reg,Mypdf,x,Q,5) )
            nnlo_nnll_local = (1/(4*np.pi))*Mypdf.alphasQ(muR)*(1/(4*np.pi))*Mypdf.alphasQ(muR)*MasslessCoeffFunc.Cb_2_loc(x,Q)*(Mypdf.xfxQ2(5,x,Q*Q) + Mypdf.xfxQ2(-5,x,Q*Q))
            nnlo_nnll_sing = (1/(4*np.pi))*Mypdf.alphasQ(muR)*(1/(4*np.pi))*Mypdf.alphasQ(muR)*PDFConvolute_plus(MasslessCoeffFunc.Cb_2_sing,Mypdf,x,Q,5)
            res += nnlo_nnll_reg + nnlo_nnll_local + nnlo_nnll_sing
        if order >= 3:
            res += 0
    return res

def FL_FO(order,pdf,x,Q,muF_ratio=1,muR_ratio=1):
    """
    Compute the FO results for the structure function FL 

    Parameters:
        order : int
            requested perturbative order (0 == LO, 1 == NLO,...)
        pdf : str or list(str)
            pdf(s) to be used 
        x : float
            x-value 
        Q : float
            Q-value
        muF_ratio : float
            ratio to Q of the factorization scale
        muR_ratio : float
            ratio to Q of the renormalization scale
    Returns:
            : float
            result
    """
    lhapdf.setVerbosity(0)
    Mypdf = None
    if isinstance(pdf,list):
        Mypdf = lhapdf.mkPDF(pdf[order-1],0)
    elif isinstance(pdf,str):
        Mypdf = lhapdf.mkPDF(pdf,0)
    muF = muF_ratio*Q
    muR = muR_ratio*Q
    res = 0. 
    if order >= 0:
        res += 0.
    if order >= 1:
        res += (1/(4*np.pi))*Mypdf.alphasQ(muR)*PDFConvolute(MassiveCoeffFunc.CLg_1_m_reg,Mypdf,x,Q,21,p1=[Mypdf.quarkMass(5)])
    if order >= 2:
        res += pow((1/(4*np.pi))*Mypdf.alphasQ(muR),2)*(PDFConvolute(MassiveCoeffFunc.CLg_2_m_reg,Mypdf,x,Q,21,p1=[Mypdf.quarkMass(5)]) + PDFConvolute(MassiveCoeffFunc.CLq_2_m_reg,Mypdf,x,Q,1,p1=[Mypdf.quarkMass(5)]))
    if order >= 3:
        res += pow((1/(4*np.pi))*Mypdf.alphasQ(muR),3)*(PDFConvolute(MassiveCoeffFunc.CLg_3_m_reg,Mypdf,x,Q,21,p1=[Mypdf.quarkMass(5)]) + PDFConvolute(MassiveCoeffFunc.CLq_3_m_reg,Mypdf,x,Q,1,p1=[Mypdf.quarkMass(5)]))
    return res 

def FL_R(order,pdf,x,Q,muF_ratio=1,muR_ratio=1):
    """
    Compute the R result for the structure function FL

    Parameters:
        order : int
            requested perturbative order (0 == LO, 1 == NLO,...)
        pdf : str or list(str)
            pdf(s) to be used 
        x : float
            x-value 
        Q : float
            Q-value
        muF_ratio : float
            ratio to Q of the factorization scale
        muR_ratio : float
            ratio to Q of the renormalization scale
    Returns:
            : float
            result
    """
    lhapdf.setVerbosity(0)
    Mypdf = None
    if isinstance(pdf,list):
        Mypdf = lhapdf.mkPDF(pdf[order-1],0)
    elif isinstance(pdf,str):
        Mypdf = lhapdf.mkPDF(pdf,0)
    muF = muF_ratio*Q
    muR = muR_ratio*Q
    res = 0.
    if order >= 0:
        res += 0.
    if order >= 1:
        res += (1/(4*np.pi))*Mypdf.alphasQ(muR)*PDFConvolute(MasslessCoeffFunc.CLg_1_reg,Mypdf,x,Q,21)
    if order >= 2:
        nnll_reg = (1/(4*np.pi))*Mypdf.alphasQ(muR)*((1/(4*np.pi))*Mypdf.alphasQ(muR)*(PDFConvolute(MasslessCoeffFunc.CLg_2_reg,Mypdf,x,Q,21) + PDFConvolute(MasslessCoeffFunc.CLq_2_reg,Mypdf,x,Q,1) ) + PDFConvolute(MasslessCoeffFunc.CLb_1_reg,Mypdf,x,Q,5) )
        res += nnll_reg 
    if order >= 3:
        n3ll_reg = (((1/(4*np.pi))*Mypdf.alphasQ(muR))**2)*((1/(4*np.pi))*Mypdf.alphasQ(muR)*(PDFConvolute(MasslessCoeffFunc.CLg_3_reg,Mypdf,x,Q,21) + PDFConvolute(MasslessCoeffFunc.CLq_3_reg,Mypdf,x,Q,1) ) + PDFConvolute(MasslessCoeffFunc.CLb_2_reg,Mypdf,x,Q,5) )
        n3ll_loc = (((1/(4*np.pi))*Mypdf.alphasQ(muR))**2)*(MasslessCoeffFunc.CLb_2_loc(x,Q)*(Mypdf.xfxQ2(5,x,Q*Q) + Mypdf.xfxQ2(-5,x,Q*Q)))
        res += n3ll_reg + n3ll_loc
    return res
    
def FL_M(order,meth,pdf,x,Q,muF_ratio=1,muR_ratio=1):
    """
    Compute the M result for the structure function FL 

    Parameters:
        order : int
            requested perturbative order (0 == LO, 1 == NLO,...)
        meth : str
            method to be used (our, fonll)
        pdf : str or list(str)
            pdf(s) to be used 
        x : float
            x-value 
        Q : float
            Q-value
        muF_ratio : float
            ratio to Q of the factorization scale
        muR_ratio : float
            ratio to Q of the renormalization scale
    Returns:
            : float
            result
    """
    lhapdf.setVerbosity(0)
    Mypdf = None
    if isinstance(pdf,list):
        Mypdf = lhapdf.mkPDF(pdf[order-1],0)
    elif isinstance(pdf,str):
        Mypdf = lhapdf.mkPDF(pdf,0)
    muF = muF_ratio*Q
    muR = muR_ratio*Q
    res = 0. 
    if meth ==  'our':
        if order >=  0:
            res += 0.
        if order >=  1:
            res += (1/(4*np.pi))*Mypdf.alphasQ(muR)*PDFConvolute(TildeCoeffFunc.CLg_1_til_reg,Mypdf,x,Q,21,p1=[Mypdf.quarkMass(5)])
        if order >=  2:
            nnlo_nnll_reg = (1/(4*np.pi))*Mypdf.alphasQ(muR)*((1/(4*np.pi))*Mypdf.alphasQ(muR)*(PDFConvolute(TildeCoeffFunc.CLg_2_til_reg,Mypdf,x,Q,21,p1=[Mypdf.quarkMass(5)]) + PDFConvolute(TildeCoeffFunc.CLq_2_til_reg,Mypdf,x,Q,1,p1=[Mypdf.quarkMass(5)]) ) + PDFConvolute(MasslessCoeffFunc.CLb_1_reg,Mypdf,x,Q,5) )
            res += nnlo_nnll_reg 
        if order >= 3:
            n3lo_n3ll_reg = (((1/(4*np.pi))*Mypdf.alphasQ(muR))**2)*((1/(4*np.pi))*Mypdf.alphasQ(muR)*(PDFConvolute(TildeCoeffFunc.CLg_3_til_reg,Mypdf,x,Q,21,p1=[Mypdf.quarkMass(5)]) + PDFConvolute(TildeCoeffFunc.CLq_3_til_reg,Mypdf,x,Q,1,p1=[Mypdf.quarkMass(5)]) ) + PDFConvolute(MasslessCoeffFunc.CLb_2_reg,Mypdf,x,Q,5) )
            n3lo_n3ll_loc = (1/(4*np.pi))*Mypdf.alphasQ(muR)*(1/(4*np.pi))*Mypdf.alphasQ(muR)*MasslessCoeffFunc.CLb_2_loc(x,Q)*(Mypdf.xfxQ2(5,x,Q*Q) + Mypdf.xfxQ2(-5,x,Q*Q))
            res += n3lo_n3ll_loc + n3lo_n3ll_reg
    if meth ==  'fonll':
        if order >=  0:
            res += 0.
        if order >=  1:
            res += (1/(4*np.pi))*Mypdf.alphasQ(muR)*(PDFConvolute(TildeCoeffFunc.CLg_1_til_reg,Mypdf,x,Q,21,p1=[Mypdf.quarkMass(5)])+PDFConvolute(MasslessCoeffFunc.CLb_1_reg,Mypdf,x,Q,5,p1=[Mypdf.quarkMass(5)]))
        if order >=  2:
            nnlo_nnll_reg = (1/(4*np.pi))*Mypdf.alphasQ(muR)*(1/(4*np.pi))*Mypdf.alphasQ(muR)*(PDFConvolute(TildeCoeffFunc.CLg_2_til_reg,Mypdf,x,Q,21,p1=[Mypdf.quarkMass(5)]) + PDFConvolute(TildeCoeffFunc.CLq_2_til_reg,Mypdf,x,Q,1,p1=[Mypdf.quarkMass(5)])  + PDFConvolute(MasslessCoeffFunc.CLb_2_reg,Mypdf,x,Q,5) )
            nnlo_nnll_loc = (1/(4*np.pi))*Mypdf.alphasQ(muR)*(1/(4*np.pi))*Mypdf.alphasQ(muR)*MasslessCoeffFunc.CLb_2_loc(x,Q)*(Mypdf.xfxQ2(5,x,Q*Q) + Mypdf.xfxQ2(-5,x,Q*Q))
            res += nnlo_nnll_reg + nnlo_nnll_loc
        if order >= 3:
            res += 0
    return res
