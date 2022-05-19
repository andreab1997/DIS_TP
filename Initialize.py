#This initializes the grids needed for the evaluation of coefficients functions.
import ReadTxt as readt
from scipy.interpolate import interp2d, interp1d
import numpy as np
def InitializeQX():
    """
    Initialize the Q and z grids over which the other functions are built
    from the files in External
    """
    global QList 
    global ZList 
    QList = readt.read1D('./External/Cg2_2_m/Q.txt')
    ZList = readt.read1D('./External/Cg2_2_m/x.txt')

def InitializeCg2_m():
    """
    Initialize the Cg2 at NNLO massive function from the file in External
    """
    global Cg2m 
    Cg2m_array = readt.readND('./External/Cg2_2_m/Cg.txt')
    Cg2m = interp2d(ZList,QList,Cg2m_array,kind='linear')
def InitializeCq2_m():
    """
    Initialize the Cq2 at NNLO massive function from the file in External
    """
    global Cq2m 
    Cq2m_array = readt.readND('./External/Cq2_2_m/Cq.txt')
    Cq2m = interp2d(ZList,QList,Cq2m_array,kind='linear')
def InitializeCLg2_m():
    """
    Initialize the CgL at NNLO massive function from the file in External
    """
    global CLg2m 
    CLg2m_array = readt.readND('./External/CgL_2_m/CgL.txt')
    CLg2m = interp2d(ZList,QList,CLg2m_array,kind='linear')
def InitializeCLq2_m():
    """
    Initialize the CqL at NNLO massive function from the file in External
    """
    global CLq2m 
    CLq2m_array = readt.readND('./External/CqL_2_m/CqL.txt')
    CLq2m = interp2d(ZList,QList,CLq2m_array,kind='linear')

def InitializeCg2_til():
    """
    Initialize the Cg at NNLO tilde function from the file in External
    """
    global Cg2_til 
    Cg2_til_array_prov = np.array(readt.readND_python('./External/Cg_2_til/Cg2til.txt'))[:-1][:]
    Cg2_til_array = Cg2_til_array_prov.transpose()[:-1][:]
    Cg2_til = interp2d(ZList[:-1],QList[:-1],Cg2_til_array,kind='linear')
#def InitializeCg3_m():
#    global Cg3m 
#    Cg3m_array = readt.readND('./External/Cg2_3_m/Cg.txt')
#    Cg3m = interp2d(ZList,QList,Cg3m_array,kind='linear')
#def InitializeCq3_m():
#    global Cq3m 
#    Cq3m_array = readt.readND('./External/Cq2_3_m/Cq.txt')
#    Cq3m = interp2d(ZList,QList,Cq3m_array,kind='linear')
#def InitializeCLg3_m():
#    global CLg3m 
#    CLg3m_array = readt.readND('./External/CgL_3_m/CgL.txt')
#    CLg3m = interp2d(ZList,QList,CLg3m_array,kind='linear')
#def InitializeCLq3_m():
#    global CLq3m 
#    CLq3m_array = readt.readND('./External/CqL_3_m/CqL.txt')
#    CLq3m = interp2d(ZList,QList,CLq3m_array,kind='linear')
def InitializeMbg2():
    """
    Initialize the Mbg at NNLO matching function from the file in External
    """
    global Mbg2 
    Mbg2_array = readt.readND('./External/Mbg_2/Mbg2.txt')
    Mbg2 = interp2d(ZList,QList,Mbg2_array,kind='linear')
def InitializeMbq2():
    """
    Initialize the Mbg at NNLO matching function from the file in External
    """
    global Mbq2
    Mbq2_array = readt.readND('./External/Mbq_2/Mbq2.txt')
    Mbq2 = interp2d(ZList,QList,Mbq2_array,kind='linear')

def InitializeHPL():
    global HPL_x_array
    global HPL_0011
    HPL_x_array = readt.read1D('./External/HPL/HPL_x.txt')
    HPL_0011_array = readt.read1D('./External/HPL/HPL_0011.txt')
    HPL_0011 = interp1d(HPL_x_array,HPL_0011_array)
    global HPL_00011
    HPL_00011_array = readt.read1D('./External/HPL/HPL_00011.txt')
    HPL_00011 = interp1d(HPL_x_array,HPL_00011_array)
    global HPL_00101
    HPL_00101_array = readt.read1D('./External/HPL/HPL_00101.txt')
    HPL_00101 = interp1d(HPL_x_array,HPL_00101_array)
    global HPL_00111
    HPL_00111_array = readt.read1D('./External/HPL/HPL_00111.txt')
    HPL_00111 = interp1d(HPL_x_array,HPL_00111_array)
    global HPL_01011
    HPL_01011_array = readt.read1D('./External/HPL/HPL_01011.txt')
    HPL_01011 = interp1d(HPL_x_array,HPL_01011_array)
    global HPL_0m1m1m1
    HPL_0m1m1m1_array = readt.read1D('./External/HPL/HPL_0m1m1m1.txt')
    HPL_0m1m1m1 = interp1d(HPL_x_array,HPL_0m1m1m1_array)
    global HPL_0m101
    HPL_0m101_array = readt.read1D('./External/HPL/HPL_0m101.txt')
    HPL_0m101 = interp1d(HPL_x_array,HPL_0m101_array)
    global HPL_00m1m1
    HPL_00m1m1_array = readt.read1D('./External/HPL/HPL_00m1m1.txt')
    HPL_00m1m1 = interp1d(HPL_x_array,HPL_00m1m1_array)
    global HPL_00m11
    HPL_00m11_array = readt.read1D('./External/HPL/HPL_00m11.txt')
    HPL_00m11 = interp1d(HPL_x_array,HPL_00m11_array)
    global HPL_001m1
    HPL_001m1_array = readt.read1D('./External/HPL/HPL_001m1.txt')
    HPL_001m1 = interp1d(HPL_x_array,HPL_001m1_array)
    global HPL_0m10m1m1
    HPL_0m10m1m1_array = readt.read1D('./External/HPL/HPL_0m10m1m1.txt')
    HPL_0m10m1m1 = interp1d(HPL_x_array,HPL_0m10m1m1_array)
    global HPL_00m1m1m1
    HPL_00m1m1m1_array = readt.read1D('./External/HPL/HPL_00m1m1m1.txt')
    HPL_00m1m1m1 = interp1d(HPL_x_array,HPL_00m1m1m1_array)
    global HPL_00m10m1
    HPL_00m10m1_array = readt.read1D('./External/HPL/HPL_00m10m1.txt')
    HPL_00m10m1 = interp1d(HPL_x_array,HPL_00m10m1_array)
    global HPL_00m101
    HPL_00m101_array = readt.read1D('./External/HPL/HPL_00m101.txt')
    HPL_00m101 = interp1d(HPL_x_array,HPL_00m101_array)
    global HPL_000m1m1
    HPL_000m1m1_array = readt.read1D('./External/HPL/HPL_000m1m1.txt')
    HPL_000m1m1 = interp1d(HPL_x_array,HPL_000m1m1_array)
    global HPL_000m11
    HPL_000m11_array = readt.read1D('./External/HPL/HPL_000m11.txt')
    HPL_000m11 = interp1d(HPL_x_array,HPL_000m11_array)
    global HPL_0001m1
    HPL_0001m1_array = readt.read1D('./External/HPL/HPL_0001m1.txt')
    HPL_0001m1 = interp1d(HPL_x_array,HPL_0001m1_array)
    global HPL_0010m1
    HPL_0010m1_array = readt.read1D('./External/HPL/HPL_0010m1.txt')
    HPL_0010m1 = interp1d(HPL_x_array,HPL_0010m1_array)
    global HPL_0m1m11
    HPL_0m1m11_array = readt.read1D('./External/HPL/HPL_0m1m11.txt')
    HPL_0m1m11 = interp1d(HPL_x_array,HPL_0m1m11_array)
    global HPL_0m11m1
    HPL_0m11m1_array = readt.read1D('./External/HPL/HPL_0m11m1.txt')
    HPL_0m11m1 = interp1d(HPL_x_array,HPL_0m11m1_array)
    global HPL_01m1m1
    HPL_01m1m1_array = readt.read1D('./External/HPL/HPL_01m1m1.txt')
    HPL_01m1m1 = interp1d(HPL_x_array,HPL_01m1m1_array)
    global HPL_0m111
    HPL_0m111_array = readt.read1D('./External/HPL/HPL_0m111.txt')
    HPL_0m111 = interp1d(HPL_x_array,HPL_0m111_array)
    global HPL_01m11
    HPL_01m11_array = readt.read1D('./External/HPL/HPL_01m11.txt')
    HPL_01m11 = interp1d(HPL_x_array,HPL_01m11_array)
    global HPL_011m1
    HPL_011m1_array = readt.read1D('./External/HPL/HPL_011m1.txt')
    HPL_011m1 = interp1d(HPL_x_array,HPL_011m1_array)
    global HPL_0m1011
    HPL_0m1011_array = readt.read1D('./External/HPL/HPL_0m1011.txt')
    HPL_0m1011 = interp1d(HPL_x_array,HPL_0m1011_array)
    global HPL_0m1m101
    HPL_0m1m101_array = readt.read1D('./External/HPL/HPL_0m1m101.txt')
    HPL_0m1m101 = interp1d(HPL_x_array,HPL_0m1m101_array)
    global HPL_0m1m11m1
    HPL_0m1m11m1_array = readt.read1D('./External/HPL/HPL_0m1m11m1.txt')
    HPL_0m1m11m1 = interp1d(HPL_x_array,HPL_0m1m11m1_array)
    global HPL_0m10m11
    HPL_0m10m11_array = readt.read1D('./External/HPL/HPL_0m10m11.txt')
    HPL_0m10m11 = interp1d(HPL_x_array,HPL_0m10m11_array)
    global HPL_0m101m1
    HPL_0m101m1_array = readt.read1D('./External/HPL/HPL_0m101m1.txt')
    HPL_0m101m1 = interp1d(HPL_x_array,HPL_0m101m1_array)
    global HPL_0m11m1m1
    HPL_0m11m1m1_array = readt.read1D('./External/HPL/HPL_0m11m1m1.txt')
    HPL_0m11m1m1 = interp1d(HPL_x_array,HPL_0m11m1m1_array)
    global HPL_00m1m11
    HPL_00m1m11_array = readt.read1D('./External/HPL/HPL_00m1m11.txt')
    HPL_00m1m11 = interp1d(HPL_x_array,HPL_00m1m11_array)
    global HPL_00m11m1
    HPL_00m11m1_array = readt.read1D('./External/HPL/HPL_00m11m1.txt')
    HPL_00m11m1 = interp1d(HPL_x_array,HPL_00m11m1_array)
    global HPL_00m111
    HPL_00m111_array = readt.read1D('./External/HPL/HPL_00m111.txt')
    HPL_00m111 = interp1d(HPL_x_array,HPL_00m111_array)
    global HPL_001m1m1
    HPL_001m1m1_array = readt.read1D('./External/HPL/HPL_001m1m1.txt')
    HPL_001m1m1 = interp1d(HPL_x_array,HPL_001m1m1_array)
    global HPL_001m11
    HPL_001m11_array = readt.read1D('./External/HPL/HPL_001m11.txt')
    HPL_001m11 = interp1d(HPL_x_array,HPL_001m11_array)
    global HPL_0011m1
    HPL_0011m1_array = readt.read1D('./External/HPL/HPL_0011m1.txt')
    HPL_0011m1 = interp1d(HPL_x_array,HPL_0011m1_array)
    global HPL_01m1m1m1
    HPL_01m1m1m1_array = readt.read1D('./External/HPL/HPL_01m1m1m1.txt')
    HPL_01m1m1m1 = interp1d(HPL_x_array,HPL_01m1m1m1_array)
    global HPL_0m1m1m1m1
    HPL_0m1m1m1m1_array = readt.read1D('./External/HPL/HPL_0m1m1m1m1.txt')
    HPL_0m1m1m1m1 = interp1d(HPL_x_array,HPL_0m1m1m1m1_array)
    global HPL_0m1m1m11
    HPL_0m1m1m11_array = readt.read1D('./External/HPL/HPL_0m1m1m11.txt')
    HPL_0m1m1m11 = interp1d(HPL_x_array,HPL_0m1m1m11_array)

def InitializeMbg_3():
    """
    Initialize the Mbg at N3LO matching condition from the file in External
    """
    global Mbg3 
    Mbg3_array = np.array(readt.readND_python('./External/Mbg_3/Mbg3.txt')).transpose()
    Mbg3 = interp2d(HPL_x_array,QList,Mbg3_array,kind='linear')

    
