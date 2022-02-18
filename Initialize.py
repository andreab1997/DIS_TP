import ReadTxt as readt
from scipy.interpolate import interp2d, interp1d
def InitializeQX():
    global QList 
    global ZList 
    QList = readt.read1D('./External/Cg2_2_m/Q.txt')
    ZList = readt.read1D('./External/Cg2_2_m/x.txt')

def InitializeCg2_m():
    global Cg2m 
    Cg2m_array = readt.readND('./External/Cg2_2_m/Cg.txt')
    Cg2m = interp2d(ZList,QList,Cg2m_array,kind='linear')
def InitializeCq2_m():
    global Cq2m 
    Cq2m_array = readt.readND('./External/Cq2_2_m/Cq.txt')
    Cq2m = interp2d(ZList,QList,Cq2m_array,kind='linear')
def InitializeCLg2_m():
    global CLg2m 
    CLg2m_array = readt.readND('./External/CgL_2_m/CgL.txt')
    CLg2m = interp2d(ZList,QList,CLg2m_array,kind='linear')
def InitializeCLq2_m():
    global CLq2m 
    CLq2m_array = readt.readND('./External/CqL_2_m/CqL.txt')
    CLq2m = interp2d(ZList,QList,CLq2m_array,kind='linear')
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
    global Mbg2 
    Mbg2_array = readt.readND('./External/Mbg_2/Mbg2.txt')
    Mbg2 = interp2d(ZList,QList,Mbg2_array,kind='linear')
def InitializeMbq2():
    global Mbq2
    Mbq2_array = readt.readND('./External/Mbq_2/Mbq2.txt')
    Mbq2 = interp2d(ZList,QList,Mbq2_array,kind='linear')
