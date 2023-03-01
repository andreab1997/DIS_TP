"""Initialize the grids needed for the evaluation of coefficients functions."""
import pathlib

import numpy as np
from scipy.interpolate import interp1d, interp2d

from . import ReadTxt as readt
from .logging import console

PATH_TO_GLOBAL = str(pathlib.Path(__file__).parent.parent.parent)


def InitializeQX():
    """Initialize the Q and z grids over which the other functions are built
    from the files in External (for Niccolo functions).
    """
    global QList
    global ZList
    QList = readt.read1D(PATH_TO_GLOBAL + "/External/Q.txt")
    ZList = readt.read1D_Nic(PATH_TO_GLOBAL + "/External/x.txt")


def append_empty(grid_list, nf_list):
    """Append an empty element when only bottom is need."""
    if nf_list == [5]:
        grid_list.append(None)
    if nf_list == [6]:
        grid_list.append(None)
        grid_list.append(None)
    return grid_list


# NNLO grids
def InitializeCg2_til(nf_list):
    """Initialize the Cg at NNLO tilde function from the file in External."""
    global Cg2_til
    Cg2_til = []
    append_empty(Cg2_til, nf_list)
    for nf in nf_list:
        Cg2_til_array = np.array(
            readt.readND(PATH_TO_GLOBAL + f"/External/Cg_2_til/Cg2til_nf{nf}.txt")
        )
        Cg2_til.append(interp2d(ZList[:-1], QList, Cg2_til_array.T, kind="quintic"))


def InitializeCq2_til(nf_list):
    """Initialize the Cg at NNLO tilde function from the file in External."""
    global Cq2_til
    Cq2_til = []
    append_empty(Cq2_til, nf_list)
    for nf in nf_list:
        Cq2_til_array = np.array(
            readt.readND(PATH_TO_GLOBAL + f"/External/Cq_2_til/Cq2til_nf{nf}.txt")
        )
        Cq2_til.append(interp2d(ZList[:-1], QList, Cq2_til_array.T, kind="quintic"))


def InitializeCLg2_til(nf_list):
    """Initialize the Cg at NNLO tilde function from the file in External."""
    global CLg2_til
    CLg2_til = []
    append_empty(CLg2_til, nf_list)
    for nf in nf_list:
        CLg2_til_array = np.array(
            readt.readND(PATH_TO_GLOBAL + f"/External/CLg_2_til/CLg2til_nf{nf}.txt")
        )
        CLg2_til.append(interp2d(ZList[:-1], QList, CLg2_til_array.T, kind="quintic"))


def InitializeCLq2_til(nf_list):
    """Initialize the Cq at NNLO tilde function from the file in External."""
    global CLq2_til
    CLq2_til = []
    append_empty(CLq2_til, nf_list)
    for nf in nf_list:
        CLq2_til_array = np.array(
            readt.readND(PATH_TO_GLOBAL + f"/External/CLq_2_til/CLq2til_nf{nf}.txt")
        )
        CLq2_til.append(interp2d(ZList[:-1], QList, CLq2_til_array.T, kind="quintic"))


def InitializeMbg2(nf_list):
    """Initialize the Mbg at NNLO matching function from the file in External."""
    global Mbg2
    Mbg2 = []
    append_empty(Mbg2, nf_list)
    for nf in nf_list:
        Mbg2_array = readt.readND(PATH_TO_GLOBAL + f"/External/Mbg_2/Mbg2_nf{nf}.txt")
        Mbg2.append(interp2d(ZList, QList, np.array(Mbg2_array).T, kind="quintic"))


def InitializeMbq2(nf_list):
    """Initialize the Mbg at NNLO matching function from the file in External."""
    global Mbq2
    Mbq2 = []
    append_empty(Mbq2, nf_list)
    for nf in nf_list:
        Mbq2_array = readt.readND(PATH_TO_GLOBAL + f"/External/Mbq_2/Mbq2_nf{nf}.txt")
        Mbq2.append(interp2d(ZList, QList, np.array(Mbq2_array).T, kind="quintic"))


# N3LO grids
def InitializeMbg_3(nf_list):
    """Initialize the Mbg at N3LO matching condition from the file in External."""
    global Mbg3
    Mbg3 = []
    append_empty(Mbg3, nf_list)
    for nf in nf_list:
        Mbg3_array = np.array(
            readt.readND(PATH_TO_GLOBAL + f"/External/Mbg_3/Mbg3_nf{nf}.txt")
        )
        Mbg3.append(interp2d(ZList, QList, Mbg3_array.T, kind="quintic"))


def InitializeMbq_3(nf_list):
    """Initialize the Mbg at N3LO matching condition from the file in External."""
    global Mbq3
    Mbq3 = []
    append_empty(Mbq3, nf_list)
    for nf in nf_list:
        Mbq3_array = np.array(
            readt.readND(PATH_TO_GLOBAL + f"/External/Mbq_3/Mbq3_nf{nf}.txt")
        )
        Mbq3.append(interp2d(ZList, QList, Mbq3_array.T, kind="quintic"))


def InitializeCq3_m(nf_list):
    """Initialize the Cq2 at N3LO massive function from the file in External."""
    global Cq3m
    Cq3m = []
    append_empty(Cq3m, nf_list)
    for nf in nf_list:
        Cq3m_array = np.array(
            readt.readND(PATH_TO_GLOBAL + f"/External/Cq_3_m/C2q_nf{nf}.txt")
        )
        Cq3m.append(interp2d(ZList[:-1], QList, Cq3m_array[:, :-1], kind="quintic"))


def InitializeCq3_til(nf_list):
    """Initialize the Cq at N3LO tilde function from the file in External."""
    global Cq3_til
    Cq3_til = []
    append_empty(Cq3_til, nf_list)
    for nf in nf_list:
        Cq3_til_array = np.array(
            readt.readND(PATH_TO_GLOBAL + f"/External/Cq_3_til/Cq3til_nf{nf}.txt")
        )
        Cq3_til.append(interp2d(ZList[:-1], QList, Cq3_til_array.T, kind="quintic"))


def InitializeCLq3_m(nf_list):
    """Initialize the CqL at N3LO massive function from the file in External."""
    global CLq3m
    CLq3m = []
    append_empty(CLq3m, nf_list)
    for nf in nf_list:
        CLq3m_array = np.array(
            readt.readND(PATH_TO_GLOBAL + f"/External/CLq_3_m/CLq_nf{nf}.txt")
        )
        CLq3m.append(interp2d(ZList, QList, CLq3m_array, kind="quintic"))


def InitializeCLq3_til(nf_list):
    """Initialize the CLq at N3LO tilde function from the file in External."""
    global CLq3_til
    CLq3_til = []
    append_empty(CLq3_til, nf_list)
    for nf in nf_list:
        CLq3_til_array = np.array(
            readt.readND(PATH_TO_GLOBAL + f"/External/CLq_3_til/CLq3til_nf{nf}.txt")
        )
        CLq3_til.append(interp2d(ZList[:-1], QList, CLq3_til_array.T, kind="quintic"))


def InitializeCg3_m(nf_list):
    """Initialize the Cg2 at N3LO massive function from the file in External."""
    global Cg3m
    Cg3m = []
    append_empty(Cg3m, nf_list)
    for nf in nf_list:
        Cg3m_array = np.array(
            readt.readND(PATH_TO_GLOBAL + f"/External/Cg_3_m/C2g_nf{nf}.txt")
        )
        Cg3m.append(interp2d(ZList[:-1], QList, Cg3m_array[:, :-1], kind="quintic"))


def InitializeCLg3_m(nf_list):
    """Initialize the CgL at N3LO massive function from the file in External."""
    global CLg3m
    CLg3m = []
    append_empty(CLg3m, nf_list)
    for nf in nf_list:
        CLg3m_array = np.array(
            readt.readND(PATH_TO_GLOBAL + f"/External/CLg_3_m/CLg_nf{nf}.txt")
        )
        CLg3m.append(interp2d(ZList, QList, CLg3m_array, kind="quintic"))


def InitializeCg3_til(nf_list):
    """Initialize the C2g at N3LO tilde function from the file in External."""
    global Cg3_til
    Cg3_til = []
    append_empty(Cg3_til, nf_list)
    for nf in nf_list:
        Cg3_til_array = np.array(
            readt.readND(PATH_TO_GLOBAL + f"/External/Cg_3_til/Cg3til_nf{nf}.txt")
        )
        Cg3_til.append(interp2d(ZList[:-1], QList, Cg3_til_array.T, kind="quintic"))


def InitializeCLg3_til(nf_list):
    """Initialize the CLg at N3LO tilde function from the file in External."""
    global CLg3_til
    CLg3_til = []
    append_empty(CLg3_til, nf_list)
    for nf in nf_list:
        CLg3_til_array = np.array(
            readt.readND(PATH_TO_GLOBAL + f"/External/CLg_3_til/CLg3til_nf{nf}.txt")
        )
        CLg3_til.append(interp2d(ZList[:-1], QList, CLg3_til_array.T, kind="quintic"))


def Initialize_all(nf):
    """Initialize all the needed global lists."""
    # TODO: allow also for Ftop
    nf_list = [4, 5] if nf is None else [nf]
    console.log("[green]Loading precomputed grids ...")
    InitializeQX()
    InitializeMbg2(nf_list)
    InitializeMbq2(nf_list)
    InitializeMbg_3(nf_list)
    InitializeMbq_3(nf_list)
    InitializeCg2_til(nf_list)
    InitializeCq2_til(nf_list)
    InitializeCLg2_til(nf_list)
    InitializeCLq2_til(nf_list)
    InitializeCq3_m(nf_list)
    InitializeCLq3_m(nf_list)
    InitializeCg3_m(nf_list)
    InitializeCLg3_m(nf_list)
    InitializeCLq3_til(nf_list)
    InitializeCLg3_til(nf_list)
    InitializeCq3_til(nf_list)
    InitializeCg3_til(nf_list)


def InitializeHPL():
    global HPL_x_array
    global HPL_0011
    HPL_x_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_x.txt")
    HPL_0011_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_0011.txt")
    HPL_0011 = interp1d(HPL_x_array, HPL_0011_array)
    global HPL_00011
    HPL_00011_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_00011.txt")
    HPL_00011 = interp1d(HPL_x_array, HPL_00011_array)
    global HPL_00101
    HPL_00101_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_00101.txt")
    HPL_00101 = interp1d(HPL_x_array, HPL_00101_array)
    global HPL_00111
    HPL_00111_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_00111.txt")
    HPL_00111 = interp1d(HPL_x_array, HPL_00111_array)
    global HPL_01011
    HPL_01011_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_01011.txt")
    HPL_01011 = interp1d(HPL_x_array, HPL_01011_array)
    global HPL_0m1m1m1
    HPL_0m1m1m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_0m1m1m1.txt")
    HPL_0m1m1m1 = interp1d(HPL_x_array, HPL_0m1m1m1_array)
    global HPL_0m101
    HPL_0m101_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_0m101.txt")
    HPL_0m101 = interp1d(HPL_x_array, HPL_0m101_array)
    global HPL_00m1m1
    HPL_00m1m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_00m1m1.txt")
    HPL_00m1m1 = interp1d(HPL_x_array, HPL_00m1m1_array)
    global HPL_00m11
    HPL_00m11_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_00m11.txt")
    HPL_00m11 = interp1d(HPL_x_array, HPL_00m11_array)
    global HPL_001m1
    HPL_001m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_001m1.txt")
    HPL_001m1 = interp1d(HPL_x_array, HPL_001m1_array)
    global HPL_0m10m1m1
    HPL_0m10m1m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_0m10m1m1.txt")
    HPL_0m10m1m1 = interp1d(HPL_x_array, HPL_0m10m1m1_array)
    global HPL_00m1m1m1
    HPL_00m1m1m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_00m1m1m1.txt")
    HPL_00m1m1m1 = interp1d(HPL_x_array, HPL_00m1m1m1_array)
    global HPL_00m10m1
    HPL_00m10m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_00m10m1.txt")
    HPL_00m10m1 = interp1d(HPL_x_array, HPL_00m10m1_array)
    global HPL_00m101
    HPL_00m101_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_00m101.txt")
    HPL_00m101 = interp1d(HPL_x_array, HPL_00m101_array)
    global HPL_000m1m1
    HPL_000m1m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_000m1m1.txt")
    HPL_000m1m1 = interp1d(HPL_x_array, HPL_000m1m1_array)
    global HPL_000m11
    HPL_000m11_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_000m11.txt")
    HPL_000m11 = interp1d(HPL_x_array, HPL_000m11_array)
    global HPL_0001m1
    HPL_0001m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_0001m1.txt")
    HPL_0001m1 = interp1d(HPL_x_array, HPL_0001m1_array)
    global HPL_0010m1
    HPL_0010m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_0010m1.txt")
    HPL_0010m1 = interp1d(HPL_x_array, HPL_0010m1_array)
    global HPL_0m1m11
    HPL_0m1m11_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_0m1m11.txt")
    HPL_0m1m11 = interp1d(HPL_x_array, HPL_0m1m11_array)
    global HPL_0m11m1
    HPL_0m11m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_0m11m1.txt")
    HPL_0m11m1 = interp1d(HPL_x_array, HPL_0m11m1_array)
    global HPL_01m1m1
    HPL_01m1m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_01m1m1.txt")
    HPL_01m1m1 = interp1d(HPL_x_array, HPL_01m1m1_array)
    global HPL_0m111
    HPL_0m111_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_0m111.txt")
    HPL_0m111 = interp1d(HPL_x_array, HPL_0m111_array)
    global HPL_01m11
    HPL_01m11_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_01m11.txt")
    HPL_01m11 = interp1d(HPL_x_array, HPL_01m11_array)
    global HPL_011m1
    HPL_011m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_011m1.txt")
    HPL_011m1 = interp1d(HPL_x_array, HPL_011m1_array)
    global HPL_0m1011
    HPL_0m1011_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_0m1011.txt")
    HPL_0m1011 = interp1d(HPL_x_array, HPL_0m1011_array)
    global HPL_0m1m101
    HPL_0m1m101_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_0m1m101.txt")
    HPL_0m1m101 = interp1d(HPL_x_array, HPL_0m1m101_array)
    global HPL_0m1m11m1
    HPL_0m1m11m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_0m1m11m1.txt")
    HPL_0m1m11m1 = interp1d(HPL_x_array, HPL_0m1m11m1_array)
    global HPL_0m10m11
    HPL_0m10m11_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_0m10m11.txt")
    HPL_0m10m11 = interp1d(HPL_x_array, HPL_0m10m11_array)
    global HPL_0m101m1
    HPL_0m101m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_0m101m1.txt")
    HPL_0m101m1 = interp1d(HPL_x_array, HPL_0m101m1_array)
    global HPL_0m11m1m1
    HPL_0m11m1m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_0m11m1m1.txt")
    HPL_0m11m1m1 = interp1d(HPL_x_array, HPL_0m11m1m1_array)
    global HPL_00m1m11
    HPL_00m1m11_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_00m1m11.txt")
    HPL_00m1m11 = interp1d(HPL_x_array, HPL_00m1m11_array)
    global HPL_00m11m1
    HPL_00m11m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_00m11m1.txt")
    HPL_00m11m1 = interp1d(HPL_x_array, HPL_00m11m1_array)
    global HPL_00m111
    HPL_00m111_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_00m111.txt")
    HPL_00m111 = interp1d(HPL_x_array, HPL_00m111_array)
    global HPL_001m1m1
    HPL_001m1m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_001m1m1.txt")
    HPL_001m1m1 = interp1d(HPL_x_array, HPL_001m1m1_array)
    global HPL_001m11
    HPL_001m11_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_001m11.txt")
    HPL_001m11 = interp1d(HPL_x_array, HPL_001m11_array)
    global HPL_0011m1
    HPL_0011m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_0011m1.txt")
    HPL_0011m1 = interp1d(HPL_x_array, HPL_0011m1_array)
    global HPL_01m1m1m1
    HPL_01m1m1m1_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_01m1m1m1.txt")
    HPL_01m1m1m1 = interp1d(HPL_x_array, HPL_01m1m1m1_array)
    global HPL_0m1m1m1m1
    HPL_0m1m1m1m1_array = readt.read1D(
        PATH_TO_GLOBAL + "/External/HPL/HPL_0m1m1m1m1.txt"
    )
    HPL_0m1m1m1m1 = interp1d(HPL_x_array, HPL_0m1m1m1m1_array)
    global HPL_0m1m1m11
    HPL_0m1m1m11_array = readt.read1D(PATH_TO_GLOBAL + "/External/HPL/HPL_0m1m1m11.txt")
    HPL_0m1m1m11 = interp1d(HPL_x_array, HPL_0m1m1m11_array)
