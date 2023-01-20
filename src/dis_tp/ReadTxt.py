# These are utility functions to read txt grids.
import numpy as np
from multiprocessing import Pool

from . import Initialize as Ini


def read1D(path_to_file):
    """
    Read a space-separated txt file and return a 1-Dimensional list of the values

    Parameters:
        path_to_file : str
            file to open
    Returns:
            : list
        list of values
    """
    for line in open(path_to_file):
        listWord = line.split(" ")
    mylist = [float(item) for item in listWord]
    return mylist


def read1D_Nic(path_to_file):
    """
    Read a space-separated txt file and return a 1-Dimensional list of the values (for triple space separated files)

    Parameters:
        path_to_file : str
            file to open
    Returns:
            : list
        list of values
    """
    for line in open(path_to_file):
        listWord = line.split("   ")
    mylist = [float(item) for item in listWord]
    return mylist


def readND(path_to_file):
    """
    Read a space-separated txt file and return a N-Dimensional list of the values

    Input:
        path_to_file : str
            file to open
    Returns:
            : list
        list of values
    """
    mylist = []
    for line in open(path_to_file):
        listWord = line.split(" ")
        mylist.append(listWord)
    list = [
        [
            float(item)
            if item != "-nan" and item != "nan"
            else float(mylist[a][mylist[a].index(item) - 1])
            for item in mylist[a]
        ]
        for a in range(len(mylist))
    ]
    return list


def readND_Nic(path_to_file):
    """
    Read a space-separated txt file and return a N-Dimensional list of the values (for double space separated files)

    Input:
        path_to_file : str
            file to open
    Returns:
            : list
        list of values
    """
    mylist = []
    for line in open(path_to_file):
        listWord = line.split("  ")
        mylist.append(listWord)
    list = [
        [
            float(item)
            if item != "-nan" and item != "nan"
            else float(mylist[a][mylist[a].index(item) - 1])
            for item in mylist[a]
        ]
        for a in range(len(mylist))
    ]
    return list


def readND_python(path_to_file):
    """
    Read a space-separated txt file and return a N-Dimensional list of the values

    Input:
        path_to_file : str
            file to open
    Returns:
            : list
        list of values
    """
    mylist = []
    for line in open(path_to_file):
        listWord = line.split(" ")
        mylist.append(listWord)
    list = [
        [
            float(item)
            if item != "-nan" and item != "nan"
            else float(mylist[a][mylist[a].index(item)])
            for item in mylist[a]
        ]
        for a in range(len(mylist))
    ]
    return list


class Construct_Grid():

    def __init__(self, func, mass, nf, path, n_pools=8):
        self.func = func
        self.mass = mass
        self.path = path
        self.xgrid = Ini.HPL_x_array # [0.0001,0.001,0.1,1]# 
        self.qgrid = Ini.QList # [1,10,100]
        self.n_pools = n_pools
        self.nf = nf
        

    def construct_single_x(self, z):
        z_func_values = []
        p = []
        i = self.xgrid.index(z)
        print(f"Computing x = {z},  {i}/{len(self.xgrid)}")
        for q in self.qgrid:
            p = [self.mass, q]
            z_func_values.append(self.func(z, p, self.nf))
        return z_func_values


    def construct_grid_matching(self):

        args = (self.construct_single_x, self.xgrid)
        with Pool(self.n_pools) as pool:
            result = pool.map(*args)

        func_values = []
        for res in result:
            func_values.append(res)
        np.savetxt(self.path, func_values)
        return func_values


def construct_grid_tilde(func, mass, path):
    func_values = []
    p = [mass]
    for z in Ini.ZList:
        print(float(Ini.ZList.index(z)) / Ini.ZList.__len__())
        z_func_values = []
        for q in np.array(Ini.QList):
            # z_func_values.append(func(z, q, p)[0])
            z_func_values.append(func(z, q, p))
        func_values.append(z_func_values)
    np.savetxt(path, func_values)
    return func_values
